#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from muggled_sam.make_sam import make_sam_from_state_dict

from eval_image_exemplar import (
    apply_grayscale,
    apply_mask_nms,
    build_detection_record,
    build_exemplar_tokens_for_object,
    build_gt_down_list,
    collect_multi_object_samples,
    compute_mask_iou,
    generate_detections_train,
    load_bgr,
    load_instance_masks_for_object,
    pad_exemplar_batch,
    parse_image_list,
    parse_ref_view_ids,
    save_mask_triptych,
    update_pq_accumulators,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SAMv3 exemplar detection with optional per-image random single-view references.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/zhenrant/rendering_prompted_muggled_sam/sam3.pt",
        help="Path to SAMv3 checkpoint (.pt).",
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        default="/sata1/data/kevin/realworld_datasets/3d_printing_meshes/renders_2442_0316",
        help="Path to reference renders.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        nargs="+",
        default=["/sata1/data/kevin/realworld_datasets/3d_printing_dataset"],
        help="Dataset roots (space-separated, and/or comma-separated).",
    )
    parser.add_argument("--ref_view_ids", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11", help="Reference view ids to use.")
    parser.add_argument(
        "--random_single_view",
        default = False,
        action="store_true",
        help="When set, choose one random reference view from --ref_view_ids for each image.",
    )
    parser.add_argument("--max_side_length", type=int, default=1008)
    parser.add_argument("--no_square", action="store_true", help="Disable square resizing in encoder.")
    parser.add_argument("--num_points_approx", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument(
        "--sub_sample",
        type=int,
        default=1,
        help="Evaluate every Nth image in the dataset (1 = use all images).",
    )
    parser.add_argument(
        "--nms_iou",
        type=float,
        default=0.5,
        help="IoU threshold for mask NMS (<=0 disables NMS).",
    )
    parser.add_argument("--det_filter", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="outputs_eval_exemplar_random_single_view")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, choices=["fp32", "bf16"], default="")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument(
        "--vis_every",
        type=int,
        default=1,
        help="Save a debug collage every N batches (0 disables).",
    )
    parser.add_argument(
        "--image_list",
        type=str,
        default="",
        help="Comma-separated or Python list of full rgb image paths to evaluate.",
    )
    parser.add_argument("--grayscale", default=False, help="Convert all input images to grayscale.")
    parser.add_argument(
        "--multi_gt_only",
        default=False,
        help="Only evaluate samples with multiple GT instances for the target object.",
    )
    parser.add_argument("--finetune_ckpt", type=str, default="", help="Optional finetuned detector checkpoint.")
    return parser.parse_args()


def expand_dataset_roots(dataset_root_arg: object) -> List[str]:
    raw_items = [dataset_root_arg] if isinstance(dataset_root_arg, str) else list(dataset_root_arg)
    dataset_roots: List[str] = []
    for item in raw_items:
        dataset_roots.extend([part.strip() for part in str(item).split(",") if part.strip()])
    return dataset_roots


def select_ref_view_ids_for_image(ref_view_ids: List[str], random_single_view: bool) -> List[str]:
    if random_single_view:
        return [random.choice(ref_view_ids)]
    return ref_view_ids


def main() -> None:
    args = parse_args()
    dataset_roots = expand_dataset_roots(args.dataset_root)
    if not dataset_roots:
        raise ValueError("No dataset roots provided.")

    ref_view_ids = parse_ref_view_ids(args.ref_view_ids)
    if not ref_view_ids:
        raise ValueError("No reference view ids resolved.")

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if args.dtype:
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    else:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    _, base_model = make_sam_from_state_dict(args.model_path)
    base_model.to(device=device, dtype=dtype)
    detmodel = base_model.make_detector_model()
    detmodel.to(device=device, dtype=dtype)
    detmodel.eval()

    if args.finetune_ckpt:
        ckpt = torch.load(args.finetune_ckpt, map_location="cpu")
        detmodel.image_exemplar_fusion.load_state_dict(ckpt["image_exemplar_fusion"])
        detmodel.exemplar_detector.load_state_dict(ckpt["exemplar_detector"])
        detmodel.exemplar_segmentation.load_state_dict(ckpt["exemplar_segmentation"])
        print("Loaded finetuned detector weights from", args.finetune_ckpt)

    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "step_outputs")
    os.makedirs(vis_dir, exist_ok=True)

    all_entries: List[Dict[str, str]] = []
    for root in dataset_roots:
        _object_samples, cur_entries = collect_multi_object_samples(root, sub_sample=args.sub_sample)
        all_entries.extend(cur_entries)
    if not all_entries:
        raise RuntimeError("No dataset entries found.")

    unique_entries: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    for entry in all_entries:
        key = (entry["frame_id"], entry["object_id"], entry["rgb_path"], entry["inst_path"])
        if key not in unique_entries:
            unique_entries[key] = entry
    all_entries = list(unique_entries.values())

    image_list = parse_image_list(args.image_list)
    if image_list:
        all_entries = [
            entry for entry in all_entries if os.path.abspath(entry["rgb_path"]) in image_list
        ]
        if not all_entries:
            raise RuntimeError("No dataset entries matched --image_list.")

    if args.shuffle:
        random.shuffle(all_entries)

    total_entries = len(all_entries)
    total_batches_est = max(1, math.ceil(total_entries / args.batch_size))
    if args.max_batches > 0:
        total_batches_est = min(total_batches_est, args.max_batches)
    print(
        "Estimated batches:",
        total_batches_est,
        f"(entries={total_entries}, batch_size={args.batch_size})",
    )

    reference_dir = Path(args.reference_dir).expanduser().resolve()
    if not reference_dir.is_dir():
        raise FileNotFoundError(reference_dir)

    ref_cache: Dict[Tuple[str, Tuple[str, ...]], torch.Tensor] = {}
    seg_cache: Dict[str, np.ndarray] = {}
    mapping_cache: Dict[str, Dict[str, List[Tuple[int, ...]]]] = {}
    batch_step = 0

    total_iou_sum = 0.0
    total_iou_count = 0
    total_correct_count = 0
    object_iou_sum: Dict[str, float] = defaultdict(float)
    object_iou_count: Dict[str, int] = defaultdict(int)
    pq_iou_threshold = 0.5
    pq_score_thresholds = [round(0.10 + 0.01 * idx, 2) for idx in range(89)]
    pq_stats: Dict[float, Dict[str, float]] = {
        thresh: {"sum_iou": 0.0, "tp": 0, "fp": 0, "fn": 0} for thresh in pq_score_thresholds
    }

    detection_log_path = os.path.join(args.output_dir, "detection_log_tless_p2.json")
    with open(detection_log_path, "w", encoding="utf-8") as detection_log:
        with torch.no_grad():
            for start in range(0, len(all_entries), args.batch_size):
                subset = all_entries[start : start + args.batch_size]
                prepared: List[Dict[str, object]] = []
                for entry in subset:
                    obj_id = entry["object_id"]
                    try:
                        image_bgr = load_bgr(entry["rgb_path"])
                    except FileNotFoundError:
                        continue
                    if args.grayscale:
                        image_bgr = apply_grayscale(image_bgr)
                    try:
                        mapping_path = Path(entry["inst_path"]).with_name(
                            f"instance_segmentation_mapping_{entry['frame_id']}.json"
                        )
                        gt_masks = load_instance_masks_for_object(
                            entry["inst_path"],
                            str(mapping_path),
                            obj_id,
                            seg_cache=seg_cache,
                            mapping_cache=mapping_cache,
                        )
                    except FileNotFoundError:
                        continue
                    if not gt_masks:
                        continue
                    if args.multi_gt_only and len(gt_masks) < 2:
                        continue

                    selected_ref_view_ids = select_ref_view_ids_for_image(
                        ref_view_ids, args.random_single_view
                    )
                    cache_key = (obj_id, tuple(selected_ref_view_ids))
                    if cache_key not in ref_cache:
                        exemplar_ref = build_exemplar_tokens_for_object(
                            detmodel=detmodel,
                            object_id=obj_id,
                            reference_dir=reference_dir,
                            ref_view_ids=selected_ref_view_ids,
                            max_side_length=args.max_side_length,
                            use_square_sizing=not args.no_square,
                            num_points_approx=args.num_points_approx,
                            device=device,
                            grayscale=args.grayscale,
                        )
                        if exemplar_ref is None:
                            continue
                        ref_cache[cache_key] = exemplar_ref.detach().cpu()

                    exemplar_ref = ref_cache[cache_key]
                    prepared.append(
                        {
                            "object_id": obj_id,
                            "frame_id": entry["frame_id"],
                            "rgb_path": entry["rgb_path"],
                            "image_bgr": image_bgr,
                            "gt_masks": gt_masks,
                            "exemplar_ref": exemplar_ref,
                            "selected_ref_view_ids": selected_ref_view_ids,
                        }
                    )

                if not prepared:
                    continue
                vis_target_idx = None
                if args.vis_every > 0 and (batch_step % args.vis_every) == 0:
                    vis_target_idx = random.randrange(len(prepared))

                group_map: Dict[Tuple[int, int], List[int]] = defaultdict(list)
                for idx, entry in enumerate(prepared):
                    img_t = detmodel.image_encoder.prepare_image(
                        entry["image_bgr"],
                        max_side_length=args.max_side_length,
                        use_square_sizing=not args.no_square,
                    )
                    entry["img_tensor"] = img_t
                    entry["preencode_hw"] = img_t.shape[2:]
                    shape_key = (img_t.shape[2], img_t.shape[3])
                    group_map[shape_key].append(idx)

                batch_ious: List[float] = []
                batch_correct = 0
                for _, idxs in group_map.items():
                    img_batch = torch.cat([prepared[i]["img_tensor"] for i in idxs], dim=0)
                    t0 = time.time()
                    encoded_img = detmodel.image_encoder(img_batch)
                    encoded_image_features_list = detmodel.image_projection.v3_projection(encoded_img)
                    t1 = time.time()
                    exemplars_list = [prepared[i]["exemplar_ref"] for i in idxs]
                    exemplar_batch, padding_mask = pad_exemplar_batch(exemplars_list, device=device)

                    mask_preds, box_preds, det_scores, _ = generate_detections_train(
                        detmodel,
                        encoded_image_features_list,
                        exemplar_batch,
                        detection_filter_threshold=args.det_filter,
                        exemplar_padding_mask_bn=padding_mask,
                    )
                    t2 = time.time()
                    display_step = batch_step + 1
                    print(
                        "step {}/{} Batch encoding time: {:.3f}s, detection time: {:.3f}s".format(
                            display_step,
                            total_batches_est,
                            t1 - t0,
                            t2 - t1,
                        )
                    )
                    if mask_preds.shape[1] == 0:
                        for data_idx in idxs:
                            preencode_hw = prepared[data_idx]["preencode_hw"]
                            gt_down_list = build_gt_down_list(
                                prepared[data_idx]["gt_masks"],
                                preencode_hw,
                                mask_preds.shape[-2:],
                                device,
                            )
                            update_pq_accumulators(pq_stats, [], gt_down_list, None, pq_iou_threshold)
                        continue

                    for local_idx, data_idx in enumerate(idxs):
                        preencode_hw = prepared[data_idx]["preencode_hw"]

                        scores = det_scores[local_idx]
                        if scores.numel() == 0:
                            gt_down_list = build_gt_down_list(
                                prepared[data_idx]["gt_masks"],
                                preencode_hw,
                                mask_preds.shape[-2:],
                                device,
                            )
                            update_pq_accumulators(pq_stats, [], gt_down_list, None, pq_iou_threshold)
                            continue

                        boxes_nms, masks_nms, scores_nms = apply_mask_nms(
                            box_preds[local_idx],
                            mask_preds[local_idx],
                            scores,
                            iou_threshold=args.nms_iou,
                        )
                        if scores_nms.numel() == 0:
                            gt_down_list = build_gt_down_list(
                                prepared[data_idx]["gt_masks"],
                                preencode_hw,
                                mask_preds.shape[-2:],
                                device,
                            )
                            update_pq_accumulators(pq_stats, [], gt_down_list, None, pq_iou_threshold)
                            continue

                        gt_down_list = build_gt_down_list(
                            prepared[data_idx]["gt_masks"],
                            preencode_hw,
                            mask_preds.shape[-2:],
                            device,
                        )

                        pred_masks_list = [(masks_nms[k] > 0) for k in range(masks_nms.shape[0])]
                        update_pq_accumulators(
                            pq_stats,
                            pred_masks_list,
                            gt_down_list,
                            pred_scores=scores_nms,
                            iou_threshold=pq_iou_threshold,
                        )

                        record = build_detection_record(
                            prepared[data_idx]["object_id"],
                            prepared[data_idx]["frame_id"],
                            scores_nms,
                            pred_masks_list,
                            gt_down_list,
                            iou_threshold=pq_iou_threshold,
                            top_k=5,
                        )
                        detection_log.write(json.dumps(record) + "\n")

                        best_iou = 0.0
                        for gt_down in gt_down_list:
                            iou = compute_mask_iou(masks_nms[0], gt_down)
                            best_iou = max(best_iou, float(iou.item()))
                        batch_ious.append(best_iou)
                        total_iou_sum += best_iou
                        total_iou_count += 1
                        if best_iou > 0.5:
                            batch_correct += 1
                            total_correct_count += 1
                        obj_id = prepared[data_idx]["object_id"]
                        object_iou_sum[obj_id] += best_iou
                        object_iou_count[obj_id] += 1

                        if vis_target_idx is not None and data_idx == vis_target_idx:
                            out_path = os.path.join(vis_dir, f"step_{batch_step:06d}.png")
                            save_mask_triptych(
                                prepared[data_idx]["image_bgr"],
                                masks_nms,
                                scores_nms,
                                prepared[data_idx]["gt_masks"],
                                object_id=prepared[data_idx]["object_id"],
                                reference_dir=reference_dir,
                                ref_view_ids=prepared[data_idx]["selected_ref_view_ids"],
                                image_name=prepared[data_idx]["rgb_path"],
                                output_path=out_path,
                            )

                if batch_ious:
                    avg_iou = sum(batch_ious) / max(1, len(batch_ious))
                    correct_rate = batch_correct / max(1, len(batch_ious))
                    display_step = batch_step + 1
                    print(
                        f"step {display_step}/{total_batches_est} avg_iou={avg_iou:.4f} "
                        f"correct_rate={correct_rate:.3f} samples={len(batch_ious)}"
                    )
                batch_step += 1

                if args.max_batches > 0 and batch_step >= args.max_batches:
                    break

    if total_iou_count > 0:
        overall_avg = total_iou_sum / total_iou_count
        overall_correct = total_correct_count / total_iou_count
        print(
            f"overall_avg_iou={overall_avg:.4f} "
            f"correct_rate={overall_correct:.3f} samples={total_iou_count}"
        )
    if object_iou_count:
        print("per_object_iou:")
        for obj_id in sorted(object_iou_count.keys()):
            count = object_iou_count[obj_id]
            avg_iou = object_iou_sum[obj_id] / max(1, count)
            print(f"  {obj_id}: avg_iou={avg_iou:.4f} samples={count}")
    for score_threshold in sorted(pq_stats.keys()):
        stats = pq_stats[score_threshold]
        denom = stats["tp"] + 0.5 * stats["fp"] + 0.5 * stats["fn"]
        pq = stats["sum_iou"] / denom if denom > 0 else 0.0
        print(
            f"PQ@score>={score_threshold:.2f}={pq:.4f} "
            f"tp={int(stats['tp'])} fp={int(stats['fp'])} fn={int(stats['fn'])}"
        )


if __name__ == "__main__":
    main()
