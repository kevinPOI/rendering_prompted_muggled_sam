#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from muggled_sam.make_sam import make_sam_from_state_dict

from eval_image_exemplar import (
    apply_mask_nms,
    build_exemplar_tokens_for_object,
    encode_detection_image_no_infer,
    generate_detections_train,
    parse_ref_view_ids,
)


# OBJECT_LIST: List[Tuple[str, str, float]] = [
#     ("c_clamp", "c_clamp", 0.15),
#     ("flat_plate", "stepper_mount", 0.18),
#     ("gear", "gear", 0.15),
# ]
OBJECT_LIST: List[Tuple[str, str, float]] = [
    # ("bv2_disc", "cutting_blade", 0.15),
    
    # ("stepper_mount", "stepper_mount", 0.18),
    ("square_tube", "square_tube", 0.15),
    ("bolt", "bolt", 0.16),
]
MASK_SHIFT: Tuple[int, int] = (-5, 0)

DEFAULT_VIDEO_PATH = "demo5.mp4"
DEFAULT_MODEL_PATH = "/home/zhenrant/rendering_prompted_muggled_sam/sam3.pt"
DEFAULT_REFERENCE_DIR = "/sata1/data/kevin/realworld_datasets/3d_printing_meshes/renders_2442_0316"
DEFAULT_FINETUNE_CKPT = "finetune_exemplar/run_20260322_172059/finetune_epoch_034.pth"

COLOR_PALETTE: List[Tuple[int, int, int]] = [
    (48, 140, 255),
    (80, 200, 120),
    (255, 170, 40),
    (210, 90, 255),
    (90, 220, 240),
    (255, 110, 110),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exemplar-based detection on every video frame and save overlays.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to SAMv3 checkpoint (.pt).")
    parser.add_argument("--video_path", type=str, default=DEFAULT_VIDEO_PATH, help="Input video path.")
    parser.add_argument("--output_path", type=str, default="", help="Output video path. Defaults next to input video.")
    parser.add_argument(
        "--no_save_frames",
        action="store_true",
        help="Disable saving individual rendered frames. Saving frames is enabled by default.",
    )
    parser.add_argument("--reference_dir", type=str, default=DEFAULT_REFERENCE_DIR, help="Path to reference renders.")
    parser.add_argument("--ref_view_ids", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--max_side_length", type=int, default=1008)
    parser.add_argument("--no_square", action="store_true", help="Disable square resizing in encoder.")
    parser.add_argument("--num_points_approx", type=int, default=24)
    parser.add_argument(
        "--first_n_frames",
        type=int,
        default=0,
        help="If > 0, only process the first N frames of the input video.",
    )
    parser.add_argument("--nms_iou", type=float, default=0.5, help="Mask NMS IoU threshold.")
    parser.add_argument("--mask_alpha", type=float, default=0.4, help="Mask overlay alpha.")
    parser.add_argument(
        "--hide_scores",
        default=True,
        action="store_true",
        help="Hide confidence scores in per-detection box labels. Scores are shown by default.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, choices=["fp32", "bf16"], default="")
    parser.add_argument(
        "--finetune_ckpt",
        type=str,
        default=DEFAULT_FINETUNE_CKPT,
        help="Optional finetuned detector checkpoint.",
    )
    parser.add_argument("--grayscale", action="store_true", help="Convert reference images to grayscale.")
    return parser.parse_args()


def resolve_output_path(video_path: str, output_path: str) -> str:
    if output_path:
        return output_path
    video = Path(video_path)
    return str(video.with_name(f"{video.stem}_mask_overlay_bbox.mp4"))


def resolve_frame_output_dir(video_path: str, output_path: str) -> Path:
    base_path = Path(output_path) if output_path else Path(video_path)
    return base_path.with_name(f"{base_path.stem}_overlayed_frames")


def pick_device_and_dtype(device_arg: str, dtype_arg: str) -> Tuple[torch.device, torch.dtype]:
    device = torch.device(device_arg if device_arg else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if dtype_arg:
        dtype = torch.bfloat16 if dtype_arg == "bf16" else torch.float32
    else:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return device, dtype


def clamp_box_xyxy(box_xyxy_px: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int]:
    x0 = int(np.clip(box_xyxy_px[0], 0, max(0, width - 1)))
    y0 = int(np.clip(box_xyxy_px[1], 0, max(0, height - 1)))
    x1 = int(np.clip(box_xyxy_px[2], 0, max(0, width - 1)))
    y1 = int(np.clip(box_xyxy_px[3], 0, max(0, height - 1)))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def mask_to_box_xyxy(mask_bool: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if not np.any(mask_bool):
        return None

    mask_uint8 = mask_bool.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels <= 1:
        ys, xs = np.where(mask_bool)
    else:
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        main_mask = labels == largest_label
        ys, xs = np.where(main_mask)

    if ys.size == 0 or xs.size == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max())
    y1 = int(ys.max())
    return x0, y0, x1, y1


def overlay_mask(image_bgr: np.ndarray, mask_bool: np.ndarray, color_bgr: Tuple[int, int, int], alpha: float) -> None:
    if not np.any(mask_bool):
        return
    color_arr = np.asarray(color_bgr, dtype=np.float32)
    image_region = image_bgr[mask_bool].astype(np.float32)
    blended = image_region * (1.0 - alpha) + color_arr * alpha
    image_bgr[mask_bool] = blended.astype(np.uint8)


def shift_mask(mask_bool: np.ndarray, shift_xy: Tuple[int, int]) -> np.ndarray:
    shift_x, shift_y = shift_xy
    if shift_x == 0 and shift_y == 0:
        return mask_bool

    shifted = np.zeros_like(mask_bool)
    src_h, src_w = mask_bool.shape

    src_x0 = max(0, -shift_x)
    src_x1 = min(src_w, src_w - shift_x) if shift_x >= 0 else src_w
    dst_x0 = max(0, shift_x)
    dst_x1 = min(src_w, src_w + shift_x) if shift_x < 0 else src_w

    src_y0 = max(0, -shift_y)
    src_y1 = min(src_h, src_h - shift_y) if shift_y >= 0 else src_h
    dst_y0 = max(0, shift_y)
    dst_y1 = min(src_h, src_h + shift_y) if shift_y < 0 else src_h

    if src_x0 >= src_x1 or src_y0 >= src_y1 or dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return shifted

    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = mask_bool[src_y0:src_y1, src_x0:src_x1]
    return shifted


def draw_label(image_bgr: np.ndarray, label: str, x0: int, y0: int, color_bgr: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.3
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    box_y0 = max(0, y0)
    box_y1 = min(image_bgr.shape[0] - 1, box_y0 + text_h + baseline + 12)
    box_x0 = max(0, x0)
    box_x1 = min(image_bgr.shape[1] - 1, box_x0 + text_w + 14)
    cv2.rectangle(image_bgr, (box_x0, box_y0), (box_x1, box_y1), color_bgr, thickness=-1)
    cv2.putText(
        image_bgr,
        label,
        (box_x0 + 7, box_y1 - baseline - 5),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_stationary_labels(image_bgr: np.ndarray, detections: Sequence[Dict[str, object]]) -> None:
    seen = set()
    unique_detections: List[Dict[str, object]] = []
    for det in detections:
        label = det["label"]
        if label in seen:
            continue
        seen.add(label)
        unique_detections.append(det)

    start_x = 16
    start_y = 16
    row_gap = 10
    current_y = start_y
    for det in unique_detections:
        draw_label(image_bgr, det["label"], start_x, current_y, det["color_bgr"])
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 3
        (_, text_h), baseline = cv2.getTextSize(det["label"], font, font_scale, thickness)
        current_y += text_h + baseline + 12 + int(round(1.4 * text_h))


def render_detections(
    frame_bgr: np.ndarray,
    detections: Sequence[Dict[str, object]],
    mask_alpha: float,
    show_scores: bool,
) -> np.ndarray:
    rendered = frame_bgr.copy()
    for det in detections:
        overlay_mask(rendered, det["mask_bool"], det["color_bgr"], mask_alpha)
    for det in detections:
        x0, y0, x1, y1 = det["box_xyxy"]
        cv2.rectangle(rendered, (x0, y0), (x1, y1), det["color_bgr"], thickness=2)
        label_text = det["label"]
        if show_scores:
            label_text = f"{label_text}: {det['score']:.2f}"
        draw_label(rendered, label_text, x0, max(0, y0 - 40), det["color_bgr"])
    return rendered


def build_object_exemplars(
    detmodel,
    object_list: Sequence[Tuple[str, str, float]],
    reference_dir: Path,
    ref_view_ids: List[str],
    max_side_length: int,
    use_square_sizing: bool,
    num_points_approx: int,
    device: torch.device,
    grayscale: bool,
) -> List[Dict[str, object]]:
    prepared: List[Dict[str, object]] = []
    for idx, (object_id, display_name, threshold) in enumerate(object_list):
        exemplar_tokens = build_exemplar_tokens_for_object(
            detmodel,
            object_id=object_id,
            reference_dir=reference_dir,
            ref_view_ids=ref_view_ids,
            max_side_length=max_side_length,
            use_square_sizing=use_square_sizing,
            num_points_approx=num_points_approx,
            device=device,
            grayscale=grayscale,
        )
        if exemplar_tokens is None:
            print(f"Skipping {object_id}: no reference exemplars could be built")
            continue
        prepared.append(
            {
                "object_id": object_id,
                "label": display_name,
                "threshold": float(threshold),
                "color_bgr": COLOR_PALETTE[idx % len(COLOR_PALETTE)],
                "exemplar_tokens": exemplar_tokens.detach().cpu(),
            }
        )
    return prepared


def main() -> None:
    args = parse_args()
    ref_view_ids = parse_ref_view_ids(args.ref_view_ids)
    if not ref_view_ids:
        raise ValueError("No reference view ids resolved.")

    output_path = resolve_output_path(args.video_path, args.output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_frames = not args.no_save_frames
    frame_output_dir = resolve_frame_output_dir(args.video_path, output_path)
    if save_frames:
        frame_output_dir.mkdir(parents=True, exist_ok=True)

    device, dtype = pick_device_and_dtype(args.device, args.dtype)
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

    object_entries = build_object_exemplars(
        detmodel=detmodel,
        object_list=OBJECT_LIST,
        reference_dir=Path(args.reference_dir),
        ref_view_ids=ref_view_ids,
        max_side_length=args.max_side_length,
        use_square_sizing=not args.no_square,
        num_points_approx=args.num_points_approx,
        device=device,
        grayscale=args.grayscale,
    )
    if not object_entries:
        raise RuntimeError("No valid object exemplars were built from the configured OBJECT_LIST.")

    video_reader = cv2.VideoCapture(args.video_path)
    if not video_reader.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video_path}")

    fps = video_reader.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    total_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {args.video_path}")
    print(f"Saving output to: {output_path}")
    if save_frames:
        print(f"Saving frames to: {frame_output_dir}")
    print(f"Objects: {[(entry['object_id'], entry['label'], entry['threshold']) for entry in object_entries]}")

    frame_idx = 0
    video_writer = None
    try:
        with torch.inference_mode():
            while True:
                if args.first_n_frames > 0 and frame_idx >= args.first_n_frames:
                    break
                ok, frame_bgr = video_reader.read()
                if not ok:
                    break
                frame_idx += 1
                frame_height, frame_width = frame_bgr.shape[:2]

                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                    if not video_writer.isOpened():
                        raise RuntimeError(f"Could not create output video: {output_path}")

                encoded_imgs, _, _ = encode_detection_image_no_infer(
                    detmodel,
                    frame_bgr,
                    max_side_length=args.max_side_length,
                    use_square_sizing=not args.no_square,
                )

                detections: List[Dict[str, object]] = []
                for entry in object_entries:
                    exemplar_tokens = entry["exemplar_tokens"].to(device=device, dtype=encoded_imgs[0].dtype)
                    mask_preds, box_preds, det_scores, _ = generate_detections_train(
                        detmodel,
                        encoded_imgs,
                        exemplar_tokens,
                        detection_filter_threshold=0.0,
                        exemplar_padding_mask_bn=None,
                    )

                    obj_scores = det_scores[0]
                    keep = obj_scores >= entry["threshold"]
                    if int(keep.sum().item()) == 0:
                        continue

                    obj_boxes = box_preds[0][keep]
                    obj_masks = mask_preds[0][keep]
                    obj_scores = obj_scores[keep]
                    obj_boxes, obj_masks, obj_scores = apply_mask_nms(
                        obj_boxes,
                        obj_masks,
                        obj_scores,
                        iou_threshold=args.nms_iou,
                    )

                    for det_idx in range(obj_scores.shape[0]):
                        mask_logits = obj_masks[det_idx].detach().float().unsqueeze(0).unsqueeze(0)
                        mask_fullres = torch.nn.functional.interpolate(
                            mask_logits,
                            size=(frame_height, frame_width),
                            mode="bilinear",
                            align_corners=False,
                        )[0, 0]
                        mask_bool = (mask_fullres > 0).cpu().numpy()
                        mask_bool = shift_mask(mask_bool, MASK_SHIFT)
                        if not np.any(mask_bool):
                            continue

                        box_xyxy = mask_to_box_xyxy(mask_bool)
                        if box_xyxy is None:
                            continue
                        detections.append(
                            {
                                "label": entry["label"],
                                "score": float(obj_scores[det_idx].item()),
                                "color_bgr": entry["color_bgr"],
                                "mask_bool": mask_bool,
                                "box_xyxy": box_xyxy,
                            }
                        )

                rendered = render_detections(
                    frame_bgr,
                    detections,
                    mask_alpha=args.mask_alpha,
                    show_scores=not args.hide_scores,
                )
                video_writer.write(rendered)
                if save_frames:
                    frame_path = frame_output_dir / f"frame_{frame_idx:06d}.png"
                    cv2.imwrite(str(frame_path), rendered)

                if frame_idx == 1 or frame_idx % 10 == 0:
                    if total_frames > 0:
                        print(f"Processed frame {frame_idx}/{total_frames}")
                    else:
                        print(f"Processed frame {frame_idx}")
    finally:
        video_reader.release()
        if video_writer is not None:
            video_writer.release()

    print(f"Done. Wrote {frame_idx} frames to {output_path}")


if __name__ == "__main__":
    main()
