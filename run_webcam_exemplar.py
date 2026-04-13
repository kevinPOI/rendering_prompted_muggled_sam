#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from muggled_sam.make_sam import make_sam_from_state_dict

from eval_image_exemplar import (
    _mask_bbox,
    apply_grayscale,
    apply_mask_nms,
    build_exemplar_tokens_for_object,
    generate_detections_train,
    pad_exemplar_batch,
    parse_ref_view_ids,
)


PALETTE_BGR: List[Tuple[int, int, int]] = [
    (0, 220, 0),
    (0, 180, 255),
    (255, 180, 0),
    (180, 0, 255),
    (0, 255, 255),
    (255, 0, 180),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-time exemplar detection on a webcam stream.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/kevin/ICL/rendering_prompted_muggled_sam/model_weights/sam3.pt",
        help="Path to SAMv3 checkpoint (.pt).",
    )
    parser.add_argument(
        "--finetune_ckpt",
        type=str,
        default="/home/kevin/ICL/rendering_prompted_muggled_sam/model_weights/0321_k12_b156_resume_from_preprinte18_s1_e34.pth",
        help="Optional finetuned detector checkpoint.",
    )
    parser.add_argument(
        "--reference_dir",
        default="/home/kevin/ICL/rendering_prompted_muggled_sam/assets/renders_2442_0316",
        type=str,
        help="Directory containing reference images and masks.",
    )
    parser.add_argument(
        "--object_id",
        type=str,
        default = "lego_brick",
        help="Object name/id to load from the reference directory.",
    )
    parser.add_argument(
        "--ref_view_ids",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9,10,11",
        help="Reference view ids to use.",
    )
    parser.add_argument(
        "--max_side_length",
        type=int,
        default=1008,
        help="Image encoder max side length.",
    )
    parser.add_argument(
        "--no_square",
        action="store_true",
        help="Disable square image sizing.",
    )
    parser.add_argument(
        "--num_points_approx",
        type=int,
        default=24,
        help="Approx number of mask points to sample per reference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device string, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "bf16"],
        default="",
        help="Override model dtype.",
    )
    parser.add_argument(
        "--det_filter",
        type=float,
        default=0.0,
        help="Detection score threshold filter.",
    )
    parser.add_argument(
        "--nms_iou",
        type=float,
        default=0.5,
        help="IoU threshold for mask NMS (<=0 disables NMS).",
    )
    parser.add_argument(
        "--max_show",
        type=int,
        default=3,
        help="Max number of masks to visualize.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay alpha for masks.",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Convert all input images to grayscale.",
    )
    parser.add_argument(
        "--cam_index",
        type=int,
        default=0,
        help="Webcam index.",
    )
    parser.add_argument("--width", type=int, default=0, help="Webcam capture width.")
    parser.add_argument("--height", type=int, default=0, help="Webcam capture height.")
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Horizontally flip the webcam image.",
    )
    return parser.parse_args()


def _draw_overlays(
    frame_bgr: np.ndarray,
    masks_nhw: torch.Tensor,
    scores_n: torch.Tensor,
    alpha: float,
    max_show: int,
) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    if masks_nhw.shape[0] == 0:
        return out
    num_show = min(max_show, masks_nhw.shape[0])
    masks_cpu = masks_nhw[:num_show].detach().float().cpu().numpy()
    scores_cpu = scores_n[:num_show].detach().float().cpu().numpy()

    for i in range(num_show):
        mask = masks_cpu[i] > 0
        if mask.max() <= 0:
            continue
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        color = PALETTE_BGR[i % len(PALETTE_BGR)]
        overlay = out.copy()
        overlay[mask_resized] = color
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
        bbox = _mask_bbox(mask_resized.astype(np.uint8))
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
            score_val = float(scores_cpu[i]) if i < scores_cpu.size else 0.0
            cv2.putText(
                out,
                f"{score_val:.2f}",
                (x0, max(0, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
    return out


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if args.dtype:
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    else:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    reference_dir = Path(args.reference_dir).expanduser().resolve()
    if not reference_dir.is_dir():
        raise FileNotFoundError(reference_dir)

    ref_view_ids = parse_ref_view_ids(args.ref_view_ids)
    if not ref_view_ids:
        raise ValueError("No reference view ids resolved.")

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

    exemplar_ref = build_exemplar_tokens_for_object(
        detmodel=detmodel,
        object_id=args.object_id,
        reference_dir=reference_dir,
        ref_view_ids=ref_view_ids,
        max_side_length=args.max_side_length,
        use_square_sizing=not args.no_square,
        num_points_approx=args.num_points_approx,
        device=device,
        grayscale=args.grayscale,
    )
    if exemplar_ref is None:
        raise RuntimeError("No exemplar tokens built. Check reference_dir/object_id/ref_view_ids.")

    cap = cv2.VideoCapture(args.cam_index)
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open webcam index {args.cam_index}")

    last_time = time.time()
    fps_ema: Optional[float] = None

    try:
        with torch.inference_mode():
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if args.flip:
                    frame_bgr = cv2.flip(frame_bgr, 1)
                if args.grayscale:
                    frame_bgr = apply_grayscale(frame_bgr)

                img_t = detmodel.image_encoder.prepare_image(
                    frame_bgr,
                    max_side_length=args.max_side_length,
                    use_square_sizing=not args.no_square,
                )
                encoded_img = detmodel.image_encoder(img_t)
                encoded_image_features_list = detmodel.image_projection.v3_projection(encoded_img)

                exemplar_batch, padding_mask = pad_exemplar_batch([exemplar_ref], device=device)
                mask_preds, box_preds, det_scores, _ = generate_detections_train(
                    detmodel,
                    encoded_image_features_list,
                    exemplar_batch,
                    detection_filter_threshold=args.det_filter,
                    exemplar_padding_mask_bn=padding_mask,
                )

                masks_nhw = mask_preds[0]
                scores_n = det_scores[0]
                if masks_nhw.shape[0] > 0 and args.nms_iou > 0:
                    _, masks_nhw, scores_n = apply_mask_nms(
                        box_preds[0], masks_nhw, scores_n, iou_threshold=args.nms_iou
                    )

                vis = _draw_overlays(
                    frame_bgr,
                    masks_nhw,
                    scores_n,
                    alpha=args.alpha,
                    max_show=args.max_show,
                )

                now = time.time()
                dt = now - last_time
                last_time = now
                if dt > 0:
                    fps = 1.0 / dt
                    fps_ema = fps if fps_ema is None else fps_ema * 0.9 + fps * 0.1
                if fps_ema is not None:
                    cv2.putText(
                        vis,
                        f"FPS: {fps_ema:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )

                cv2.imshow("Exemplar Webcam", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
