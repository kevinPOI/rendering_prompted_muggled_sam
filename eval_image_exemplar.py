#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import json
import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import time
from matplotlib.patches import Rectangle
from muggled_sam.make_sam import make_sam_from_state_dict


IGNORED_LABELS = {"BACKGROUND", "UNLABELLED"}


def parse_color_key(key: str) -> Tuple[int, ...]:
    stripped = key.strip().strip("()")
    parts = [part.strip() for part in stripped.split(",") if part.strip()]
    return tuple(int(part) for part in parts)


def parse_image_list(raw: str) -> Optional[set]:
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    items: List[str] = []
    if raw.startswith("["):
        try:
            value = ast.literal_eval(raw)
            if isinstance(value, (list, tuple, set)):
                items = [str(v) for v in value]
            else:
                items = [str(value)]
        except (ValueError, SyntaxError):
            items = [part.strip() for part in raw.strip("[]").split(",") if part.strip()]
    else:
        items = [part.strip() for part in raw.split(",") if part.strip()]
    items = [os.path.abspath(os.path.expanduser(item)) for item in items if item]
    return set(items) if items else None


def load_color_mapping(json_path: Path) -> Dict[str, List[Tuple[int, ...]]]:
    with open(json_path, "r") as handle:
        raw = json.load(handle)
    object_colors: Dict[str, List[Tuple[int, ...]]] = defaultdict(list)
    for color_key, label in raw.items():
        clean_label = label.strip()
        if not clean_label or clean_label.upper() in IGNORED_LABELS:
            continue
        object_colors[clean_label].append(parse_color_key(color_key))
    return dict(object_colors)


def collect_multi_object_samples(
    dataset_root: str,
    sub_sample: int = 5,
) -> Tuple[Dict[str, List[Dict[str, str]]], List[Dict[str, str]]]:
    dataset_path = Path(dataset_root).expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(dataset_root)
    pattern = re.compile(r"instance_segmentation_(\d{4})\.png$")
    object_map: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    all_entries: List[Dict[str, str]] = []
    if sub_sample < 1:
        raise ValueError("sub_sample must be >= 1")
    for idx, inst_path in enumerate(sorted(dataset_path.glob("instance_segmentation_*.png"))):
        if sub_sample > 1 and (idx % sub_sample) != 0:
            continue
        match = pattern.match(inst_path.name)
        if not match:
            continue
        frame_id = match.group(1)
        rgb_path = dataset_path / f"rgb_{frame_id}.png"
        if not rgb_path.is_file():
            jpg_fallback = dataset_path / f"rgb_{frame_id}.jpg"
            if jpg_fallback.is_file():
                rgb_path = jpg_fallback
        mapping_path = dataset_path / f"instance_segmentation_mapping_{frame_id}.json"
        if not (rgb_path.is_file() and mapping_path.is_file()):
            continue
        color_map = load_color_mapping(mapping_path)
        if not color_map:
            continue
        for object_id, colors in color_map.items():
            for color in colors:
                entry = {
                    "object_id": object_id,
                    "frame_id": frame_id,
                    "rgb_path": str(rgb_path),
                    "inst_path": str(inst_path),
                    "color": color,
                }
                object_map[object_id].append(entry)
                all_entries.append(entry)
    if not all_entries:
        raise RuntimeError(f"No multi-object samples found under {dataset_root}")
    return object_map, all_entries


def load_instance_mask(inst_path: str, color: Tuple[int, ...]) -> np.ndarray:
    seg = load_instance_segmentation(inst_path)
    return mask_for_color(seg, color)


def load_instance_segmentation(inst_path: str) -> np.ndarray:
    seg = cv2.imread(inst_path, cv2.IMREAD_UNCHANGED)
    if seg is None:
        raise FileNotFoundError(inst_path)
    if seg.ndim == 2:
        seg = seg[..., None]
    elif seg.shape[2] == 4:
        seg = cv2.cvtColor(seg, cv2.COLOR_BGRA2RGBA)
    elif seg.shape[2] == 3:
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
    return seg.astype(np.uint8)


def mask_for_color(seg: np.ndarray, color: Tuple[int, ...]) -> np.ndarray:
    target = np.array(color, dtype=np.uint8)
    channels = seg.shape[2]
    if target.shape[0] > channels:
        target = target[:channels]
    elif target.shape[0] < channels:
        pad = np.zeros(channels - target.shape[0], dtype=np.uint8)
        target = np.concatenate([target, pad], axis=0)
    target = target.reshape(1, 1, -1)
    mask = np.all(seg == target, axis=-1)
    return mask.astype(np.float32)


def load_instance_masks_for_object(
    inst_path: str,
    mapping_path: str,
    object_id: str,
    seg_cache: Optional[Dict[str, np.ndarray]] = None,
    mapping_cache: Optional[Dict[str, Dict[str, List[Tuple[int, ...]]]]] = None,
) -> List[np.ndarray]:
    mapping_key = str(mapping_path)
    if mapping_cache is not None and mapping_key in mapping_cache:
        color_map = mapping_cache[mapping_key]
    else:
        color_map = load_color_mapping(Path(mapping_path))
        if mapping_cache is not None:
            mapping_cache[mapping_key] = color_map
    colors = color_map.get(object_id, [])
    if not colors:
        return []

    inst_key = str(inst_path)
    if seg_cache is not None and inst_key in seg_cache:
        seg = seg_cache[inst_key]
    else:
        seg = load_instance_segmentation(inst_path)
        if seg_cache is not None:
            seg_cache[inst_key] = seg

    masks: List[np.ndarray] = []
    for color in colors:
        mask = mask_for_color(seg, color)
        if mask.sum() > 0:
            masks.append(mask)
    return masks


def load_bgr(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def apply_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2 or image_bgr.shape[2] == 1:
        gray = image_bgr if image_bgr.ndim == 2 else image_bgr[:, :, 0]
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def load_mask_gray(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask > 0).astype(np.uint8)


def sample_points_from_mask(mask_image: np.ndarray, num_points_approx: int = 25) -> List[Tuple[float, float]]:
    golden_ratio = (1.0 + 5.0**0.5) / 2.0
    num_fib_pts = golden_ratio * num_points_approx
    pt_idx = np.arange(0, num_fib_pts, dtype=np.float32)
    r = (1) * np.sqrt(pt_idx / num_fib_pts) / np.sqrt(2, dtype=np.float32)
    theta = 2.0 * np.pi * (pt_idx / golden_ratio)

    fib_sample_x_norm = 0.5 + r * np.cos(theta)
    fib_sample_y_norm = 0.5 + r * np.sin(theta)

    ok_x_pts = np.bitwise_and(fib_sample_x_norm > 0.0, fib_sample_x_norm < 1.0)
    ok_y_pts = np.bitwise_and(fib_sample_y_norm > 0.0, fib_sample_y_norm < 1.0)
    ok_pts = np.bitwise_and(ok_x_pts, ok_y_pts)
    fib_sample_x_norm, fib_sample_y_norm = fib_sample_x_norm[ok_pts], fib_sample_y_norm[ok_pts]

    ref_h, ref_w = mask_image.shape[0:2]
    if mask_image.ndim > 2:
        if mask_image.shape[2] == 3:
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        else:
            mask_image = mask_image[:, :, 0]

    mask_bin = mask_image > 0
    contours_px_list, _ = cv2.findContours(np.uint8(mask_bin), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_sample_xy_px_list = []
    for contour_pts_list in contours_px_list:
        if len(contour_pts_list) < 3:
            continue
        x1_px, y1_px, w_px, h_px = cv2.boundingRect(contour_pts_list)
        if w_px < 1 or h_px < 1:
            continue

        sample_x_px = np.round(x1_px + fib_sample_x_norm * (w_px - 1)).astype(np.int32)
        sample_y_px = np.round(y1_px + fib_sample_y_norm * (h_px - 1)).astype(np.int32)
        is_in_mask = mask_bin[sample_y_px, sample_x_px]
        final_sample_xy_px = np.column_stack((sample_x_px[is_in_mask], sample_y_px[is_in_mask]))
        final_sample_xy_px_list.append(final_sample_xy_px)

    if not final_sample_xy_px_list:
        return []

    out_xy_norm = np.concatenate(final_sample_xy_px_list) / np.float32((ref_w - 1, ref_h - 1))
    return out_xy_norm.tolist()


def resize_mask(mask: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    return cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)


def build_gt_down_list(
    gt_masks: List[np.ndarray],
    preencode_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
    device: torch.device,
) -> List[torch.Tensor]:
    gt_down_list: List[torch.Tensor] = []
    for gt_mask in gt_masks:
        gt_preenc = resize_mask(gt_mask, preencode_hw)
        gt_tensor = torch.from_numpy(gt_preenc).to(device).unsqueeze(0).unsqueeze(0)
        gt_down = F.interpolate(gt_tensor, size=target_hw, mode="nearest").squeeze(0).squeeze(0)
        gt_down_list.append(gt_down > 0.5)
    return gt_down_list


def parse_ref_view_ids(value: str) -> List[str]:
    if not value:
        return []
    parts = [part.strip() for part in value.split(",") if part.strip()]
    ids: List[str] = []
    for part in parts:
        try:
            as_int = int(part)
            ids.append(f"{as_int:02d}")
        except ValueError:
            ids.append(part)
    return ids


def generate_detections_train(
    detmodel,
    encoded_image_features_list: List[torch.Tensor],
    encoded_exemplars_bnc: torch.Tensor,
    detection_filter_threshold: float = 0.0,
    exemplar_padding_mask_bn: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lowres_imgenc_bchw, hiresx2_imgenc_bchw, hiresx4_imgenc_bchw = encoded_image_features_list
    no_exemplars = encoded_exemplars_bnc.shape[1] == 0
    if no_exemplars:
        blk_tok, blk_box, blk_score, blk_score_logits, blk_pres = detmodel.exemplar_detector.create_blank_output(
            lowres_imgenc_bchw
        )
        blk_masks, _ = detmodel.exemplar_segmentation.create_blank_output(blk_tok, lowres_imgenc_bchw)
        return blk_masks, blk_box, blk_score, blk_pres

    fused_imgexm_tokens_bchw = detmodel.image_exemplar_fusion(
        lowres_imgenc_bchw,
        encoded_exemplars_bnc,
        exemplar_padding_mask_bn,
    )
    enc_det_tokens_bnc, boxes_xy1xy2_bn22, det_scores_bn, det_scores_logits_bn, pres_scores = detmodel.exemplar_detector(
        fused_imgexm_tokens_bchw, encoded_exemplars_bnc, exemplar_padding_mask_bn
    )

    if detection_filter_threshold > 1e-3:
        if det_scores_bn.shape[0] != 1:
            raise ValueError("Cannot pre-filter detections when using batched inputs!")
        ok_filter = det_scores_bn[0] > detection_filter_threshold
        enc_det_tokens_bnc = enc_det_tokens_bnc[:, ok_filter]
        boxes_xy1xy2_bn22 = boxes_xy1xy2_bn22[:, ok_filter]
        det_scores_bn = det_scores_bn[:, ok_filter]

    mask_preds_bnhw, _ = detmodel.exemplar_segmentation(
        enc_det_tokens_bnc,
        fused_imgexm_tokens_bchw,
        hiresx2_imgenc_bchw,
        hiresx4_imgenc_bchw,
        encoded_exemplars_bnc,
        exemplar_padding_mask_bn,
    )
    return mask_preds_bnhw, boxes_xy1xy2_bn22, det_scores_bn, pres_scores


def encode_detection_image_no_infer(
    detmodel,
    image_bgr: np.ndarray,
    max_side_length: int,
    use_square_sizing: bool,
) -> Tuple[List[torch.Tensor], Tuple[int, int], Tuple[int, int]]:
    image_rgb_normalized_bchw = detmodel.image_encoder.prepare_image(
        image_bgr, max_side_length=max_side_length, use_square_sizing=use_square_sizing
    )
    with torch.no_grad():
        encoded_img = detmodel.image_encoder(image_rgb_normalized_bchw)
        encoded_image_features_list = detmodel.image_projection.v3_projection(encoded_img)
    patch_grid_hw = encoded_image_features_list[0].shape[2:]
    image_preenc_hw = image_rgb_normalized_bchw.shape[2:]
    return encoded_image_features_list, patch_grid_hw, image_preenc_hw


def encode_exemplars_no_infer(
    detmodel,
    encoded_image_features_list: List[torch.Tensor],
    text: Optional[str],
    point_xy_norm_list: Optional[List[Tuple[float, float]]],
    include_coordinate_encodings: bool,
) -> torch.Tensor:
    lowres_imgenc_bchw = encoded_image_features_list[0]
    img_b, img_c, _, _ = lowres_imgenc_bchw.shape
    device, dtype = lowres_imgenc_bchw.device, lowres_imgenc_bchw.dtype
    missing_input_tensor_bnc = torch.empty((img_b, 0, img_c), device=device, dtype=dtype)
    encoded_text_bnc = missing_input_tensor_bnc
    encoded_sampling_bnc = missing_input_tensor_bnc
    with torch.no_grad():
        if isinstance(text, str) and len(text) > 0:
            encoded_text_bnc = detmodel.text_encoder(text)
        if point_xy_norm_list is not None:
            encoded_sampling_bnc = detmodel.sampling_encoder(
                lowres_imgenc_bchw,
                boxes_bn22=detmodel.sampling_encoder.prepare_box_input(None),
                points_bn2=detmodel.sampling_encoder.prepare_point_input(point_xy_norm_list),
                negative_boxes_bn22=detmodel.sampling_encoder.prepare_box_input(None),
                negative_points_bn2=detmodel.sampling_encoder.prepare_point_input(None),
                include_coordinate_encodings=include_coordinate_encodings,
            )
    return torch.cat((encoded_sampling_bnc, encoded_text_bnc), dim=1)


def select_best_mask(mask_preds_nhw: torch.Tensor, gt_mask_hw: torch.Tensor) -> Tuple[int, torch.Tensor]:
    pred_bin = mask_preds_nhw > 0
    gt_bin = gt_mask_hw > 0.5
    intersection = (pred_bin & gt_bin).sum(dim=(1, 2))
    union = (pred_bin | gt_bin).sum(dim=(1, 2)).clamp_min(1)
    iou = intersection.float() / union.float()
    best_idx = int(torch.argmax(iou).item())
    return best_idx, iou[best_idx]


def compute_mask_iou(pred_mask_hw: torch.Tensor, gt_mask_hw: torch.Tensor) -> torch.Tensor:
    pred_bin = pred_mask_hw > 0
    gt_bin = gt_mask_hw > 0.5
    intersection = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum().clamp_min(1)
    return intersection.float() / union.float()


def match_masks_to_gts(
    pred_masks: List[torch.Tensor],
    gt_masks: List[torch.Tensor],
    iou_threshold: float = 0.5,
) -> List[Tuple[int, int, float]]:
    if not pred_masks or not gt_masks:
        return []
    pairs: List[Tuple[float, int, int]] = []
    for p_idx, pred in enumerate(pred_masks):
        for g_idx, gt in enumerate(gt_masks):
            iou = compute_mask_iou(pred, gt)
            iou_val = float(iou.item())
            if iou_val >= iou_threshold:
                pairs.append((iou_val, p_idx, g_idx))
    pairs.sort(key=lambda x: x[0], reverse=True)

    matched_pred = [False] * len(pred_masks)
    matched_gt = [False] * len(gt_masks)
    matches: List[Tuple[int, int, float]] = []
    for iou_val, p_idx, g_idx in pairs:
        if matched_pred[p_idx] or matched_gt[g_idx]:
            continue
        matched_pred[p_idx] = True
        matched_gt[g_idx] = True
        matches.append((p_idx, g_idx, iou_val))
    return matches


def compute_pq_stats(
    pred_masks: List[torch.Tensor],
    gt_masks: List[torch.Tensor],
    pred_scores: Optional[torch.Tensor] = None,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25,
) -> Tuple[float, int, int, int]:
    if pred_scores is not None and pred_masks:
        filtered_masks: List[torch.Tensor] = []
        for idx, mask in enumerate(pred_masks):
            if float(pred_scores[idx].item()) >= score_threshold:
                filtered_masks.append(mask)
        pred_masks = filtered_masks

    if not pred_masks and not gt_masks:
        return 0.0, 0, 0, 0
    if not pred_masks:
        return 0.0, 0, 0, len(gt_masks)
    if not gt_masks:
        return 0.0, 0, len(pred_masks), 0

    matches = match_masks_to_gts(pred_masks, gt_masks, iou_threshold=iou_threshold)
    sum_iou = sum(match[2] for match in matches)
    tp = len(matches)

    fp = len(pred_masks) - tp
    fn = len(gt_masks) - tp
    return sum_iou, tp, fp, fn


def update_pq_accumulators(
    pq_stats: Dict[float, Dict[str, float]],
    pred_masks: List[torch.Tensor],
    gt_masks: List[torch.Tensor],
    pred_scores: Optional[torch.Tensor],
    iou_threshold: float,
) -> None:
    for score_threshold, stats in pq_stats.items():
        sum_iou, tp, fp, fn = compute_pq_stats(
            pred_masks,
            gt_masks,
            pred_scores=pred_scores,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        stats["sum_iou"] += sum_iou
        stats["tp"] += tp
        stats["fp"] += fp
        stats["fn"] += fn


def build_detection_record(
    object_id: str,
    frame_id: str,
    pred_scores: Optional[torch.Tensor],
    pred_masks: List[torch.Tensor],
    gt_masks: List[torch.Tensor],
    iou_threshold: float,
    top_k: int = 5,
) -> Dict[str, object]:
    matches = match_masks_to_gts(pred_masks, gt_masks, iou_threshold=iou_threshold)
    match_by_pred = {p_idx: (g_idx, iou_val) for p_idx, g_idx, iou_val in matches}
    scores_cpu: Optional[torch.Tensor] = None
    if pred_scores is not None:
        scores_cpu = pred_scores.detach().float().cpu()

    record: Dict[str, object] = {
        "object_id": object_id,
        "frame_id": frame_id,
        "num_gt": len(gt_masks),
        "num_pred": len(pred_masks),
    }
    for idx in range(top_k):
        score_val: Optional[float] = None
        if scores_cpu is not None and idx < scores_cpu.numel():
            score_val = float(scores_cpu[idx].item())
        record[f"mask{idx + 1}_score"] = score_val
        if idx in match_by_pred:
            record[f"mask{idx + 1}_matched_gt"] = int(match_by_pred[idx][0])
        else:
            record[f"mask{idx + 1}_matched_gt"] = None
    return record


def apply_mask_nms(
    box_preds_n22: torch.Tensor,
    mask_preds_nhw: torch.Tensor,
    det_scores_n: torch.Tensor,
    iou_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if det_scores_n.numel() == 0:
        return box_preds_n22, mask_preds_nhw, det_scores_n

    score_order = torch.argsort(det_scores_n, descending=True)
    if iou_threshold <= 0:
        return box_preds_n22[score_order], mask_preds_nhw[score_order], det_scores_n[score_order]

    masks_bin = mask_preds_nhw > 0
    keep: List[int] = []
    for idx in score_order.tolist():
        if not keep:
            keep.append(int(idx))
            continue
        suppress = False
        cand = masks_bin[idx]
        cand_area = cand.sum().clamp_min(1)
        for kept_idx in keep:
            kept = masks_bin[kept_idx]
            inter = (cand & kept).sum()
            union = (cand | kept).sum().clamp_min(1)
            kept_area = kept.sum().clamp_min(1)
            iou = inter.float() / union.float()
            overlap_cand = inter.float() / cand_area.float()
            overlap_kept = inter.float() / kept_area.float()
            if (
                float(iou.item()) > iou_threshold
                or float(overlap_cand.item()) >= 0.95
                or float(overlap_kept.item()) >= 0.95
            ):
                suppress = True
                break
        if not suppress:
            keep.append(int(idx))

    if not keep:
        empty = det_scores_n[:0]
        return box_preds_n22[:0], mask_preds_nhw[:0], empty
    keep_tensor = torch.as_tensor(keep, device=det_scores_n.device, dtype=torch.long)
    return box_preds_n22[keep_tensor], mask_preds_nhw[keep_tensor], det_scores_n[keep_tensor]


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def pad_exemplar_batch(
    exemplars_list: List[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not exemplars_list:
        raise ValueError("No exemplars provided for batching.")
    max_n = max(t.shape[1] for t in exemplars_list)
    feat_dim = exemplars_list[0].shape[2]
    batch = []
    padding_masks = []
    for ex in exemplars_list:
        ex = ex.to(device)
        n = ex.shape[1]
        if n < max_n:
            pad = torch.zeros((1, max_n - n, feat_dim), device=device, dtype=ex.dtype)
            ex = torch.cat([ex, pad], dim=1)
            mask = torch.zeros((max_n,), device=device, dtype=torch.bool)
            mask[n:] = True
        else:
            mask = torch.zeros((max_n,), device=device, dtype=torch.bool)
        batch.append(ex)
        padding_masks.append(mask)
    batch_bnc = torch.cat(batch, dim=0)
    padding_mask_bn = torch.stack(padding_masks, dim=0)
    return batch_bnc, padding_mask_bn


def _mask_bbox(mask_hw: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask_hw > 0)
    if ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1, y1


def load_reference_images(
    object_id: str,
    reference_dir: Path,
    ref_view_ids: List[str],
    max_views: int = 4,
) -> List[np.ndarray]:
    ref_images: List[np.ndarray] = []
    lookup_ids = [object_id, object_id.upper(), object_id.lower()]
    for ref_id in ref_view_ids:
        if len(ref_images) >= max_views:
            break
        ref_img_path = None
        for lookup_id in lookup_ids:
            ref_stub = f"{lookup_id}_stl_base_{ref_id}"
            candidate = reference_dir / f"{ref_stub}.png"
            if candidate.is_file():
                ref_img_path = candidate
                break
        if ref_img_path is None:
            continue
        try:
            ref_images.append(load_bgr(str(ref_img_path)))
        except FileNotFoundError:
            continue
    return ref_images


def build_reference_grid(ref_images: List[np.ndarray], target_height: int) -> np.ndarray:
    tile_h = max(1, target_height // 2)
    tile_w = tile_h
    tiles: List[np.ndarray] = []
    for idx in range(4):
        if idx < len(ref_images):
            tile = cv2.resize(ref_images[idx], (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        else:
            tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        tiles.append(tile)

    row_top = np.concatenate(tiles[:2], axis=1)
    row_bottom = np.concatenate(tiles[2:], axis=1)
    grid = np.concatenate([row_top, row_bottom], axis=0)

    if grid.shape[0] < target_height:
        pad_h = target_height - grid.shape[0]
        pad = np.zeros((pad_h, grid.shape[1], 3), dtype=np.uint8)
        grid = np.concatenate([grid, pad], axis=0)
    elif grid.shape[0] > target_height:
        grid = grid[:target_height, :, :]
    return grid


def save_mask_triptych(
    image_bgr: np.ndarray,
    mask_preds_nhw: torch.Tensor,
    detection_scores_n: torch.Tensor,
    gt_masks: Optional[List[np.ndarray]],
    object_id: str,
    reference_dir: Path,
    ref_view_ids: List[str],
    image_name: str,
    output_path: str,
) -> None:
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle(f"image={image_name} | object_id={object_id}", fontsize=12)

    ref_images = load_reference_images(object_id, reference_dir, ref_view_ids)
    ref_grid = build_reference_grid(ref_images, h)
    axes[0].imshow(cv2.cvtColor(ref_grid, cv2.COLOR_BGR2RGB))
    axes[0].axis("off")

    axes[1].imshow(image_rgb)
    axes[2].imshow(image_rgb)

    palette = plt.get_cmap("tab10").colors
    scores_cpu = detection_scores_n.detach().float().cpu().numpy() if detection_scores_n.numel() > 0 else np.array([])
    num_masks = int(mask_preds_nhw.shape[0]) if mask_preds_nhw.ndim == 3 else 0

    if gt_masks:
        for i, gt_mask in enumerate(gt_masks):
            if gt_mask is None:
                continue
            color = palette[(i + 1) % len(palette)]
            mask_resized = cv2.resize(gt_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            if mask_resized.max() <= 0:
                continue
            rgba = np.zeros((h, w, 4), dtype=np.float32)
            rgba[..., :3] = color
            rgba[..., 3] = (mask_resized > 0) * 0.5
            axes[1].imshow(rgba)

            bbox = _mask_bbox(mask_resized)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            rect = Rectangle(
                (x0, y0),
                x1 - x0 + 1,
                y1 - y0 + 1,
                edgecolor=color,
                facecolor="none",
                linewidth=1.5,
            )
            axes[1].add_patch(rect)
            axes[1].text(
                x0,
                max(0, y0 - 4),
                f"(gt={i})",
                color=color,
                fontsize=10,
                verticalalignment="bottom",
            )

    if num_masks > 0:
        topk = min(2, num_masks, int(scores_cpu.size))
        if topk > 0:
            top_idx = np.argsort(scores_cpu)[-topk:][::-1]
        else:
            top_idx = []
    else:
        top_idx = []

    for rank, i in enumerate(top_idx):
        color = palette[rank + 1% len(palette)]
        mask = mask_preds_nhw[int(i)]
        mask_bin = (mask > 0).detach().float().cpu().numpy()
        if mask_bin.max() <= 0:
            continue
        mask_resized = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)

        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[..., :3] = color
        rgba[..., 3] = mask_resized * 0.5
        axes[2].imshow(rgba)

        bbox = _mask_bbox(mask_resized)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        rect = Rectangle(
            (x0, y0),
            x1 - x0 + 1,
            y1 - y0 + 1,
            edgecolor=color,
            facecolor="none",
            linewidth=1.5,
        )
        axes[2].add_patch(rect)
        score_val = float(scores_cpu[int(i)]) if scores_cpu.size > int(i) else 0.0
        axes[2].text(
            x0,
            max(0, y0 - 4),
            f"(id={int(i)}, prob={score_val:.2f})",
            color=color,
            fontsize=10,
            verticalalignment="bottom",
        )

    for ax in axes:
        ax.axis("off")
    fig.tight_layout(pad=0.1, rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_exemplar_tokens_for_object(
    detmodel,
    object_id: str,
    reference_dir: Path,
    ref_view_ids: List[str],
    max_side_length: int,
    use_square_sizing: bool,
    num_points_approx: int,
    device: torch.device,
    grayscale: bool,
) -> Optional[torch.Tensor]:
    lookup_id = object_id
    feats: List[torch.Tensor] = []
    for ref_id in ref_view_ids:
        ref_stub = f"{lookup_id}_stl_base_{ref_id}"
        ref_img_path = reference_dir / f"{ref_stub}.png"
        ref_mask_path = reference_dir / f"{ref_stub}_mask.png"
        if not (ref_img_path.is_file() and ref_mask_path.is_file()):
            continue
        try:
            ref_image = load_bgr(str(ref_img_path))
            ref_mask = load_mask_gray(str(ref_mask_path))
        except FileNotFoundError:
            continue
        if grayscale:
            ref_image = apply_grayscale(ref_image)
        if ref_mask.shape[:2] != ref_image.shape[:2]:
            ref_mask = resize_mask(ref_mask, ref_image.shape[:2])
        pts = sample_points_from_mask(ref_mask, num_points_approx=num_points_approx)
        if not pts:
            continue
        encimg_ref, _, _ = encode_detection_image_no_infer(
            detmodel, ref_image, max_side_length=max_side_length, use_square_sizing=use_square_sizing
        )
        exemplar_tokens = encode_exemplars_no_infer(
            detmodel,
            encimg_ref,
            text="visual",
            point_xy_norm_list=pts,
            include_coordinate_encodings=False,
        )
        feats.append(exemplar_tokens.detach().cpu())
    if not feats:
        return None
    exemplar_ref = torch.cat(feats, dim=1).to(device)
    return exemplar_ref


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAMv3 exemplar detection modules.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="model_weights/sam3.pt",
        help="Path to SAMv3 checkpoint (.pt).",
    )
    # parser.add_argument(
    #     "--dataset_root",
    #     type=str,
    #     default=["/sata1/data/kevin/realworld_datasets/persam_v2"],
    #     help="Comma-separated dataset roots.",
    # )
    parser.add_argument(
        "--reference_dir",
        type=str,
        default="assets/renders_2442_0316",
        help="Path to reference renders.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=["assets/3d_printing_dataset"],
        help="Comma-separated dataset roots.",
    )
    parser.add_argument("--ref_view_ids", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11", help="Reference view ids to use.")
    parser.add_argument("--max_side_length", type=int, default=1008)
    parser.add_argument("--no_square", action="store_true", help="Disable square resizing in encoder.")
    parser.add_argument("--num_points_approx", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--sub_sample",
        type=int,
        default=1 ,
        help="Evaluate every Nth image in the dataset (1 = use all images).",
    )
    parser.add_argument(
        "--nms_iou",
        type=float,
        default=0.5,
        help="IoU threshold for mask NMS (<=0 disables NMS).",
    )
    parser.add_argument("--det_filter", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="outputs_eval_exemplar")
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
    #[/sata1/data/kevin/realworld_datasets/primesense_converted/000011/rgb_0009.png]
    parser.add_argument("--grayscale", default = False, help="Convert all input images to grayscale.")
    parser.add_argument(
        "--multi_gt_only",
        default = False,
        help="Only evaluate samples with multiple GT instances for the target object.",
    )
    parser.add_argument("--finetune_ckpt", type=str, default="model_weights/0321_k12_b156_resume_from_preprinte18_s1_e34.pth", help="Optional finetuned detector checkpoint.")
    return parser.parse_args()

#/home/kevin/muggled_sam/finetune_exemplar/multi_object_best/finetune_epoch_017.pth
def main() -> None:
    args = parse_args()
    dataset_roots: List[str] = []
    for item in args.dataset_root:
        dataset_roots.extend([part.strip() for part in item.split(",") if part.strip()])
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

    object_samples: Dict[str, List[Dict[str, str]]] = {}
    all_entries: List[Dict[str, str]] = []
    for root in dataset_roots:
        cur_samples, cur_entries = collect_multi_object_samples(root, sub_sample=args.sub_sample)
        for obj_id, entries in cur_samples.items():
            object_samples.setdefault(obj_id, []).extend(entries)
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
            entry
            for entry in all_entries
            if os.path.abspath(entry["rgb_path"]) in image_list
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

    ref_cache: Dict[str, torch.Tensor] = {}
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

                    if obj_id not in ref_cache:
                        exemplar_ref = build_exemplar_tokens_for_object(
                            detmodel=detmodel,
                            object_id=obj_id,
                            reference_dir=reference_dir,
                            ref_view_ids=ref_view_ids,
                            max_side_length=args.max_side_length,
                            use_square_sizing=not args.no_square,
                            num_points_approx=args.num_points_approx,
                            device=device,
                            grayscale=args.grayscale,
                        )
                        if exemplar_ref is None:
                            continue
                        ref_cache[obj_id] = exemplar_ref.detach().cpu()

                    exemplar_ref = ref_cache[obj_id]
                    prepared.append(
                        {
                            "object_id": obj_id,
                            "frame_id": entry["frame_id"],
                            "rgb_path": entry["rgb_path"],
                            "image_bgr": image_bgr,
                            "gt_masks": gt_masks,
                            "exemplar_ref": exemplar_ref,
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

                    mask_preds, box_preds, det_scores, pres_scores = generate_detections_train(
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
                                ref_view_ids=ref_view_ids,
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
        if denom > 0:
            pq = stats["sum_iou"] / denom
        else:
            pq = 0.0
        print(
            f"PQ@score>={score_threshold:.2f}={pq:.4f} "
            f"tp={int(stats['tp'])} fp={int(stats['fp'])} fn={int(stats['fn'])}"
        )


if __name__ == "__main__":
    main()
