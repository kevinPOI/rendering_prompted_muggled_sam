#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "assets" / "3d_printing_dataset"

RGB_PATH = DATASET_DIR / "rgb_0128.png"
INSTANCE_SEG_PATH = DATASET_DIR / "instance_segmentation_0128.png"
MAPPING_PATH = DATASET_DIR / "instance_segmentation_mapping_0128.json"
OUTPUT_PATH = ROOT / "overlay_roller_core_gt_0128.png"

BACKGROUND_BRIGHTNESS = 0.5
MASK_BRIGHTNESS = 1.35
ORANGE_TINT_ALPHA = 0.6
INSTANCE_TINTS_BGR = [
    np.array([40.0, 140.0, 255.0], dtype=np.float32),
    np.array([70.0, 220.0, 70.0], dtype=np.float32),
    np.array([255.0, 160.0, 60.0], dtype=np.float32),
]


def parse_rgba_key(key: str) -> tuple[int, int, int, int]:
    values = key.strip().removeprefix("(").removesuffix(")")
    return tuple(int(part.strip()) for part in values.split(","))  # type: ignore[return-value]


def main() -> None:
    rgb_bgr = cv2.imread(str(RGB_PATH), cv2.IMREAD_COLOR)
    inst_bgra = cv2.imread(str(INSTANCE_SEG_PATH), cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {RGB_PATH}")
    if inst_bgra is None:
        raise FileNotFoundError(f"Could not read instance segmentation image: {INSTANCE_SEG_PATH}")
    if inst_bgra.ndim != 3 or inst_bgra.shape[2] != 4:
        raise ValueError(f"Expected RGBA instance segmentation image, got shape {inst_bgra.shape}")
    if rgb_bgr.shape[:2] != inst_bgra.shape[:2]:
        raise ValueError("RGB image and instance segmentation image must have the same size")

    with MAPPING_PATH.open("r", encoding="utf-8") as f:
        color_mapping = json.load(f)

    roller_core_colors_rgba = [
        parse_rgba_key(color_key)
        for color_key, label in color_mapping.items()
        if label == "roller_core"
    ]
    if not roller_core_colors_rgba:
        raise ValueError("No roller_core entries found in mapping JSON")
    if len(roller_core_colors_rgba) > len(INSTANCE_TINTS_BGR):
        raise ValueError("Not enough tint colors configured for roller_core instances")

    inst_rgba = cv2.cvtColor(inst_bgra, cv2.COLOR_BGRA2RGBA)
    mask = np.zeros(inst_rgba.shape[:2], dtype=bool)
    instance_masks: list[np.ndarray] = []
    for color in roller_core_colors_rgba:
        instance_mask = np.all(inst_rgba == np.array(color, dtype=np.uint8), axis=2)
        instance_masks.append(instance_mask)
        mask |= np.all(inst_rgba == np.array(color, dtype=np.uint8), axis=2)

    rgb_float = rgb_bgr.astype(np.float32)
    output = np.clip(rgb_float * BACKGROUND_BRIGHTNESS, 0, 255)

    for instance_mask, tint_bgr in zip(instance_masks, INSTANCE_TINTS_BGR):
        roller_core_pixels = np.clip(rgb_float[instance_mask] * MASK_BRIGHTNESS, 0, 255)
        roller_core_tinted = np.clip(
            roller_core_pixels * (1.0 - ORANGE_TINT_ALPHA) + tint_bgr * ORANGE_TINT_ALPHA,
            0,
            255,
        )
        output[instance_mask] = roller_core_tinted

    ok = cv2.imwrite(str(OUTPUT_PATH), output.astype(np.uint8))
    if not ok:
        raise RuntimeError(f"Failed to write output image: {OUTPUT_PATH}")

    print(f"Saved {OUTPUT_PATH}")
    print(f"roller_core pixels: {int(mask.sum())}")


if __name__ == "__main__":
    main()
