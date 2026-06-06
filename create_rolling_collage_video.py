#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a horizontal rolling MP4 from a wide collage image. "
            "The video starts at the left edge and pans to the right edge."
        )
    )
    parser.add_argument("input_image", type=Path, help="Path to the source collage image")
    parser.add_argument("output_video", type=Path, help="Path to the output MP4 file")
    parser.add_argument("--width", type=int, default=1920, help="Output video width")
    parser.add_argument("--height", type=int, default=1080, help="Output video height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--duration", type=float, default=15.0, help="Video duration in seconds")
    return parser.parse_args()


def load_and_prepare_image(image_path: Path, target_height: int) -> Image.Image:
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    image = Image.open(image_path).convert("RGB")
    if image.height == target_height:
        return image

    scale = target_height / image.height
    resized_width = max(1, round(image.width * scale))
    return image.resize((resized_width, target_height), Image.Resampling.LANCZOS)


def main() -> None:
    args = parse_args()
    frame_count = max(1, round(args.fps * args.duration))
    image = load_and_prepare_image(args.input_image, args.height)

    if image.width < args.width:
        raise ValueError(
            f"Input image width after scaling ({image.width}) is smaller than output width ({args.width})"
        )

    max_offset = image.width - args.width
    offsets = np.linspace(0, max_offset, frame_count)

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {args.output_video}")

    try:
        for offset in offsets:
            left = int(round(offset))
            frame = image.crop((left, 0, left + args.width, args.height))
            frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()

    print(
        f"Wrote {args.output_video} "
        f"({args.width}x{args.height}, {args.fps} fps, {frame_count} frames, pan {max_offset}px)"
    )


if __name__ == "__main__":
    main()
