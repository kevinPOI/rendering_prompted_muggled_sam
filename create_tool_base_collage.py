#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch `<object>_stl_base` renders into a collage with a white background."
    )
    parser.add_argument(
        "--object-name",
        type=str,
        default="tool_base",
        help="Object stem used in `<object-name>_stl_base_XX.png` filenames",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/kevin/ICL/rendering_prompted_muggled_sam/assets/renders_2442_0316"),
        help="Directory containing `<object-name>_stl_base_XX.png` render PNGs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/kevin/ICL/rendering_prompted_muggled_sam/assets/renders_2442_0316/tool_base_stl_base_collage_4x3.png"
        ),
        help="Output collage path",
    )
    parser.add_argument("--cols", type=int, default=4, help="Number of collage columns")
    parser.add_argument("--rows", type=int, default=3, help="Number of collage rows")
    parser.add_argument("--gap", type=int, default=0, help="Gap in pixels between tiles")
    parser.add_argument(
        "--background",
        type=int,
        nargs=3,
        metavar=("R", "G", "B"),
        default=(255, 255, 255),
        help="Background color used behind transparent pixels",
    )
    return parser.parse_args()


def find_images(input_dir: Path, object_name: str) -> list[Path]:
    pattern = f"{object_name}_stl_base_[0-9][0-9].png"
    paths = sorted(input_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No matching renders found in {input_dir} for pattern {pattern}")
    return paths


def composite_on_background(image: Image.Image, background_rgb: tuple[int, int, int]) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (*background_rgb, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def main() -> None:
    args = parse_args()
    images = find_images(args.input_dir, args.object_name)
    expected = args.cols * args.rows
    if len(images) != expected:
        raise ValueError(f"Expected {expected} images for a {args.cols}x{args.rows} collage, found {len(images)}")

    with Image.open(images[0]) as first_image:
        tile_w, tile_h = first_image.size

    canvas_w = args.cols * tile_w + (args.cols - 1) * args.gap
    canvas_h = args.rows * tile_h + (args.rows - 1) * args.gap
    background_rgb = tuple(args.background)
    canvas = Image.new("RGB", (canvas_w, canvas_h), background_rgb)

    for idx, image_path in enumerate(images):
        row, col = divmod(idx, args.cols)
        x = col * (tile_w + args.gap)
        y = row * (tile_h + args.gap)
        with Image.open(image_path) as image:
            tile = composite_on_background(image, background_rgb)
        canvas.paste(tile, (x, y))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Wrote {args.output} ({canvas_w}x{canvas_h}) from {len(images)} images")


if __name__ == "__main__":
    main()
