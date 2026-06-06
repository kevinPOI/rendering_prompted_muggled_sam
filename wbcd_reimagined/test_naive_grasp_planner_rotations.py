#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from naive_grasp_planner import _resolve_mesh_path, demo


def _parse_float_list(values: str) -> list[float]:
    out: list[float] = []
    for item in values.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(float(item))
    return out


def _rotation_from_axis(axis: str, angle_deg: float, base_rpy_deg: tuple[float, float, float]) -> tuple[float, float, float]:
    roll, pitch, yaw = base_rpy_deg
    if axis == "x":
        roll += angle_deg
    elif axis == "y":
        pitch += angle_deg
    elif axis == "z":
        yaw += angle_deg
    else:
        raise ValueError(f"Unsupported axis: {axis}")
    return roll, pitch, yaw


def _iter_rotations(
    axis: str,
    angles_deg: Iterable[float],
    base_rpy_deg: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    return [_rotation_from_axis(axis, angle_deg, base_rpy_deg) for angle_deg in angles_deg]


def parse_args() -> argparse.Namespace:
    local_stl_dir = SCRIPT_DIR / "stl"
    local_mesh_path = local_stl_dir / "pipe_assem.STL"
    parser = argparse.ArgumentParser(
        description="Run naive_grasp_planner on one mesh across multiple object rotations."
    )
    parser.add_argument("--mesh", default="", help="Path to mesh file (overrides mesh_dir/object_id).")
    parser.add_argument(
        "--mesh_dir",
        type=str,
        default=str(local_stl_dir),
        help="Directory used when --mesh/--mesh_path is not provided.",
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        default=str(local_mesh_path),
        help="Default local mesh path under wbcd_reimagined/stl.",
    )
    parser.add_argument(
        "--object_id",
        type=str,
        default="rod_mount",
        help="Default object to load when no explicit mesh path is provided.",
    )
    parser.add_argument("--top_k", type=int, default=1, help="Number of grasps to score per rotation.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--axis",
        choices=("x", "y", "z"),
        default="z",
        help="Axis to sweep when generating rotations.",
    )
    parser.add_argument(
        "--angles_deg",
        type=str,
        default="0,45,90,135,180,225,270,315",
        help="Comma-separated angles in degrees applied about the sweep axis.",
    )
    parser.add_argument(
        "--base_rpy_deg",
        type=str,
        default="0,0,0",
        help="Comma-separated base roll,pitch,yaw in degrees added before the sweep angle.",
    )
    parser.add_argument(
        "--use_convex_hull",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If true, sample grasp contacts on the mesh convex hull.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh_path_arg = args.mesh if args.mesh else args.mesh_path
    mesh_path = _resolve_mesh_path(args.mesh_dir, mesh_path_arg, args.object_id)
    mesh_path = str(Path(mesh_path).expanduser().resolve())

    base_rpy = _parse_float_list(args.base_rpy_deg)
    if len(base_rpy) != 3:
        raise ValueError("--base_rpy_deg must contain exactly three comma-separated values.")
    angles_deg = _parse_float_list(args.angles_deg)
    if not angles_deg:
        raise ValueError("--angles_deg must contain at least one angle.")

    rotations = _iter_rotations(args.axis, angles_deg, tuple(base_rpy))

    print(f"Mesh: {mesh_path}")
    print(f"Object: {args.object_id}")
    print(f"Testing {len(rotations)} rotations about {args.axis.upper()} axis.")
    print("Close each visualization window to continue to the next rotation.")

    for idx, rpy_deg in enumerate(rotations, start=1):
        print("")
        print(
            f"[{idx}/{len(rotations)}] "
            f"rpy_deg=({rpy_deg[0]:.1f}, {rpy_deg[1]:.1f}, {rpy_deg[2]:.1f})"
        )
        best_pose = demo(
            mesh_path=mesh_path,
            top_k=args.top_k,
            seed=args.seed,
            rpy_deg=rpy_deg,
            use_convex_hull=args.use_convex_hull,
        )
        if best_pose is None:
            print("No grasp found for this rotation.")
        else:
            print("Best grasp pose:")
            print(best_pose)


if __name__ == "__main__":
    main()
