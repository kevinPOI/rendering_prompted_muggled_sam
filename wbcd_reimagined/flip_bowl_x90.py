#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-time script to rotate a bowl mesh +90 degrees about X and save backup."
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        default="/home/kevin/ICL/rendering_prompted_muggled_sam/assets/wbcd_meshes/bowl.STL",
        help="Path to the bowl mesh to rotate.",
    )
    parser.add_argument(
        "--backup_name",
        type=str,
        default="bowl_orginal.stl",
        help="Backup filename to store the pre-flip mesh in the same directory.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="",
        help="Optional output path. Defaults to overwriting mesh_path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh_path = Path(args.mesh_path).expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)

    backup_path = mesh_path.with_name(args.backup_name)
    if not backup_path.exists():
        shutil.copy2(mesh_path, backup_path)
    else:
        print(f"Backup already exists at {backup_path}; leaving it unchanged.")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty() or not mesh.has_triangles():
        if backup_path.is_file():
            print(f"Mesh at {mesh_path} is empty; falling back to backup {backup_path}.")
            mesh = o3d.io.read_triangle_mesh(str(backup_path))
    if mesh.is_empty() or not mesh.has_triangles():
        raise ValueError(f"Mesh invalid or empty: {mesh_path}")

    # +90 degrees about X axis.
    angle_rad = np.deg2rad(90.0)
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0.0, np.sin(angle_rad), np.cos(angle_rad)],
        ],
        dtype=float,
    )
    mesh.rotate(rot_x, center=(0.0, 0.0, 0.0))
    mesh.compute_vertex_normals()

    out_path = Path(args.out_path).expanduser().resolve() if args.out_path else mesh_path
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if not o3d.io.write_triangle_mesh(str(tmp_path), mesh):
        raise RuntimeError(f"Failed to write mesh to {out_path}")
    shutil.move(str(tmp_path), str(out_path))

    print(f"Wrote rotated mesh to {out_path}")
    print(f"Backup saved at {backup_path}")


if __name__ == "__main__":
    main()
