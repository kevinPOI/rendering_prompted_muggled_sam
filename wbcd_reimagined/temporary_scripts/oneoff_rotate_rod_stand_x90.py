#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-off script:
- load rod_stand
- rotate 90 degrees about +X
- save as rod_stand_x90.stl alongside the original
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def _resolve_mesh_path(mesh_dir: Path, mesh_path: str, object_id: str) -> Path:
    if mesh_path:
        return Path(mesh_path).expanduser().resolve()
    candidates = [
        mesh_dir / f"{object_id}.stl",
        mesh_dir / f"{object_id}.STL",
        mesh_dir / f"{object_id}.ply",
        mesh_dir / f"{object_id}.obj",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"No mesh found for object_id={object_id} in {mesh_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rotate rod_stand mesh 90 degrees about +X.")
    parser.add_argument(
        "--mesh_dir",
        type=str,
        default="/home/kevin/ICL/rendering_prompted_muggled_sam/assets/mesh_0316",
    )
    parser.add_argument("--mesh_path", type=str, default="")
    parser.add_argument("--object_id", type=str, default="rod_stand")
    parser.add_argument("--output", type=str, default="rod_stand_x90.stl")
    args = parser.parse_args()

    mesh_dir = Path(args.mesh_dir).expanduser().resolve()
    mesh_path = _resolve_mesh_path(mesh_dir, args.mesh_path, args.object_id)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"Loaded mesh is empty: {mesh_path}")

    angle = np.deg2rad(90.0)
    rot = mesh.get_rotation_matrix_from_axis_angle([angle, 0.0, 0.0])
    mesh.rotate(rot, center=[0.0, 0.0, 0.0])
    mesh.compute_vertex_normals()

    output_path = mesh_dir / args.output
    if not o3d.io.write_triangle_mesh(str(output_path), mesh):
        raise RuntimeError(f"Failed to write mesh to {output_path}")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
