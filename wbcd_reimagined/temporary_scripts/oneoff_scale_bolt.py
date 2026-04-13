#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-off script:
- rename bolt.stl -> old_bolt.stl
- scale mesh by 300% (3.0x) around bbox origin
- save as bolt.stl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import open3d as o3d


def _resolve_mesh_path(mesh_dir: Path, filename: str) -> Path:
    mesh_path = mesh_dir / filename
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    return mesh_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Scale bolt.stl by 3.0x and keep centered on bbox origin")
    parser.add_argument(
        "--mesh_dir",
        type=str,
        default="/home/kevin/ICL/rendering_prompted_muggled_sam/assets/mesh_0316",
    )
    parser.add_argument("--input", type=str, default="bolt.STL")
    parser.add_argument("--backup", type=str, default="old_bolt.STL")
    args = parser.parse_args()

    mesh_dir = Path(args.mesh_dir).expanduser().resolve()
    mesh_path = _resolve_mesh_path(mesh_dir, args.input)
    backup_path = mesh_dir / args.backup

    if backup_path.exists():
        raise FileExistsError(f"Backup already exists: {backup_path}")

    mesh_path.rename(backup_path)
    mesh = o3d.io.read_triangle_mesh(str(backup_path))
    if mesh.is_empty():
        raise ValueError(f"Loaded mesh is empty: {backup_path}")

    aabb = mesh.get_axis_aligned_bounding_box()
    center = aabb.get_center()
    mesh.translate(-center)
    mesh.scale(3.0, center=[0.0, 0.0, 0.0])
    mesh.compute_vertex_normals()

    if not o3d.io.write_triangle_mesh(str(mesh_path), mesh):
        raise RuntimeError(f"Failed to write mesh to {mesh_path}")
    print(f"Renamed {mesh_path.name} to {backup_path.name} and wrote scaled mesh to {mesh_path}")


if __name__ == "__main__":
    main()
