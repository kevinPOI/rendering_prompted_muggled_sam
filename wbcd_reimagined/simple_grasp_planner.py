#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple parallel-jaw grasp planner for watertight triangle meshes.

Outputs top-K grasp poses (4x4) and scores using surface sampling,
contact-pair generation, simple box-geometry collision checks, and
lightweight scoring. Designed to work on convex and thin-shell objects.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import trimesh


@dataclass(frozen=True)
class GripperParams:
    max_width: float
    finger_thickness: float
    finger_length: float
    palm_width: float
    palm_depth: float
    palm_height: float
    finger_height: Optional[float] = None


@dataclass(frozen=True)
class PlannerParams:
    num_surface_points: int = 600
    max_pairs: int = 5000
    normal_dot_threshold: float = -0.5
    min_width: float = 0.0
    approach_directions: int = 12
    approach_steps: int = 6
    collision_allowance: float = 0.001
    box_point_density: int = 3


@dataclass(frozen=True)
class GraspCandidate:
    pose: np.ndarray
    score: float
    midpoint: np.ndarray
    distance: float
    normal_opposition: float
    clearance: float


def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v.copy()
    return v / n


def _transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    homog = np.hstack([points, np.ones((points.shape[0], 1), dtype=points.dtype)])
    out = (T @ homog.T).T
    return out[:, :3]


def _box_surface_points(extents: np.ndarray, density: int) -> np.ndarray:
    """Create surface points on an axis-aligned box centered at origin."""
    density = max(2, int(density))
    xs = np.linspace(-extents[0] / 2.0, extents[0] / 2.0, density)
    ys = np.linspace(-extents[1] / 2.0, extents[1] / 2.0, density)
    zs = np.linspace(-extents[2] / 2.0, extents[2] / 2.0, density)
    pts = []
    for x in (xs[0], xs[-1]):
        for y in ys:
            for z in zs:
                pts.append([x, y, z])
    for y in (ys[0], ys[-1]):
        for x in xs:
            for z in zs:
                pts.append([x, y, z])
    for z in (zs[0], zs[-1]):
        for x in xs:
            for y in ys:
                pts.append([x, y, z])
    return np.unique(np.asarray(pts, dtype=np.float64), axis=0)


def sample_surface_points(mesh: trimesh.Trimesh, num_points: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sample surface points and normals."""
    points, face_idx = trimesh.sample.sample_surface(mesh, num_points, seed=seed)
    normals = mesh.face_normals[face_idx]
    return np.asarray(points, dtype=np.float64), np.asarray(normals, dtype=np.float64)


def generate_contact_pairs(
    points: np.ndarray,
    normals: np.ndarray,
    max_width: float,
    min_width: float,
    normal_dot_threshold: float,
    max_pairs: int,
    seed: int = 0,
    distance_bins: int = 5,
) -> list[tuple[int, int, float]]:
    """Generate candidate contact pairs with opposing normals."""
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    if n < 2:
        return []
    diff = points[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    dot = normals @ normals.T
    mask = (dists <= max_width) & (dists >= min_width) & (dot <= normal_dot_threshold)
    iu = np.triu_indices(n, k=1)
    valid = np.where(mask[iu])[0]
    if valid.size == 0:
        return []
    pairs = list(zip(iu[0][valid], iu[1][valid], dists[iu][valid]))
    if len(pairs) <= max_pairs:
        rng.shuffle(pairs)
        return pairs
    # Stratified sampling over distance to avoid bias toward large widths.
    distances = np.array([p[2] for p in pairs])
    bins = np.linspace(min_width, max_width, distance_bins + 1)
    selected: list[tuple[int, int, float]] = []
    per_bin = max(1, max_pairs // distance_bins)
    for b0, b1 in zip(bins[:-1], bins[1:]):
        idx = np.where((distances >= b0) & (distances < b1))[0]
        if idx.size == 0:
            continue
        take = min(per_bin, idx.size)
        chosen = rng.choice(idx, size=take, replace=False)
        selected.extend([pairs[i] for i in chosen])
    if len(selected) < max_pairs:
        remaining = [p for p in pairs if p not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_pairs - len(selected)])
    rng.shuffle(selected)
    return selected[:max_pairs]


def _gripper_boxes(
    opening: float,
    params: GripperParams,
    density: int,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return list of (name, box_transform_in_gripper, box_points_local)."""
    finger_h = params.finger_height if params.finger_height is not None else params.finger_thickness
    finger_extents = np.array([params.finger_thickness, params.finger_length, finger_h], dtype=np.float64)
    palm_extents = np.array([params.palm_width, params.palm_depth, params.palm_height], dtype=np.float64)

    finger_pts = _box_surface_points(finger_extents, density)
    palm_pts = _box_surface_points(palm_extents, density)

    x_offset = opening / 2.0 + params.finger_thickness / 2.0
    right_T = np.eye(4, dtype=np.float64)
    right_T[:3, 3] = np.array([x_offset, params.finger_length / 2.0, 0.0])
    left_T = np.eye(4, dtype=np.float64)
    left_T[:3, 3] = np.array([-x_offset, params.finger_length / 2.0, 0.0])

    palm_T = np.eye(4, dtype=np.float64)
    palm_T[:3, 3] = np.array([0.0, -params.palm_depth / 2.0, 0.0])

    return [
        ("finger_right", right_T, finger_pts),
        ("finger_left", left_T, finger_pts),
        ("palm", palm_T, palm_pts),
    ]


def _signed_distance(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    try:
        return trimesh.proximity.signed_distance(mesh, points)
    except Exception:
        nearest, dist, _ = mesh.nearest.on_surface(points)
        inside = mesh.contains(points)
        signed = dist.copy()
        signed[inside] *= -1.0
        return signed


def _check_collision(
    mesh: trimesh.Trimesh,
    T_gripper: np.ndarray,
    box_specs: list[tuple[str, np.ndarray, np.ndarray]],
    allowance: float,
    table_plane: Optional[tuple[np.ndarray, float]] = None,
) -> tuple[bool, float]:
    """Return (collision, min_signed_distance)."""
    min_sd = float("inf")
    for _, T_box, pts_local in box_specs:
        pts_world = _transform_points(T_gripper @ T_box, pts_local)
        if table_plane is not None:
            n, d = table_plane
            plane_sd = (pts_world @ n) - d
            if np.any(plane_sd < 0.0):
                return True, -float(np.min(plane_sd))
        sd = _signed_distance(mesh, pts_world)
        min_sd = min(min_sd, float(np.min(sd)))
        if np.any(sd < -allowance):
            return True, min_sd
    return False, min_sd


def _approach_directions(closing_axis: np.ndarray, num_dirs: int) -> list[np.ndarray]:
    closing_axis = _normalize(closing_axis)
    # Pick any vector not parallel to closing axis to form a basis.
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(helper, closing_axis))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = _normalize(np.cross(closing_axis, helper))
    v = _normalize(np.cross(closing_axis, u))
    dirs = []
    for k in range(num_dirs):
        theta = (2.0 * math.pi * k) / num_dirs
        dirs.append(_normalize(math.cos(theta) * u + math.sin(theta) * v))
    return dirs


def construct_gripper_pose(
    p_i: np.ndarray,
    p_j: np.ndarray,
    mesh: trimesh.Trimesh,
    params: GripperParams,
    planner: PlannerParams,
    table_plane: Optional[tuple[np.ndarray, float]] = None,
) -> Optional[tuple[np.ndarray, float]]:
    """Return (pose, clearance) or None."""
    closing_axis = _normalize(p_j - p_i)
    if np.linalg.norm(closing_axis) < 1e-9:
        return None
    midpoint = 0.5 * (p_i + p_j)
    opening = float(np.linalg.norm(p_j - p_i))
    box_specs = _gripper_boxes(opening, params, planner.box_point_density)

    best_pose = None
    best_clearance = -float("inf")
    for approach in _approach_directions(closing_axis, planner.approach_directions):
        y_axis = _normalize(np.cross(approach, closing_axis))
        if np.linalg.norm(y_axis) < 1e-6:
            continue
        z_axis = _normalize(np.cross(closing_axis, y_axis))
        R = np.column_stack([closing_axis, y_axis, z_axis])
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = midpoint
        # Shift so that the finger tips (y = finger_length) align to the midpoint.
        T[:3, 3] += R[:, 1] * (params.finger_length / 2.0)

        collision, clearance = _check_collision(
            mesh,
            T,
            box_specs,
            allowance=planner.collision_allowance,
            table_plane=table_plane,
        )
        if collision:
            continue
        if clearance > best_clearance:
            best_clearance = clearance
            best_pose = T

    if best_pose is None:
        return None
    return best_pose, float(best_clearance)


def _check_approach_sweep(
    mesh: trimesh.Trimesh,
    T_gripper: np.ndarray,
    params: GripperParams,
    planner: PlannerParams,
    opening: float,
    table_plane: Optional[tuple[np.ndarray, float]] = None,
) -> bool:
    z_axis = T_gripper[:3, 2]
    sweep_dist = params.finger_length + params.palm_depth
    box_specs = _gripper_boxes(opening, params, planner.box_point_density)
    for step in range(planner.approach_steps):
        alpha = step / max(1, planner.approach_steps - 1)
        offset = -z_axis * sweep_dist * (1.0 - alpha)
        T = T_gripper.copy()
        T[:3, 3] += offset
        collision, _ = _check_collision(
            mesh,
            T,
            box_specs,
            allowance=planner.collision_allowance,
            table_plane=table_plane,
        )
        if collision:
            return False
    return True


def _score_candidate(
    midpoint: np.ndarray,
    com: np.ndarray,
    distance: float,
    normal_i: np.ndarray,
    normal_j: np.ndarray,
    clearance: float,
    params: GripperParams,
    mesh_scale: float,
) -> tuple[float, float, float, float, float]:
    dot = float(np.dot(_normalize(normal_i), _normalize(normal_j)))
    normal_opposition = (1.0 - dot) / 2.0
    clearance_score = max(0.0, min(1.0, clearance / (2.0 * max(1e-6, params.finger_thickness))))
    com_dist = float(np.linalg.norm(midpoint - com))
    com_score = math.exp(-com_dist / max(1e-6, 0.5 * mesh_scale))
    target = 0.6 * params.max_width
    sigma = 0.35 * params.max_width
    separation_score = math.exp(-((distance - target) ** 2) / (2.0 * sigma * sigma))

    score = (
        0.4 * normal_opposition
        + 0.3 * clearance_score
        + 0.2 * com_score
        + 0.1 * separation_score
    )
    return score, normal_opposition, clearance_score, com_score, separation_score


def plan_grasps(
    mesh: trimesh.Trimesh,
    gripper: GripperParams,
    planner: PlannerParams,
    top_k: int = 10,
    table_plane: Optional[tuple[np.ndarray, float]] = None,
    seed: int = 0,
) -> list[GraspCandidate]:
    points, normals = sample_surface_points(mesh, planner.num_surface_points, seed=seed)
    pairs = generate_contact_pairs(
        points,
        normals,
        max_width=gripper.max_width,
        min_width=planner.min_width,
        normal_dot_threshold=planner.normal_dot_threshold,
        max_pairs=planner.max_pairs,
        seed=seed,
    )
    if not pairs:
        return []

    com = mesh.center_mass
    bbox_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))

    candidates: list[GraspCandidate] = []
    for i, j, dist in pairs:
        p_i = points[i]
        p_j = points[j]
        result = construct_gripper_pose(
            p_i,
            p_j,
            mesh,
            gripper,
            planner,
            table_plane=table_plane,
        )
        if result is None:
            continue
        T, clearance = result
        if not _check_approach_sweep(
            mesh,
            T,
            gripper,
            planner,
            opening=dist,
            table_plane=table_plane,
        ):
            continue
        midpoint = 0.5 * (p_i + p_j)
        score, normal_opposition, _, _, _ = _score_candidate(
            midpoint,
            com,
            dist,
            normals[i],
            normals[j],
            clearance,
            gripper,
            bbox_diag,
        )
        candidates.append(
            GraspCandidate(
                pose=T,
                score=score,
                midpoint=midpoint,
                distance=dist,
                normal_opposition=normal_opposition,
                clearance=clearance,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k]


def _gripper_meshes(T: np.ndarray, opening: float, params: GripperParams) -> Iterable[trimesh.Trimesh]:
    finger_h = params.finger_height if params.finger_height is not None else params.finger_thickness
    finger_extents = [params.finger_thickness, params.finger_length, finger_h]
    palm_extents = [params.palm_width, params.palm_depth, params.palm_height]
    x_offset = opening / 2.0 + params.finger_thickness / 2.0

    right_T = np.eye(4, dtype=np.float64)
    right_T[:3, 3] = np.array([x_offset, params.finger_length / 2.0, 0.0])
    left_T = np.eye(4, dtype=np.float64)
    left_T[:3, 3] = np.array([-x_offset, params.finger_length / 2.0, 0.0])
    palm_T = np.eye(4, dtype=np.float64)
    palm_T[:3, 3] = np.array([0.0, -params.palm_depth / 2.0, 0.0])

    for T_box, extents, color in (
        (right_T, finger_extents, [220, 40, 40, 200]),
        (left_T, finger_extents, [220, 40, 40, 200]),
        (palm_T, palm_extents, [40, 40, 220, 200]),
    ):
        box = trimesh.creation.box(extents=extents)
        box.apply_transform(T @ T_box)
        box.visual.face_colors = color
        yield box


def _resolve_mesh_path(mesh_dir: str, mesh_path: str, object_id: str) -> str:
    if mesh_path:
        return str(Path(mesh_path).expanduser().resolve())
    mesh_dir_path = Path(mesh_dir).expanduser().resolve()
    if not mesh_dir_path.is_dir():
        raise FileNotFoundError(f"Mesh dir not found: {mesh_dir_path}")
    candidates = [
        mesh_dir_path / f"{object_id}.stl",
        mesh_dir_path / f"{object_id}.STL",
        mesh_dir_path / f"{object_id}.ply",
        mesh_dir_path / f"{object_id}.obj",
    ]
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(f"No mesh found for object_id={object_id} in {mesh_dir_path}")


def _demo(mesh_path: str, top_k: int, seed: int) -> None:
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Loaded mesh is not a trimesh.Trimesh")
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.process(validate=True)
    bbox_center = 0.5 * (mesh.bounds[0] + mesh.bounds[1])
    mesh.apply_translation(-bbox_center)
    bbox_size = mesh.bounds[1] - mesh.bounds[0]
    print(
        "Object bbox size (m): "
        f"{bbox_size[0]:.4f}, {bbox_size[1]:.4f}, {bbox_size[2]:.4f}"
    )

    gripper = GripperParams(
        max_width=0.15,
        finger_thickness=0.01,
        finger_length=0.06,
        palm_width=0.10,
        palm_depth=0.02,
        palm_height=0.04,
    )
    planner = PlannerParams()

    grasps = plan_grasps(mesh, gripper, planner, top_k=top_k, seed=seed)
    if not grasps:
        print("No grasps found.")
        return
    print(f"Found {len(grasps)} grasps. Best score: {grasps[0].score:.3f}")

    scene = trimesh.Scene()
    mesh.visual.face_colors = [180, 180, 180, 255]
    scene.add_geometry(mesh)
    axis_len = 0.6 * float(np.max(bbox_size))
    axis = trimesh.load_path(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [axis_len, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, axis_len, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, axis_len],
            ],
            dtype=np.float64,
        )
    )
    scene.add_geometry(axis, node_name="axis")

    for idx, grasp in enumerate(grasps):
        for part_idx, geom in enumerate(_gripper_meshes(grasp.pose, grasp.distance, gripper)):
            scene.add_geometry(geom, node_name=f"gripper_{idx}_{part_idx}")
    scene.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple parallel-jaw grasp planner demo.")
    parser.add_argument("--mesh", default="", help="Path to mesh file (overrides mesh_dir/object_id)")
    parser.add_argument(
        "--mesh_dir",
        type=str,
        default="/home/kevin/ICL/rendering_prompted_muggled_sam/assets/wbcd_meshes",
    )
    parser.add_argument("--mesh_path", type=str, default="")
    parser.add_argument("--object_id", type=str, default="cube")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    mesh_path = args.mesh if args.mesh else args.mesh_path
    mesh_path = _resolve_mesh_path(args.mesh_dir, mesh_path, args.object_id)
    _demo(mesh_path, args.top_k, args.seed)


if __name__ == "__main__":
    main()
