#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Naive parallel-jaw grasp planner.

Assumptions:
- Gripper approaches straight down with opening face along -Z.
- Sample contact pairs only near 50% height of object (z midplane).
- Grasp midpoint (finger tips) is midpoint between the contact pair.
- No collision or approach checks.
"""

from __future__ import annotations

import argparse
import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import open3d as o3d

ICP_VIEW_FRONT = [0.9288, -0.2951, -0.2242]
ICP_VIEW_UP = [-0.3402, -0.9189, -0.1996]
GRASP_VIEW_ZOOM = 0.7
VIS_FRAME_FLIP_Y = np.diag([1.0, -1.0, 1.0, 1.0])


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
    num_surface_points: int = 6000
    max_pairs: int = 3000
    normal_dot_threshold: float = -0.3
    min_width: float = 0.01
    z_band_ratio: float = 0.01
    use_convex_hull: bool = False


@dataclass(frozen=True)
class GraspCandidate:
    pose: np.ndarray
    score: float
    midpoint: np.ndarray
    distance: float
    normal_opposition: float
    grasp_outer_edge: float
    contact_i: np.ndarray
    contact_j: np.ndarray
    normal_i: np.ndarray
    normal_j: np.ndarray


def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v.copy()
    return v / n


def _sampling_mesh(
    mesh: o3d.geometry.TriangleMesh,
    use_convex_hull: bool = False,
) -> o3d.geometry.TriangleMesh:
    if not use_convex_hull:
        sampling_mesh = mesh
    else:
        hull_result = mesh.compute_convex_hull()
        sampling_mesh = hull_result[0] if isinstance(hull_result, tuple) else hull_result
    sampling_mesh = o3d.geometry.TriangleMesh(sampling_mesh)
    sampling_mesh.compute_triangle_normals()
    sampling_mesh.compute_vertex_normals()
    return sampling_mesh


def sample_surface_points(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int,
    seed: int = 0,
    use_convex_hull: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    _ = seed
    sampling_mesh = _sampling_mesh(mesh, use_convex_hull=use_convex_hull)
    pcd = sampling_mesh.sample_points_uniformly(number_of_points=num_points)
    points = np.asarray(pcd.points, dtype=np.float64)
    normals = np.asarray(pcd.normals, dtype=np.float64)
    return points, normals


def generate_contact_pairs(
    points: np.ndarray,
    normals: np.ndarray,
    max_width: float,
    min_width: float,
    normal_dot_threshold: float,
    max_pairs: int,
    seed: int = 0,
) -> list[tuple[int, int, float]]:
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
    rng.shuffle(pairs)
    return pairs[:max_pairs]


def _topdown_pose(p_i: np.ndarray, p_j: np.ndarray, finger_length: float, z_center: float) -> Optional[np.ndarray]:
    closing_axis = p_j - p_i
    closing_axis[2] = 0.0
    closing_axis = _normalize(closing_axis)
    if np.linalg.norm(closing_axis) < 1e-9:
        return None
    # Gripper opening faces down: finger length axis aligns with -Z.
    y_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    x_axis = closing_axis
    z_axis = _normalize(np.cross(x_axis, y_axis))
    if np.linalg.norm(z_axis) < 1e-9:
        z_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = _normalize(np.cross(y_axis, z_axis))
    else:
        x_axis = _normalize(np.cross(y_axis, z_axis))
    R = np.column_stack([x_axis, y_axis, z_axis])
    midpoint = 0.5 * (p_i + p_j)
    midpoint[2] = float(z_center)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    # Place fingertip plane midpoint at the contact midpoint.
    T[:3, 3] = midpoint - R[:, 1] * finger_length
    return T


def _score_candidate(
    contact_i: np.ndarray,
    contact_j: np.ndarray,
    normal_i: np.ndarray,
    normal_j: np.ndarray,
    closing_axis: np.ndarray,
    midpoint: np.ndarray,
    obj_center: np.ndarray,
    obj_scale: float,
    xy_scale: float,
    axis_vec: np.ndarray,
) -> tuple[float, float, float, float]:
    def _outer_edge_term(contact: np.ndarray) -> float:
        radial_xy = float(np.linalg.norm((contact - obj_center)[:2]))
        return math.sqrt(max(0.0, radial_xy) / max(1e-6, xy_scale))

    dot = float(np.dot(_normalize(normal_i), _normalize(normal_j)))
    normal_opposition = (1.0 - dot) / 2.0
    axis = _normalize(closing_axis)
    perp_i = abs(float(np.dot(_normalize(normal_i), axis)))
    perp_j = abs(float(np.dot(_normalize(normal_j), axis)))
    perp_score = 0.5 * (perp_i + perp_j)
    grasp_outer_edge = 0.5 * (_outer_edge_term(contact_i) + _outer_edge_term(contact_j))
    center_dist = float(np.linalg.norm(midpoint - obj_center))
    center_score = math.exp(-center_dist / max(1e-6, 0.5 * obj_scale))
    score = 0.45 * normal_opposition + 0.5 * perp_score + 0.25 * center_score

    face_score = 0.5 * (
        abs(float(np.dot(_normalize(normal_i), axis_vec)))
        + abs(float(np.dot(_normalize(normal_j), axis_vec)))
    )
    score += 0.2 * face_score
    # score += 0.15 * grasp_outer_edge

    # Penalize inward-pointing normals (toward grasp midpoint).
    to_mid_i = _normalize(midpoint - contact_i)
    to_mid_j = _normalize(midpoint - contact_j)
    if float(np.dot(_normalize(normal_i), to_mid_i)) > 0.0 or float(np.dot(_normalize(normal_j), to_mid_j)) > 0.0:
        score *= 0.2

    return score, normal_opposition, perp_score, grasp_outer_edge


def plan_grasps(
    mesh: o3d.geometry.TriangleMesh,
    gripper: GripperParams,
    planner: PlannerParams,
    top_k: int = 10,
    seed: int = 0,
    z_center: float = 0.0,
) -> list[GraspCandidate]:
    points, normals = sample_surface_points(
        mesh,
        planner.num_surface_points,
        seed=seed,
        use_convex_hull=planner.use_convex_hull,
    )
    aabb = mesh.get_axis_aligned_bounding_box()
    z_min, z_max = float(aabb.min_bound[2]), float(aabb.max_bound[2])
    z_mid = float(z_min + 0.5 * (z_max - z_min))
    z_band = planner.z_band_ratio * (z_max - z_min)
    band_mask = np.abs(points[:, 2] - z_mid) <= z_band
    points = points[band_mask]
    normals = normals[band_mask]
    if points.shape[0] < 2:
        return []

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
    obj_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    obj_scale = float(np.linalg.norm(aabb.get_extent()))
    xy_scale = float(min(aabb.get_extent()[0], aabb.get_extent()[1]))
    short_axis = int(np.argmin(aabb.get_extent()))
    axis_vec = np.eye(3, dtype=np.float64)[short_axis]

    candidates: list[GraspCandidate] = []
    for i, j, dist in pairs:
        p_i = points[i]
        p_j = points[j]
        T = _topdown_pose(p_i, p_j, gripper.finger_length, z_center=z_mid)
        if T is None:
            continue
        tip_mid = (T @ np.array([0.0, gripper.finger_length, 0.0, 1.0], dtype=np.float64))[:3]
        closing_axis = p_j - p_i
        score, normal_opposition, _, grasp_outer_edge = _score_candidate(
            p_i,
            p_j,
            normals[i],
            normals[j],
            closing_axis,
            midpoint=tip_mid,
            obj_center=obj_center,
            obj_scale=obj_scale,
            xy_scale=xy_scale,
            axis_vec=axis_vec,
        )
        candidates.append(
            GraspCandidate(
                pose=T,
                score=score,
                midpoint=tip_mid,
                distance=dist,
                normal_opposition=normal_opposition,
                grasp_outer_edge=grasp_outer_edge,
                contact_i=p_i.copy(),
                contact_j=p_j.copy(),
                normal_i=normals[i].copy(),
                normal_j=normals[j].copy(),
            )
        )
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k]


def _gripper_meshes(
    T: np.ndarray,
    opening: float,
    params: GripperParams,
    color: Optional[np.ndarray] = None,
) -> Iterable[o3d.geometry.TriangleMesh]:
    finger_h = params.finger_height if params.finger_height is not None else params.finger_thickness
    finger_extents = np.array([params.finger_thickness, params.finger_length, finger_h], dtype=np.float64)
    palm_extents = np.array([params.palm_width, params.palm_depth, params.palm_height], dtype=np.float64)
    x_offset = opening / 2.0 + params.finger_thickness / 2.0

    right_T = np.eye(4, dtype=np.float64)
    right_T[:3, 3] = np.array([x_offset, params.finger_length / 2.0, 0.0])
    left_T = np.eye(4, dtype=np.float64)
    left_T[:3, 3] = np.array([-x_offset, params.finger_length / 2.0, 0.0])
    palm_T = np.eye(4, dtype=np.float64)
    palm_T[:3, 3] = np.array([0.0, -params.palm_depth / 2.0, 0.0])

    for T_box, extents, default_color in (
        (right_T, finger_extents, [220, 40, 40, 200]),
        (left_T, finger_extents, [220, 40, 40, 200]),
        (palm_T, palm_extents, [40, 40, 220, 200]),
    ):
        box = o3d.geometry.TriangleMesh.create_box(
            width=float(extents[0]),
            height=float(extents[1]),
            depth=float(extents[2]),
        )
        box.translate(-extents / 2.0)
        box.transform(T @ T_box)
        use_color = color if color is not None else np.array(default_color[:3], dtype=np.float64) / 255.0
        box.paint_uniform_color(use_color)
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


def _transparent_material(alpha: float) -> "o3d.visualization.rendering.MaterialRecord":
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLitTransparency"
    mat.base_color = (0.7, 0.7, 0.7, float(alpha))
    return mat


def _transparent_material_color(color: np.ndarray, alpha: float) -> "o3d.visualization.rendering.MaterialRecord":
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLitTransparency"
    mat.base_color = (float(color[0]), float(color[1]), float(color[2]), float(alpha))
    return mat


def _score_color(score: float, min_score: float, max_score: float) -> np.ndarray:
    if max_score <= min_score + 1e-9:
        t = 0.0
    else:
        t = (score - min_score) / (max_score - min_score)
    # Blue -> Red
    return np.array([t, 0.0, 1.0 - t], dtype=np.float64)


def _normal_arrow(center: np.ndarray, normal: np.ndarray, length: float) -> o3d.geometry.TriangleMesh:
    n = _normalize(normal)
    shaft_radius = 0.01 * length
    cone_radius = 0.02 * length
    shaft_length = 0.8 * length
    cone_length = 0.2 * length
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=shaft_radius,
        cone_radius=cone_radius,
        cylinder_height=shaft_length,
        cone_height=cone_length,
    )
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    v = np.cross(z_axis, n)
    c = float(np.dot(z_axis, n))
    if np.linalg.norm(v) < 1e-9:
        if c > 0.0:
            rot = np.eye(3, dtype=np.float64)
        else:
            rot = o3d.geometry.get_rotation_matrix_from_axis_angle([math.pi, 0.0, 0.0])
    else:
        axis = v / np.linalg.norm(v)
        angle = math.atan2(np.linalg.norm(v), c)
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
    arrow.rotate(rot, center=[0.0, 0.0, 0.0])
    arrow.translate(center)
    arrow.paint_uniform_color([1.0, 0.9, 0.1])
    arrow.compute_vertex_normals()
    return arrow


def apply_temporary_display_transform(
    geometries: Iterable[o3d.geometry.Geometry],
) -> list[o3d.geometry.Geometry]:
    # Temporary visualization-only adjustment: rotate +90 deg about Y, then -90 deg about the rotated local X axis.
    display_rot = np.eye(4, dtype=np.float64)
    display_rot[:3, :3] = (
        o3d.geometry.get_rotation_matrix_from_xyz((0.0, math.pi / 2, 0.0))

    )
    out = list(geometries)
    for geom in out:
        geom.transform(display_rot)
    return out


def demo(
    mesh_path: str,
    top_k: int = 2,
    seed: int = 0,
    rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    use_convex_hull: bool = False,
) -> Optional[np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise ValueError("Loaded mesh is empty or invalid.")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    # Keep mesh axes as-is (assume Z-up); only recenter to bbox center.
    aabb = mesh.get_axis_aligned_bounding_box()
    bbox_size = aabb.get_extent()
    if float(np.max(bbox_size)) > 2.0:
        mesh.scale(0.001, center=[0.0, 0.0, 0.0])
        aabb = mesh.get_axis_aligned_bounding_box()
        bbox_size = aabb.get_extent()
    bbox_center = aabb.get_center()
    mesh.translate(-bbox_center)
    if any(abs(float(v)) > 1e-6 for v in rpy_deg):
        rpy_rad = np.deg2rad([rpy_deg[0], rpy_deg[1], rpy_deg[2]])
        rot = o3d.geometry.get_rotation_matrix_from_xyz(rpy_rad)
        mesh.rotate(rot, center=[0.0, 0.0, 0.0])
    aabb = mesh.get_axis_aligned_bounding_box()
    bbox_size = aabb.get_extent()
    z_center = 0.0
    print(
        "Object bbox size (m): "
        f"{bbox_size[0]:.4f}, {bbox_size[1]:.4f}, {bbox_size[2]:.4f}"
    )
    print("Using mesh frame (Z-up); no axis reorientation applied.")

    gripper = GripperParams(
        max_width=0.09,
        finger_thickness=0.01,
        finger_length=0.06,
        palm_width=0.10,
        palm_depth=0.01,
        palm_height=0.01,
    )
    planner = PlannerParams(use_convex_hull=use_convex_hull)

    t_grasp_calc_start = time.perf_counter()
    points, normals = sample_surface_points(
        mesh,
        planner.num_surface_points,
        seed=seed,
        use_convex_hull=planner.use_convex_hull,
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    colors = 0.5 * (normals + 1.0)
    colors = np.clip(colors, 0.0, 1.0)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw([{"name": "pcd", "geometry": pcd}], title="Sampled Point Cloud")

    grasps = plan_grasps(mesh, gripper, planner, top_k=top_k, seed=seed, z_center=z_center)
    if not grasps:
        print("No grasps found.")
        return None
    print(f"Found {len(grasps)} grasps. Best score: {grasps[0].score:.3f}")
    grasp_calc_time_s = time.perf_counter() - t_grasp_calc_start
    print(f"Grasp calculation time: {grasp_calc_time_s * 1000.0:.1f} ms")

    axis_len = 0.6 * float(np.max(bbox_size))
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_len, origin=[0.0, 0.0, 0.0])
    axis.transform(VIS_FRAME_FLIP_Y)
    axis.compute_vertex_normals()
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.7])

    geometries = [
        {"name": "mesh", "geometry": mesh, "material": _transparent_material(0.6)},
        {"name": "axis", "geometry": axis},
    ]
    gidx = 0
    scores = [g.score for g in grasps]
    min_score = float(min(scores))
    max_score = float(max(scores))
    for idx, grasp in enumerate(grasps):
        grasp_color = _score_color(grasp.score, min_score, max_score)
        midpoint_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02 * float(np.max(bbox_size)))
        midpoint_sphere.translate(grasp.midpoint)
        midpoint_sphere.paint_uniform_color([1.0, 0.8, 0.1])
        midpoint_sphere.compute_vertex_normals()
        geometries.append({"name": f"midpoint_{idx}", "geometry": midpoint_sphere})
        print(
            f"Grasp {idx} contacts: "
            f"p_i=({grasp.contact_i[0]:.4f}, {grasp.contact_i[1]:.4f}, {grasp.contact_i[2]:.4f}) "
            f"p_j=({grasp.contact_j[0]:.4f}, {grasp.contact_j[1]:.4f}, {grasp.contact_j[2]:.4f})"
        )
        for c_idx, contact in enumerate((grasp.contact_i, grasp.contact_j)):
            contact_sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=0.015 * float(np.max(bbox_size))
            )
            contact_sphere.translate(contact)
            contact_sphere.paint_uniform_color([0.9, 0.2, 0.9])
            contact_sphere.compute_vertex_normals()
            geometries.append({"name": f"contact_{idx}_{c_idx}", "geometry": contact_sphere})
            normal = grasp.normal_i if c_idx == 0 else grasp.normal_j
            arrow = _normal_arrow(contact, normal, length=0.12 * float(np.max(bbox_size)))
            geometries.append({"name": f"normal_arrow_{idx}_{c_idx}", "geometry": arrow})
        contact_line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(
                np.vstack([grasp.contact_i, grasp.contact_j])
            ),
            lines=o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32)),
        )
        contact_line.colors = o3d.utility.Vector3dVector([[0.9, 0.2, 0.9]])
        geometries.append({"name": f"contact_line_{idx}", "geometry": contact_line})
        tip_offset = grasp.distance / 2.0
        tip_local = np.array(
            [
                [tip_offset, gripper.finger_length, 0.0, 1.0],
                [-tip_offset, gripper.finger_length, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        tip_world = (grasp.pose @ tip_local.T).T[:, :3]
        for t_idx, tip in enumerate(tip_world):
            tip_sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=0.015 * float(np.max(bbox_size))
            )
            tip_sphere.translate(tip)
            tip_sphere.paint_uniform_color([0.1, 0.9, 0.3])
            tip_sphere.compute_vertex_normals()
            geometries.append({"name": f"tip_{idx}_{t_idx}", "geometry": tip_sphere})
        for geom in _gripper_meshes(grasp.pose, grasp.distance, gripper, color=grasp_color):
            geom.compute_vertex_normals()
            geometries.append(
                {
                    "name": f"gripper_{gidx}",
                    "geometry": geom,
                    "material": _transparent_material_color(grasp_color, 0.35),
                }
            )
            gidx += 1
    draw_geometries = [
        geom["geometry"] if isinstance(geom, dict) else geom
        for geom in geometries
    ]
    draw_geometries = apply_temporary_display_transform(draw_geometries)
    o3d.visualization.draw_geometries(
        draw_geometries,
        window_name="Naive Grasp Planner",
        zoom=GRASP_VIEW_ZOOM,
        front=ICP_VIEW_FRONT,
        lookat=[0.0, 0.0, 0.0],
        up=ICP_VIEW_UP,
    )
    return grasps[0].pose.copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Naive parallel-jaw grasp planner demo.")
    parser.add_argument("--mesh", default="", help="Path to mesh file (overrides mesh_dir/object_id)")
    # parser.add_argument(
    #     "--mesh_dir",
    #     type=str,
    #     default="/home/kevin/ICL/rendering_prompted_muggled_sam/assets/wbcd_meshes",
    # )
    parser.add_argument(
        "--mesh_dir",
        type=str,
        default="assets/mesh_0316",
    )
    parser.add_argument("--mesh_path", type=str, default="")
    parser.add_argument("--object_id", type=str, default="rod_mount")
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use_convex_hull",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If true, sample grasp contacts on the mesh convex hull instead of the raw mesh surface.",
    )
    args = parser.parse_args()
    mesh_path = args.mesh if args.mesh else args.mesh_path
    mesh_path = _resolve_mesh_path(args.mesh_dir, mesh_path, args.object_id)
    demo(mesh_path, args.top_k, args.seed, use_convex_hull=args.use_convex_hull)


if __name__ == "__main__":
    main()
