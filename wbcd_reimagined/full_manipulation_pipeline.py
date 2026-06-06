#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import math
import socket
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wbcd_reimagined.grasp_methods import GraspMethods, GraspSpec
from wbcd_reimagined.naive_grasp_planner import demo as naive_grasp_demo
from wbcd_reimagined.naive_grasp_planner import GripperParams, PlannerParams, plan_grasps
from wbcd_reimagined.run_realsense_exemplar_pose import (
    ExemplarPosePipeline,
    Realsense,
    _compose_detection_icp_vis,
    _resolve_mesh_path,
    get_pose_and_pointcloud_from_mask,
    get_pose_from_mask,
    load_detector_model,
)


T_RC = np.array(
    [
        [0.0, 0.0, 1.0, 0.17],
        [-1.0, 0.0, 0.0, 0.05],
        [0.0, -1.0, 0.0, 0.03],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

DEFAULT_CMD = (0.3, 0.15, 0.1, 180.0, 0.0, 90.0, 0.0)
FIXED_TAIL = (180.0, 0.0, 90.0, 0.0)
MIN_ROBOT_Z = -0.07
DROP_OFF_POSES = {
    "gear": (0.3, 0.30, 0.04, 180.0, 0.0, 90.0, 0.0),
    "rod_mount": (0.43, 0.30, 0.04, 180.0, 0.0, 90.0, 0.0),
    "bowl": (0.43, 0.40, 0.04, 180.0, 0.0, 90.0, 0.0),
    "cube": (0.43, 0.38, 0.04, 180.0, 0.0, 90.0, 0.0),
    "square_tube": (0.43, 0.30, 0.04, 180.0, 0.0, 90.0, 0.0),
    "bolt": (0.43, 0.40, 0.04, 180.0, 0.0, 90.0, 0.0),
    "lego_brick": (0.43, 0.40, 0.04, 180.0, 0.0, 90.0, 0.0),
}
GUARD_X = (0.05, 0.70)
GUARD_Y = (-0.4, 0.4)
GUARD_Z = (-0.12, 0.3)


GRASP_METHODS = GraspMethods()
GRASP_METHODS.register("bowl", GraspSpec(0.0, -0.05, 0.0, gripper_close_pos=0.98, use_object_yaw=False, use_convex_hull=False))
GRASP_METHODS.register("cube", GraspSpec(0.0, 0.0, 0.0, gripper_close_pos=0.5, use_object_yaw=True, use_convex_hull=False))
GRASP_METHODS.register("gear", GraspSpec(0.0, 0.0, 0.0, gripper_close_pos=0.8, use_object_yaw=True, use_convex_hull=True))
GRASP_METHODS.register("rod_mount", GraspSpec(0.0, 0.0, 0.0, gripper_close_pos=0.8, use_object_yaw=True, use_convex_hull=False))
GRASP_METHODS.register("square_tube", GraspSpec(0.0, 0.0, 0.0, gripper_close_pos=0.7, use_object_yaw=True, use_convex_hull=True))
GRASP_METHODS.register("bolt", GraspSpec(0.0, 0.0, 0.0, gripper_close_pos=0.85, use_object_yaw=True, use_convex_hull=True))
GRASP_METHODS.register("lego_brick", GraspSpec(0.0, 0.0, 0.0, gripper_close_pos=0.8, use_object_yaw=True, use_convex_hull=True))

DEFAULT_REFERENCE_DIR = [
    "/home/kevin/ICL/rendering_prompted_muggled_sam/assets/renders_2442_0316",
    "/home/kevin/ICL/rendering_prompted_muggled_sam/assets/wbcd_renders_2442_0316",
]
DEFAULT_MESH_DIRS = [
    "/home/kevin/ICL/rendering_prompted_muggled_sam/assets/wbcd_meshes",
    "/home/kevin/ICL/rendering_prompted_muggled_sam/assets/mesh_0316",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full manipulation pipeline with one or more objects.")
    parser.add_argument("--model_path", type=str, default="/home/kevin/ICL/rendering_prompted_muggled_sam/model_weights/sam3.pt")
    parser.add_argument("--finetune_ckpt", type=str, default="model_weights/0321_k12_b156_resume_from_preprinte18_s1_e34.pth")
    # parser.add_argument("--reference_dir", type=str, default="/home/kevin/ICL/rendering_prompted_muggled_sam/assets/wbcd_renders_2442_0316")
    # parser.add_argument("--mesh_dir", type=str, default="/home/kevin/ICL/rendering_prompted_muggled_sam/assets/wbcd_meshes")
    parser.add_argument(
        "--reference_dir",
        type=str,
        action="append",
        default=None,
        help=(
            "Reference dir(s). Repeat flag or pass comma-separated or list string. "
            f"Default: {DEFAULT_REFERENCE_DIR}"
        ),
    )
    parser.add_argument(
        "--mesh_dir",
        type=str,
        action="append",
        default=None,
        help=(
            "Mesh dir(s). Repeat flag or pass comma-separated or list string. "
            f"Default: {DEFAULT_MESH_DIRS}"
        ),
    )
    parser.add_argument("--mesh_path", type=str, default="")
    parser.add_argument("--object_ids", type=str, default="gear")
    parser.add_argument("--ref_view_ids", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--max_side_length", type=int, default=1008)
    parser.add_argument("--no_square", action="store_true")
    parser.add_argument("--num_points_approx", type=int, default=24)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, choices=["fp32", "bf16"], default="")
    parser.add_argument("--det_filter", type=float, default=0.0)
    parser.add_argument("--nms_iou", type=float, default=0.5)
    parser.add_argument("--max_show", type=int, default=3)
    parser.add_argument("--max_objects", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--visualize_icp", default = False, action="store_true")
    parser.add_argument("--rs_timeout_ms", type=int, default=50000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--dryrun", default=True, action="store_true", help="Run without robot connection.")
    parser.add_argument(
        "--predict_grasp_pose",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Predict grasp pose from mesh using naive_grasp_planner.",
    )
    parser.add_argument(
        "--replane_with_pose",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="If true, rotate mesh by detected pose before planning, then translate planned grasp.",
    )
    parser.add_argument(
        "--snap_rp_90",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="If true and replane_with_pose, snap roll/pitch to nearest 90 degrees for planning.",
    )
    parser.add_argument(
        "--grasp_z_comp",
        type=float,
        default=-0.065,
        help="Additive Z compensation (meters) applied to planned grasp pose.",
    )
    return parser.parse_args()


def _parse_dir_list(value: Optional[object], name: str) -> list[str]:
    if value is None:
        return []
    raw = list(value) if isinstance(value, (list, tuple)) else [value]
    out: list[str] = []
    for item in raw:
        if item is None:
            continue
        if not isinstance(item, str):
            raise TypeError(f"{name} must be a string or list of strings.")
        s = item.strip()
        if not s:
            continue
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"{name} list string could not be parsed: {s}") from exc
            if not isinstance(parsed, (list, tuple)):
                raise ValueError(f"{name} list string must evaluate to a list of strings.")
            for elem in parsed:
                if not isinstance(elem, str) or not elem.strip():
                    raise ValueError(f"{name} list string must contain only non-empty strings.")
                out.append(elem.strip())
        else:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            out.extend(parts)
    return out


def _assign_dirs(object_ids: list[str], dirs: list[str], name: str) -> dict[str, str]:
    if len(dirs) == 1:
        return {obj_id: dirs[0] for obj_id in object_ids}
    if len(dirs) == len(object_ids):
        return {obj_id: dirs[i] for i, obj_id in enumerate(object_ids)}
    raise ValueError(
        f"{name} must have 1 entry or match object_ids length ({len(object_ids)}). Got {len(dirs)}."
    )


def _resolve_mesh_path_any(mesh_dirs: list[str], mesh_path: str, object_id: str) -> tuple[Path, str]:
    if mesh_path:
        resolved = Path(mesh_path).expanduser().resolve()
        return resolved, str(resolved.parent)
    last_err = None
    for mesh_dir in mesh_dirs:
        mesh_dir_path = Path(mesh_dir).expanduser().resolve()
        try:
            resolved = _resolve_mesh_path(mesh_dir_path, mesh_path, object_id)
        except FileNotFoundError as exc:
            last_err = exc
            continue
        return resolved, str(mesh_dir_path)
    searched = ", ".join(str(Path(d).expanduser().resolve()) for d in mesh_dirs)
    if last_err is not None:
        raise FileNotFoundError(
            f"No mesh found for object_id={object_id} in any of: {searched}"
        ) from last_err
    raise FileNotFoundError(f"No mesh found for object_id={object_id} in any of: {searched}")


def _resolve_reference_dir_any(reference_dirs: list[str], object_id: str) -> str:
    for ref_dir in reference_dirs:
        ref_dir_path = Path(ref_dir).expanduser().resolve()
        if not ref_dir_path.is_dir():
            continue
        mask_paths = ref_dir_path.glob(f"{object_id}_stl_base_*_mask.png")
        for mask_path in mask_paths:
            img_path = mask_path.with_name(mask_path.name.replace("_mask.png", ".png"))
            if img_path.is_file():
                return str(ref_dir_path)
    searched = ", ".join(str(Path(d).expanduser().resolve()) for d in reference_dirs)
    raise FileNotFoundError(f"No reference views found for object_id={object_id} in any of: {searched}")


def _best_pose(poses, scores_n: np.ndarray) -> Optional[np.ndarray]:
    if scores_n.size == 0:
        return None
    idx = int(np.argmax(scores_n))
    if idx >= len(poses):
        return None
    return poses[idx]


def _transform_pose(T_rc: np.ndarray, T_co: np.ndarray) -> np.ndarray:
    return T_rc @ T_co


def _yaw_deg_from_pose(T_ro: np.ndarray) -> float:
    r = T_ro[:3, :3]
    yaw_rad = math.atan2(r[1, 0], r[0, 0])
    return math.degrees(yaw_rad)


def _rpy_deg_from_pose(T: np.ndarray) -> tuple[float, float, float]:
    r = T[:3, :3]
    sy = math.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    singular = sy < 1e-9
    if not singular:
        roll = math.atan2(r[2, 1], r[2, 2])
        pitch = math.atan2(-r[2, 0], sy)
        yaw = math.atan2(r[1, 0], r[0, 0])
    else:
        roll = math.atan2(-r[1, 2], r[1, 1])
        pitch = math.atan2(-r[2, 0], sy)
        yaw = 0.0
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def _snap_90(deg: float) -> float:
    return round(deg / 90.0) * 90.0


def _within_guard(x: float, y: float, z: float) -> bool:
    return (GUARD_X[0] <= x <= GUARD_X[1]) and (GUARD_Y[0] <= y <= GUARD_Y[1]) and (GUARD_Z[0] <= z <= GUARD_Z[1])


def _plan_grasp_pose_naive_mesh(
    mesh_base: o3d.geometry.TriangleMesh,
    use_convex_hull: bool = False,
    top_k: int = 1,
) -> Optional[np.ndarray]:
    if mesh_base.is_empty():
        return None
    mesh = o3d.geometry.TriangleMesh(mesh_base)
    aabb = mesh.get_axis_aligned_bounding_box()
    z_center = float(0.5 * (aabb.min_bound[2] + aabb.max_bound[2]))
    gripper = GripperParams(
        max_width=0.09,
        finger_thickness=0.01,
        finger_length=0.06,
        palm_width=0.10,
        palm_depth=0.01,
        palm_height=0.01,
    )
    planner = PlannerParams(use_convex_hull=use_convex_hull)
    grasps = plan_grasps(mesh, gripper, planner, top_k=top_k, seed=0, z_center=z_center)
    if not grasps:
        return None
    return grasps[0].pose.copy()


def _compose_planned_grasp(
    pose_robot: np.ndarray,
    grasp_pose_mesh: np.ndarray,
    z_comp: float,
) -> tuple[float, float, float, float]:
    object_yaw_deg = _yaw_deg_from_pose(pose_robot)
    grasp_yaw_deg = _yaw_deg_from_pose(grasp_pose_mesh)
    gx, gy, gz = grasp_pose_mesh[:3, 3].tolist()
    yaw_rad = math.radians(object_yaw_deg)
    gx_r = math.cos(yaw_rad) * gx - math.sin(yaw_rad) * gy
    gy_r = math.sin(yaw_rad) * gx + math.cos(yaw_rad) * gy
    x = pose_robot[0, 3] + gx_r
    y = pose_robot[1, 3] + gy_r
    z = pose_robot[2, 3] + gz + z_comp
    yaw = object_yaw_deg + grasp_yaw_deg
    return x, y, z, yaw


def main() -> None:
    args = parse_args()
    object_ids = [s.strip() for s in args.object_ids.split(",") if s.strip()]
    if len(object_ids) < 1:
        raise ValueError("object_ids must contain at least one id, e.g. gear or gear,rod_mount")

    ref_dirs = _parse_dir_list(args.reference_dir, "reference_dir")
    if not ref_dirs:
        ref_dirs = list(DEFAULT_REFERENCE_DIR)
    mesh_dirs = _parse_dir_list(args.mesh_dir, "mesh_dir")
    if not mesh_dirs:
        mesh_dirs = list(DEFAULT_MESH_DIRS)

    if len(ref_dirs) > 1:
        ref_dir_map = {obj_id: _resolve_reference_dir_any(ref_dirs, obj_id) for obj_id in object_ids}
    else:
        ref_dir_map = _assign_dirs(object_ids, ref_dirs, "reference_dir")

    mesh_paths = {}
    mesh_dir_map = {}
    for obj_id in object_ids:
        if len(mesh_dirs) > 1:
            mesh_path_resolved, mesh_dir_resolved = _resolve_mesh_path_any(
                mesh_dirs, args.mesh_path, obj_id
            )
            mesh_paths[obj_id] = mesh_path_resolved
            mesh_dir_map[obj_id] = mesh_dir_resolved
        else:
            if not mesh_dir_map:
                mesh_dir_map = _assign_dirs(object_ids, mesh_dirs, "mesh_dir")
            mesh_dir_path = Path(mesh_dir_map[obj_id]).expanduser().resolve()
            mesh_paths[obj_id] = _resolve_mesh_path(mesh_dir_path, args.mesh_path, obj_id)

    # ref_dir_map is already resolved above when multiple reference dirs are provided

    pipelines = {}
    meshes = {}
    grasp_pose_preview = {}
    grasp_pose_preview_time_s = {}
    shared_detmodel, _, _ = load_detector_model(
        model_path=args.model_path,
        finetune_ckpt=args.finetune_ckpt,
        device=args.device,
        dtype=args.dtype,
    )

    for obj_id in object_ids:
        pipelines[obj_id] = ExemplarPosePipeline(
            model_path=args.model_path,
            finetune_ckpt=args.finetune_ckpt,
            reference_dir=ref_dir_map[obj_id],
            mesh_dir=mesh_dir_map[obj_id],
            mesh_path=args.mesh_path,
            object_id=obj_id,
            ref_view_ids=args.ref_view_ids,
            max_side_length=args.max_side_length,
            no_square=args.no_square,
            num_points_approx=args.num_points_approx,
            device=args.device,
            dtype=args.dtype,
            det_filter=args.det_filter,
            nms_iou=args.nms_iou,
            max_objects=args.max_objects,
            grayscale=args.grayscale,
            visualize_icp=False,
            to_base=False,
            detmodel=shared_detmodel,
        )
        mesh = o3d.io.read_triangle_mesh(str(mesh_paths[obj_id]))
        if mesh.is_empty():
            raise ValueError(f"Mesh invalid or empty: {mesh_paths[obj_id]}")
        meshes[obj_id] = mesh
        if args.predict_grasp_pose:
            t_grasp_start = time.perf_counter()
            grasp_pose_preview[obj_id] = naive_grasp_demo(
                str(mesh_paths[obj_id]),
                top_k=1,
                seed=0,
                use_convex_hull=GRASP_METHODS.get_spec(obj_id).use_convex_hull,
            )
            grasp_pose_preview_time_s[obj_id] = time.perf_counter() - t_grasp_start
            print(
                f"[{obj_id}] grasp pose preview time: "
                f"{grasp_pose_preview_time_s[obj_id] * 1000.0:.1f} ms"
            )

    sock = None
    conn = None
    if not args.dryrun:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((args.host, args.port))
        sock.listen(1)
        print(f"Listening on {args.host}:{args.port}")
        conn, addr = sock.accept()
        print("Robot connected from", addr)
    else:
        print("Dry run enabled: skipping robot connection.")

    realsense = Realsense(width=args.width, height=args.height, fps=args.fps)

    last_time = time.time()
    fps_ema: Optional[float] = None

    try:
        last_gripper = 0.0
        while True:
            frame_bgr, depth_image = realsense.get_frames(timeout_ms=args.rs_timeout_ms)

            vis_list = []
            detections = []
            per_obj_data = {}
            for obj_id in object_ids:
                poses, masks_nhw, scores_n, _, timings = pipelines[obj_id].process_frame(
                    frame_bgr,
                    depth_image,
                    realsense.K,
                    alpha=args.alpha,
                    max_show=args.max_show,
                    draw_overlays=True,
                    run_icp=False,
                    return_timings=True,
                )
                per_obj_data[obj_id] = (masks_nhw, scores_n)
                print(
                    f"[{obj_id}] encode image: {timings.get('encode_image_s', 0.0) * 1000.0:.1f} ms, "
                    f"detect: {timings.get('detect_s', 0.0) * 1000.0:.1f} ms"
                )
                if masks_nhw.shape[0] > 0:
                    scores_cpu = scores_n.detach().float().cpu().numpy()
                    detections.append((obj_id, masks_nhw, scores_cpu))

            per_obj_best = {}
            best_obj = None
            best_score = -float("inf")
            for obj_id, masks_nhw, scores_cpu in detections:
                if scores_cpu.size == 0:
                    continue
                idx = int(np.argmax(scores_cpu))
                score = float(scores_cpu[idx])
                mask = masks_nhw[idx].detach().float().cpu().numpy() > 0
                per_obj_best[obj_id] = (mask, score)
                if score > best_score:
                    best_score = score
                    best_obj = obj_id

            pose_cam = None
            per_obj_pose = {}
            per_obj_pointcloud_mm = {}
            per_obj_pose_robot = {}
            per_obj_icp_time_s = {}
            if per_obj_best:
                h, w = frame_bgr.shape[:2]
                for obj_id, (mask, _) in per_obj_best.items():
                    mask_resized = cv2.resize(
                        mask.astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    t_icp_start = time.perf_counter()
                    pose, pointcloud_mm = get_pose_and_pointcloud_from_mask(
                        mask_resized.astype(np.uint8),
                        depth_image,
                        realsense.K,
                        meshes[obj_id],
                        visualize=False,
                    )
                    per_obj_icp_time_s[obj_id] = time.perf_counter() - t_icp_start
                    per_obj_pose[obj_id] = pose
                    per_obj_pointcloud_mm[obj_id] = pointcloud_mm
                    if pose is not None:
                        per_obj_pose_robot[obj_id] = _transform_pose(T_RC, pose)
                # choose best object in guard
                best_obj_guarded = None
                best_score_guarded = -float("inf")
                for obj_id, (_, score) in per_obj_best.items():
                    if obj_id not in per_obj_pose_robot:
                        continue
                    t = per_obj_pose_robot[obj_id][:3, 3]
                    if _within_guard(float(t[0]), float(t[1]), float(t[2])):
                        if score > best_score_guarded:
                            best_score_guarded = score
                            best_obj_guarded = obj_id
                if best_obj_guarded is not None:
                    best_obj = best_obj_guarded
                    pose_cam = per_obj_pose.get(best_obj)
                else:
                    print("WARNING: no detections within guard.")
                # visualize ICP only for chosen best_obj
                if best_obj is not None and best_obj in per_obj_best:
                    mask, _ = per_obj_best[best_obj]
                    mask_resized = cv2.resize(
                        mask.astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    t_icp_start = time.perf_counter()
                    pose_cam = get_pose_from_mask(
                        mask_resized.astype(np.uint8),
                        depth_image,
                        realsense.K,
                        meshes[best_obj],
                        visualize=args.visualize_icp,
                    )
                    per_obj_icp_time_s[best_obj] = time.perf_counter() - t_icp_start

            for obj_id in object_ids:
                if obj_id in per_obj_data:
                    masks_nhw, scores_n = per_obj_data[obj_id]
                    num_show = min(args.max_show, masks_nhw.shape[0])
                    poses_show = [None for _ in range(num_show)]
                    if obj_id in per_obj_best:
                        best_mask, _ = per_obj_best[obj_id]
                        if masks_nhw.shape[0] > 0:
                            scores_cpu = scores_n.detach().float().cpu().numpy()
                            best_idx = int(np.argmax(scores_cpu))
                            if best_idx < num_show:
                                poses_show[best_idx] = per_obj_pose.get(obj_id)
                    vis = _compose_detection_icp_vis(
                        frame_bgr,
                        masks_nhw,
                        scores_n,
                        poses_show,
                        K=realsense.K,
                        mesh=meshes[obj_id],
                        pointclouds_mm=[per_obj_pointcloud_mm.get(obj_id) if p is not None else None for p in poses_show],
                        alpha=args.alpha,
                        max_show=args.max_show,
                        object_id=obj_id,
                    )
                    vis_list.append(vis)
                else:
                    vis_list.append(frame_bgr.copy())

            pose_robot = None
            grasp_cmd = None
            grasp_pose_time_s = None
            if pose_cam is not None and best_obj is not None:
                pose_robot = _transform_pose(T_RC, pose_cam)
                cam_t = pose_cam[:3, 3]
                rob_t = pose_robot[:3, 3]
                if best_obj in per_obj_icp_time_s:
                    print(f"[{best_obj}] icp: {per_obj_icp_time_s[best_obj] * 1000.0:.1f} ms")
                if args.predict_grasp_pose and best_obj in grasp_pose_preview and grasp_pose_preview[best_obj] is not None:
                    if args.replane_with_pose:
                        rpy_deg = _rpy_deg_from_pose(pose_robot)
                        if args.snap_rp_90:
                            rpy_deg = (_snap_90(rpy_deg[0]), _snap_90(rpy_deg[1]), rpy_deg[2])
                        t_grasp_start = time.perf_counter()
                        grasp_pose_mesh = naive_grasp_demo(
                            str(mesh_paths[best_obj]),
                            top_k=1,
                            seed=0,
                            rpy_deg=rpy_deg,
                            use_convex_hull=GRASP_METHODS.get_spec(best_obj).use_convex_hull,
                        )
                        grasp_pose_time_s = time.perf_counter() - t_grasp_start
                        if grasp_pose_mesh is not None:
                            print("[replane] grasp pose mesh matrix:")
                            print(grasp_pose_mesh)
                            gp_x, gp_y, gp_z = grasp_pose_mesh[:3, 3].tolist()
                            gp_yaw = _yaw_deg_from_pose(grasp_pose_mesh)
                            print(
                                "[replane] grasp pose mesh xyz/yaw: "
                                f"{gp_x:.4f} {gp_y:.4f} {gp_z:.4f} {gp_yaw:.1f}"
                            )
                            x = pose_robot[0, 3] + grasp_pose_mesh[0, 3]
                            y = pose_robot[1, 3] + grasp_pose_mesh[1, 3]
                            z = pose_robot[2, 3] + grasp_pose_mesh[2, 3] + args.grasp_z_comp
                            yaw = _yaw_deg_from_pose(grasp_pose_mesh)
                            grip = GRASP_METHODS.get_spec(best_obj).gripper_close_pos
                        else:
                            object_yaw_deg = _yaw_deg_from_pose(pose_robot)
                            x, y, z, grip, yaw = GRASP_METHODS.apply_to_pose(
                                best_obj,
                                pose_robot,
                                object_yaw_deg=object_yaw_deg,
                                current_yaw_deg=FIXED_TAIL[2],
                            )
                    else:
                        grasp_pose_mesh = grasp_pose_preview[best_obj]
                        grasp_pose_time_s = grasp_pose_preview_time_s.get(best_obj)
                        x, y, z, yaw = _compose_planned_grasp(pose_robot, grasp_pose_mesh, args.grasp_z_comp)
                        grip = GRASP_METHODS.get_spec(best_obj).gripper_close_pos
                else:
                    object_yaw_deg = _yaw_deg_from_pose(pose_robot)
                    x, y, z, grip, yaw = GRASP_METHODS.apply_to_pose(
                        best_obj,
                        pose_robot,
                        object_yaw_deg=object_yaw_deg,
                        current_yaw_deg=FIXED_TAIL[2],
                    )
                if grasp_pose_time_s is not None:
                    suffix = " (cached)" if not args.replane_with_pose else ""
                    print(f"[{best_obj}] grasp pose: {grasp_pose_time_s * 1000.0:.1f} ms{suffix}")
                if z < MIN_ROBOT_Z:
                    z = MIN_ROBOT_Z
                if not _within_guard(x, y, z):
                    print(
                        f"[{best_obj}] WARNING: commanded pose out of bounds "
                        f"x={x:.4f} y={y:.4f} z={z:.4f} (guard x={GUARD_X} y={GUARD_Y} z={GUARD_Z}); skipping command."
                    )
                    grasp_cmd = None
                else:
                    grasp_cmd = (x, y, z, FIXED_TAIL[0], FIXED_TAIL[1], yaw, grip)
                print(f"[{best_obj}] camera frame pose (m): {cam_t[0]:.4f} {cam_t[1]:.4f} {cam_t[2]:.4f}")
                rpy = _rpy_deg_from_pose(pose_robot)
                print(
                    f"[{best_obj}] robot  frame pose (m): "
                    f"{rob_t[0]:.4f} {rob_t[1]:.4f} {rob_t[2]:.4f} "
                    f"rpy (deg): {rpy[0]:.1f} {rpy[1]:.1f} {rpy[2]:.1f}"
                )
                print(
                    f"[{best_obj}] commanded grasp (m, deg, grip): "
                    f"{x:.4f} {y:.4f} {z:.4f} {yaw:.1f} {grip:.4f}"
                )
            else:
                print("No pose detected.")

            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps = 1.0 / dt
                fps_ema = fps if fps_ema is None else fps_ema * 0.9 + fps * 0.1

            if vis_list:
                vis = np.hstack(vis_list)
                cv2.imshow("Full Manipulation Pipeline", vis)

            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("o"), ord("O")):
                msg = (
                    f"goto,{DEFAULT_CMD[0]:.4f} {DEFAULT_CMD[1]:.4f} {DEFAULT_CMD[2]:.4f} "
                    f"{DEFAULT_CMD[3]:.1f} {DEFAULT_CMD[4]:.1f} {DEFAULT_CMD[5]:.1f} {last_gripper:.4f}\n"
                )
                if conn is not None:
                    conn.sendall(msg.encode())
                    print("Sent default pose.")
                else:
                    print("Dry run: default pose not sent.")
                continue
            if key in (ord("c"), ord("C")):
                grip = GRASP_METHODS.get_spec(best_obj if best_obj else object_ids[0]).gripper_close_pos
                last_gripper = grip
                msg = f"goto,a,a,a,a,a,a,{grip:.4f}\n"
                if conn is not None:
                    conn.sendall(msg.encode())
                    print("Sent close gripper command: " + msg)
                else:
                    print("Dry run: close gripper command not sent.")
                continue
            if key in (ord("r"), ord("R")):
                last_gripper = 0.0
                msg = "goto,a,a,a,a,a,a,0.0\n"
                if conn is not None:
                    conn.sendall(msg.encode())
                    print("Sent open gripper command: " + msg)
                else:
                    print("Dry run: open gripper command not sent.")
                continue
            if key in (10, 13) and grasp_cmd is not None:
                msg = (
                    f"approach,{grasp_cmd[0]:.4f} {grasp_cmd[1]:.4f} {grasp_cmd[2]:.4f} "
                    f"{grasp_cmd[3]:.1f} {grasp_cmd[4]:.1f} {grasp_cmd[5]:.1f} {0.0:.4f}\n"
                )
                if conn is not None:
                    conn.sendall(msg.encode())
                    print("Sent grasp command: " + msg)
                else:
                    print("Dry run: grasp command not sent.")
            elif key in (ord("e"), ord("E")) and grasp_cmd is not None and best_obj is not None:
                drop_pose = DROP_OFF_POSES.get(best_obj)
                if drop_pose is None:
                    print(f"No dropoff pose configured for {best_obj}.")
                    continue
                if not _within_guard(grasp_cmd[0], grasp_cmd[1], grasp_cmd[2]):
                    print(
                        f"[{best_obj}] WARNING: commanded pose out of bounds "
                        f"x={grasp_cmd[0]:.4f} y={grasp_cmd[1]:.4f} z={grasp_cmd[2]:.4f} "
                        f"(guard x={GUARD_X} y={GUARD_Y} z={GUARD_Z}); skipping sequence."
                    )
                    continue
                if not _within_guard(drop_pose[0], drop_pose[1], drop_pose[2]):
                    print(
                        f"[{best_obj}] WARNING: dropoff pose out of bounds "
                        f"x={drop_pose[0]:.4f} y={drop_pose[1]:.4f} z={drop_pose[2]:.4f} "
                        f"(guard x={GUARD_X} y={GUARD_Y} z={GUARD_Z}); skipping sequence."
                    )
                    continue
                grip = GRASP_METHODS.get_spec(best_obj).gripper_close_pos
                last_gripper = grip
                approach_msg = (
                    f"approach,{grasp_cmd[0]:.4f} {grasp_cmd[1]:.4f} {grasp_cmd[2]:.4f} "
                    f"{grasp_cmd[3]:.1f} {grasp_cmd[4]:.1f} {grasp_cmd[5]:.1f} {0.0:.4f}"
                )
                close_msg = f"goto,a,a,a,a,a,a,{grip:.4f}"
                default_msg = (
                    f"goto,{DEFAULT_CMD[0]:.4f} {DEFAULT_CMD[1]:.4f} {DEFAULT_CMD[2]:.4f} "
                    f"{DEFAULT_CMD[3]:.1f} {DEFAULT_CMD[4]:.1f} {DEFAULT_CMD[5]:.1f} {last_gripper:.4f}"
                )
                drop_msg = (
                    f"goto,{drop_pose[0]:.4f} {drop_pose[1]:.4f} {drop_pose[2]:.4f} "
                    f"{drop_pose[3]:.1f} {drop_pose[4]:.1f} {drop_pose[5]:.1f} {last_gripper:.4f}"
                )
                open_msg = "goto,a,a,a,a,a,a,0.0"
                default_after_open_msg = (
                    f"goto,{DEFAULT_CMD[0]:.4f} {DEFAULT_CMD[1]:.4f} {DEFAULT_CMD[2]:.4f} "
                    f"{DEFAULT_CMD[3]:.1f} {DEFAULT_CMD[4]:.1f} {DEFAULT_CMD[5]:.1f} {0.0:.4f}"
                )
                msg = ";".join(
                    [approach_msg, close_msg, default_msg, drop_msg, open_msg, default_after_open_msg]
                ) + "\n"
                if conn is not None:
                    conn.sendall(msg.encode())
                    print("Sent pickup/dropoff sequence: " + msg)
                else:
                    print("Dry run: pickup/dropoff sequence not sent.")
            elif key == 32:
                print("Skipped.")
    finally:
        realsense.stop()
        if conn is not None:
            conn.close()
        if sock is not None:
            sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
