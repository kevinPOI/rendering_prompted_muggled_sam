#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import math
import socket
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_realsense_exemplar_pose import ExemplarPosePipeline, Realsense, _resolve_mesh_path
from wbcd_reimagined.grasp_methods import GraspMethods, GraspSpec
from wbcd_reimagined.naive_grasp_planner import (
    GripperParams,
    PlannerParams,
    _transparent_material,
    _transparent_material_color,
    apply_temporary_display_transform,
    demo as naive_grasp_demo,
    plan_grasps,
)


T_RC = np.array(
    [
        [0.0, 0.0, 1.0, 0.16],
        [-1.0, 0.0, 0.0, 0.06],
        [0.0, -1.0, 0.0, 0.03],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

DEFAULT_CMD = (0.3, 0.15, 0.2, 180.0, 0.0, 90.0, 0.0)
FIXED_TAIL = (180.0, 0.0, 90.0, 0.0)
MIN_ROBOT_Z = -0.09


GRASP_METHODS = GraspMethods()
GRASP_METHODS.register("bowl", GraspSpec(0.0, -0.05, 0.0, gripper_close_pos=0.98, use_object_yaw=False, use_convex_hull=False))
GRASP_METHODS.register("cube", GraspSpec(0.0, 0.0, 0.0, gripper_close_pos=0.6, use_object_yaw=True, use_convex_hull=False))
GRASP_METHODS.register("gear", GraspSpec(0.0, 0.0, 0.0, gripper_close_pos=0.6, use_object_yaw=True, use_convex_hull=False))
GRASP_METHODS.register("rod_mount", GraspSpec(0.0, 0.0, 0.0, gripper_close_pos=0.8, use_object_yaw=True, use_convex_hull=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manipulation pipeline with RealSense exemplar pose.")
    parser.add_argument("--model_path", type=str, default="/home/kevin/ICL/rendering_prompted_muggled_sam/model_weights/sam3.pt")
    parser.add_argument("--finetune_ckpt", type=str, default="model_weights/0321_k12_b156_resume_from_preprinte18_s1_e34.pth")
    # parser.add_argument("--reference_dir", type=str, default="/home/kevin/ICL/rendering_prompted_muggled_sam/assets/wbcd_renders_2442_0316")
    # parser.add_argument("--mesh_dir", type=str, default="/home/kevin/ICL/rendering_prompted_muggled_sam/assets/wbcd_meshes")
    parser.add_argument("--reference_dir", type=str, default="/home/kevin/ICL/rendering_prompted_muggled_sam/assets/renders_2442_0316")
    parser.add_argument(
        "--mesh_dir",
        type=str,
        default="/home/kevin/ICL/rendering_prompted_muggled_sam/assets/mesh_0316",
    )
    parser.add_argument("--mesh_path", type=str, default="")
    parser.add_argument("--object_id", type=str, default="gear")
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
    parser.add_argument("--visualize_icp", default = True, action="store_true")
    
    parser.add_argument("--rs_timeout_ms", type=int, default=50000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--dryrun", default = False, action="store_true", help="Run without robot connection.")
    parser.add_argument(
        "--predict_grasp_pose",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Predict grasp pose from mesh using naive_grasp_planner.",
    )
    parser.add_argument(
        "--replane_with_pose",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If true, rotate mesh by detected pose before planning, then translate planned grasp.",
    )
    parser.add_argument(
        "--grasp_z_comp",
        type=float,
        default=-0.065,
        help="Additive Z compensation (meters) applied to planned grasp pose.",
    )
    return parser.parse_args()


def _best_pose(poses, scores_n: torch.Tensor) -> Optional[np.ndarray]:
    if scores_n.numel() == 0:
        return None
    idx = int(torch.argmax(scores_n).item())
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


def _compute_z_center(mesh: o3d.geometry.TriangleMesh) -> float:
    aabb = mesh.get_axis_aligned_bounding_box()
    return float(0.5 * (aabb.min_bound[2] + aabb.max_bound[2]))


def _visualize_planning_plane(mesh: o3d.geometry.TriangleMesh, z_center: float) -> None:
    aabb = mesh.get_axis_aligned_bounding_box()
    bbox_size = aabb.get_extent()
    center = aabb.get_center()
    plane_w = max(1e-6, float(bbox_size[0]) * 1.2)
    plane_h = max(1e-6, float(bbox_size[1]) * 1.2)
    plane_t = max(0.002, 0.02 * min(plane_w, plane_h))

    plane = o3d.geometry.TriangleMesh.create_box(width=plane_w, height=plane_h, depth=plane_t)
    plane.translate([-plane_w / 2.0, -plane_h / 2.0, -plane_t / 2.0])
    plane.translate([center[0], center[1], z_center])
    plane.compute_vertex_normals()

    axis_len = 0.6 * float(np.max(bbox_size))
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_len, origin=[0.0, 0.0, 0.0])
    axis.compute_vertex_normals()

    mesh_vis = copy.deepcopy(mesh)
    mesh_vis.compute_vertex_normals()
    mesh_vis.paint_uniform_color([0.7, 0.7, 0.7])

    geometries = [
        {"name": "mesh", "geometry": mesh_vis, "material": _transparent_material(0.25)},
        {
            "name": "plane",
            "geometry": plane,
            "material": _transparent_material_color(np.array([0.2, 0.6, 0.9]), 0.25),
        },
        {"name": "axis", "geometry": axis},
    ]
    draw_geometries = [
        geom["geometry"] if isinstance(geom, dict) else geom
        for geom in geometries
    ]
    draw_geometries = apply_temporary_display_transform(draw_geometries)
    o3d.visualization.draw_geometries(draw_geometries, window_name="Naive Grasp Planner")


def _plan_grasp_pose_naive(
    mesh_base: o3d.geometry.TriangleMesh,
    pose_robot: np.ndarray,
    replane_with_pose: bool,
    use_convex_hull: bool = False,
    top_k: int = 1,
) -> Optional[np.ndarray]:
    if mesh_base.is_empty():
        return None
    mesh = copy.deepcopy(mesh_base)
    if replane_with_pose:
        mesh.rotate(pose_robot[:3, :3], center=[0.0, 0.0, 0.0])
    z_center = _compute_z_center(mesh)
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
    T_grasp = grasps[0].pose.copy()
    if replane_with_pose:
        T_grasp[:3, 3] += pose_robot[:3, 3]
        return T_grasp
    T_out = pose_robot @ T_grasp
    # Keep translation as a direct offset in robot frame (no rotation),
    # matching the naive planner's expectation for top-down offset.
    T_out[:3, 3] = pose_robot[:3, 3] + T_grasp[:3, 3]
    return T_out


def _plan_grasp_pose_naive_mesh(
    mesh_base: o3d.geometry.TriangleMesh,
    use_convex_hull: bool = False,
    top_k: int = 1,
) -> Optional[np.ndarray]:
    if mesh_base.is_empty():
        return None
    mesh = copy.deepcopy(mesh_base)
    z_center = _compute_z_center(mesh)
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


def main() -> None:
    args = parse_args()

    mesh_base = None
    grasp_pose_preview = None
    if args.predict_grasp_pose:
        try:
            mesh_path_resolved = _resolve_mesh_path(
                Path(args.mesh_dir).expanduser().resolve(), args.mesh_path, args.object_id
            )
            mesh_base = o3d.io.read_triangle_mesh(str(mesh_path_resolved))
            if mesh_base.is_empty():
                print("Failed to load mesh for grasp planning; falling back to template grasp.")
                mesh_base = None
            else:
                mesh_base.remove_duplicated_vertices()
                mesh_base.remove_duplicated_triangles()
                mesh_base.remove_degenerate_triangles()
                mesh_base.remove_non_manifold_edges()
        except Exception as exc:
            print(f"Mesh load for grasp planning failed: {exc}")
            mesh_base = None
        if mesh_base is not None and not args.replane_with_pose:
            grasp_pose_preview = naive_grasp_demo(
                str(mesh_path_resolved),
                top_k=1,
                seed=0,
                use_convex_hull=GRASP_METHODS.get_spec(args.object_id).use_convex_hull,
            )
            if grasp_pose_preview is not None:
                print("Planner grasp pose (mesh frame):")
                print(grasp_pose_preview)
            else:
                print("Planner found no grasps for visualization preview.")

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

    pipeline = ExemplarPosePipeline(
        model_path=args.model_path,
        finetune_ckpt=args.finetune_ckpt,
        reference_dir=args.reference_dir,
        mesh_dir=args.mesh_dir,
        mesh_path=args.mesh_path,
        object_id=args.object_id,
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
        visualize_icp=args.visualize_icp,
        to_base=False,
    )

    realsense = Realsense(width=args.width, height=args.height, fps=args.fps)

    last_time = time.time()
    fps_ema: Optional[float] = None

    try:
        last_gripper = 0.0
        while True:
            frame_bgr, depth_image = realsense.get_frames(timeout_ms=args.rs_timeout_ms)

            poses, masks_nhw, scores_n, vis = pipeline.process_frame(
                frame_bgr,
                depth_image,
                realsense.K,
                alpha=args.alpha,
                max_show=args.max_show,
                draw_overlays=True,
            )

            pose_cam = _best_pose(poses, scores_n)
            pose_robot = None
            grasp_cmd = None
            if pose_cam is not None:
                pose_robot = _transform_pose(T_RC, pose_cam)
                cam_t = pose_cam[:3, 3]
                rob_t = pose_robot[:3, 3]
                if args.predict_grasp_pose and mesh_base is not None:
                    if args.replane_with_pose:
                        grasp_pose_robot = _plan_grasp_pose_naive(
                            mesh_base,
                            pose_robot,
                            replane_with_pose=True,
                            use_convex_hull=GRASP_METHODS.get_spec(args.object_id).use_convex_hull,
                            top_k=1,
                        )
                        if grasp_pose_robot is not None:
                            x, y, z = grasp_pose_robot[:3, 3].tolist()
                            z += args.grasp_z_comp
                            yaw = _yaw_deg_from_pose(grasp_pose_robot)
                            grip = GRASP_METHODS.get_spec(args.object_id).gripper_close_pos
                        else:
                            object_yaw_deg = _yaw_deg_from_pose(pose_robot)
                            x, y, z, grip, yaw = GRASP_METHODS.apply_to_pose(
                                args.object_id,
                                pose_robot,
                                object_yaw_deg=object_yaw_deg,
                                current_yaw_deg=FIXED_TAIL[2],
                            )
                    else:
                        grasp_pose_mesh = grasp_pose_preview
                        if grasp_pose_mesh is not None:
                            object_yaw_deg = _yaw_deg_from_pose(pose_robot)
                            grasp_yaw_deg = _yaw_deg_from_pose(grasp_pose_mesh)
                            gx, gy, gz = grasp_pose_mesh[:3, 3].tolist()
                            yaw_rad = math.radians(object_yaw_deg)
                            gx_r = math.cos(yaw_rad) * gx - math.sin(yaw_rad) * gy
                            gy_r = math.sin(yaw_rad) * gx + math.cos(yaw_rad) * gy
                            x = pose_robot[0, 3] + gx_r
                            y = pose_robot[1, 3] + gy_r
                            z = pose_robot[2, 3] + gz + args.grasp_z_comp
                            yaw = object_yaw_deg + grasp_yaw_deg
                            grip = GRASP_METHODS.get_spec(args.object_id).gripper_close_pos
                        else:
                            print("WARNING: Grasp planner found no grasps; falling back to template grasp.")
                            breakpoint()
                            object_yaw_deg = _yaw_deg_from_pose(pose_robot)
                            x, y, z, grip, yaw = GRASP_METHODS.apply_to_pose(
                                args.object_id,
                                pose_robot,
                                object_yaw_deg=object_yaw_deg,
                                current_yaw_deg=FIXED_TAIL[2],
                            )
                else:
                    object_yaw_deg = _yaw_deg_from_pose(pose_robot)
                    x, y, z, grip, yaw = GRASP_METHODS.apply_to_pose(
                        args.object_id,
                        pose_robot,
                        object_yaw_deg=object_yaw_deg,
                        current_yaw_deg=FIXED_TAIL[2],
                    )
                if z < MIN_ROBOT_Z:
                    z = MIN_ROBOT_Z
                grasp_cmd = (x, y, z, FIXED_TAIL[0], FIXED_TAIL[1], yaw, grip)
                print(f"camera frame pose (m): {cam_t[0]:.4f} {cam_t[1]:.4f} {cam_t[2]:.4f}")
                rpy = _rpy_deg_from_pose(pose_robot)
                print(
                    "robot  frame pose (m): "
                    f"{rob_t[0]:.4f} {rob_t[1]:.4f} {rob_t[2]:.4f} "
                    f"rpy (deg): {rpy[0]:.1f} {rpy[1]:.1f} {rpy[2]:.1f}"
                )
                print(
                    "commanded grasp (m, deg, grip): "
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
            if fps_ema is not None and vis is not None:
                cv2.putText(
                    vis,
                    f"FPS: {fps_ema:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            if vis is not None:
                cv2.imshow("Manipulation Pipeline", vis)
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
                grip = GRASP_METHODS.get_spec(args.object_id).gripper_close_pos
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
