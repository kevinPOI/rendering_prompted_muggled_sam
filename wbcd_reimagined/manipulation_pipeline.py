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


T_RC = np.array(
    [
        [0.0, 0.0, 1.0, 0.15],
        [-1.0, 0.0, 0.0, 0.06],
        [0.0, -1.0, 0.0, 0.03],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

DEFAULT_CMD = (0.3, 0.15, 0.2, 180.0, 0.0, 90.0, 0.0)
FIXED_TAIL = (180.0, 0.0, 90.0, 0.0)
MIN_ROBOT_Z = -0.07
GRASPNET_ROOT_DEFAULT = "/home/kevin/ICL/graspnet-baseline"
GRASPNET_CKPT_DEFAULT = "/home/kevin/ICL/graspnet-baseline/checkpoint.tar"


GRASP_METHODS = GraspMethods()
GRASP_METHODS.register("bowl", GraspSpec(0.0, -0.05, 0.0, gripper_close_pos=0.98, use_object_yaw=False))
GRASP_METHODS.register("cube", GraspSpec(0.0, 0.0, 0.0, gripper_close_pos=0.6, use_object_yaw=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manipulation pipeline with RealSense exemplar pose.")
    parser.add_argument("--model_path", type=str, default="/home/kevin/ICL/rendering_prompted_muggled_sam/model_weights/sam3.pt")
    parser.add_argument("--finetune_ckpt", type=str, default="model_weights/0321_k12_b156_resume_from_preprinte18_s1_e34.pth")
    parser.add_argument("--reference_dir", type=str, default="/home/kevin/ICL/rendering_prompted_muggled_sam/assets/wbcd_renders_2442_0316")
    parser.add_argument("--mesh_dir", type=str, default="/home/kevin/ICL/rendering_prompted_muggled_sam/assets/wbcd_meshes")
    parser.add_argument("--mesh_path", type=str, default="")
    parser.add_argument("--object_id", type=str, default="cube")
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
    parser.add_argument("--visualize_icp", action="store_true")
    parser.add_argument("--rs_timeout_ms", type=int, default=50000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--dryrun", default = False, action="store_true", help="Run without robot connection.")
    parser.add_argument(
        "--predict_grasp_pose",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Predict grasp pose from mesh using GraspNet.",
    )
    parser.add_argument("--graspnet_root", type=str, default=GRASPNET_ROOT_DEFAULT)
    parser.add_argument("--graspnet_ckpt", type=str, default=GRASPNET_CKPT_DEFAULT)
    parser.add_argument("--graspnet_num_point", type=int, default=20000)
    parser.add_argument("--graspnet_num_view", type=int, default=300)
    parser.add_argument(
        "--graspnet_sample_method",
        type=str,
        choices=["poisson", "uniform"],
        default="poisson",
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


def _ensure_graspnet_paths(graspnet_root: Path) -> None:
    root = graspnet_root.resolve()
    for rel in ("models", "utils", "graspnetAPI"):
        path = str(root / rel)
        if path not in sys.path:
            sys.path.insert(0, path)


def _get_graspnet_net(graspnet_root: Path, checkpoint_path: str, num_view: int):
    _ensure_graspnet_paths(graspnet_root)
    from graspnet import GraspNet  # type: ignore

    net = GraspNet(
        input_feature_dim=0,
        num_view=num_view,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    return net


def _mesh_to_endpoints(
    mesh: o3d.geometry.TriangleMesh, num_point: int, sample_method: str
) -> tuple[dict, o3d.geometry.PointCloud]:
    mesh.compute_vertex_normals()
    if sample_method == "poisson":
        pcd = mesh.sample_points_poisson_disk(num_point)
    else:
        pcd = mesh.sample_points_uniformly(num_point)
    pts = np.asarray(pcd.points, dtype=np.float32)
    end_points = {}
    cloud_sampled = torch.from_numpy(pts[None, ...])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    end_points["point_clouds"] = cloud_sampled.to(device)
    end_points["cloud_colors"] = np.zeros_like(pts)
    return end_points, pcd


def _force_topdown_views(end_points, graspnet_root: Path, up_axis: str = "y"):
    _ensure_graspnet_paths(graspnet_root)
    from loss_utils import batch_viewpoint_params_to_matrix  # type: ignore

    seed_xyz = end_points["fp2_xyz"]
    device = seed_xyz.device
    dtype = seed_xyz.dtype
    if up_axis == "x":
        up = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    elif up_axis == "y":
        up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    else:
        up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    B, N, _ = seed_xyz.shape
    towards = up.view(1, 1, 3).repeat(B, N, 1)
    end_points["grasp_top_view_xyz"] = towards
    towards_flat = towards.view(-1, 3)
    angles = torch.zeros(towards_flat.shape[0], device=device, dtype=dtype)
    rots = batch_viewpoint_params_to_matrix(towards_flat, angles).view(B, N, 3, 3)
    end_points["grasp_top_view_rot"] = rots
    return end_points


def _get_grasps(net, end_points, graspnet_root: Path, topdown_only: bool = True, up_axis: str = "y"):
    _ensure_graspnet_paths(graspnet_root)
    from graspnet import pred_decode  # type: ignore
    try:
        from graspnetAPI import GraspGroup  # type: ignore
    except ImportError:
        from graspnetAPI.graspnetAPI import GraspGroup  # type: ignore

    with torch.no_grad():
        if topdown_only:
            end_points = net.view_estimator(end_points)
            end_points = _force_topdown_views(end_points, graspnet_root=graspnet_root, up_axis=up_axis)
            end_points = net.grasp_generator(end_points)
        else:
            end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    return GraspGroup(gg_array)


def _prepare_mesh_for_graspnet(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh = copy.deepcopy(mesh)
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    if max(extent) > 2.0:
        mesh.scale(0.001, center=mesh.get_center())
        bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    mesh.translate(-center)
    return mesh


def _predict_grasp_pose_from_mesh(
    mesh: o3d.geometry.TriangleMesh,
    net,
    num_point: int,
    sample_method: str,
    graspnet_root: Path,
) -> Optional[np.ndarray]:
    end_points, cloud = _mesh_to_endpoints(mesh, num_point=num_point, sample_method=sample_method)
    gg = _get_grasps(net, end_points, graspnet_root=graspnet_root, topdown_only=True, up_axis="y")
    if len(gg) == 0:
        return None
    gg.nms()
    gg.sort_by_score()
    vis_grasps(gg, cloud)
    grasp = gg[0]
    T_og = np.eye(4, dtype=np.float64)
    T_og[:3, :3] = grasp.rotation_matrix
    T_og[:3, 3] = grasp.translation
    return T_og


def _visualize_grasp_pose(mesh: o3d.geometry.TriangleMesh, grasp_pose: np.ndarray) -> None:
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    frame.transform(grasp_pose)
    o3d.visualization.draw_geometries([mesh, frame])


def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[0:4]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])


def main() -> None:
    args = parse_args()

    grasp_pose_obj = None
    if args.predict_grasp_pose:
        try:
            graspnet_root = Path(args.graspnet_root).expanduser().resolve()
            net = _get_graspnet_net(graspnet_root, args.graspnet_ckpt, args.graspnet_num_view)
            mesh_path_resolved = _resolve_mesh_path(
                Path(args.mesh_dir).expanduser().resolve(), args.mesh_path, args.object_id
            )
            mesh = o3d.io.read_triangle_mesh(str(mesh_path_resolved))
            mesh_gn = _prepare_mesh_for_graspnet(mesh)
            grasp_pose_obj = _predict_grasp_pose_from_mesh(
                mesh_gn,
                net,
                num_point=args.graspnet_num_point,
                sample_method=args.graspnet_sample_method,
                graspnet_root=graspnet_root,
            )
            if grasp_pose_obj is not None:
                _visualize_grasp_pose(mesh_gn, grasp_pose_obj)
                print(
                    "Loaded GraspNet grasp (mesh frame, m): "
                    f"{grasp_pose_obj[0,3]:.4f} {grasp_pose_obj[1,3]:.4f} {grasp_pose_obj[2,3]:.4f}"
                )
            else:
                print("GraspNet produced no grasps; falling back to template grasp.")
        except Exception as exc:
            print(f"GraspNet grasp prediction failed: {exc}")
            grasp_pose_obj = None

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
                if args.predict_grasp_pose and grasp_pose_obj is not None:
                    grasp_offset = grasp_pose_obj[:3, 3]
                    x, y, z = (pose_robot[:3, 3] + grasp_offset).tolist()
                    grasp_yaw = _yaw_deg_from_pose(grasp_pose_obj)
                    object_yaw_deg = _yaw_deg_from_pose(pose_robot)
                    use_obj_yaw = GRASP_METHODS.get_spec(args.object_id).use_object_yaw
                    base_yaw = object_yaw_deg if use_obj_yaw else FIXED_TAIL[2]
                    yaw = base_yaw + grasp_yaw
                    grip = GRASP_METHODS.get_spec(args.object_id).gripper_close_pos
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
                print(f"robot  frame pose (m): {rob_t[0]:.4f} {rob_t[1]:.4f} {rob_t[2]:.4f}")
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
                    f"{DEFAULT_CMD[0]:.4f} {DEFAULT_CMD[1]:.4f} {DEFAULT_CMD[2]:.4f} "
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
                msg = f"a,a,a,a,a,a,{grip:.4f}\n"
                if conn is not None:
                    conn.sendall(msg.encode())
                    print("Sent close gripper command: " + msg)
                else:
                    print("Dry run: close gripper command not sent.")
                continue
            if key in (ord("r"), ord("R")):
                last_gripper = 0.0
                msg = "a,a,a,a,a,a,0.0\n"
                if conn is not None:
                    conn.sendall(msg.encode())
                    print("Sent open gripper command: " + msg)
                else:
                    print("Dry run: open gripper command not sent.")
                continue
            if key in (10, 13) and grasp_cmd is not None:
                msg = (
                    f"{grasp_cmd[0]:.4f} {grasp_cmd[1]:.4f} {grasp_cmd[2]:.4f} "
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
