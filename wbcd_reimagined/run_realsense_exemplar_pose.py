#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from muggled_sam.make_sam import make_sam_from_state_dict

from eval_image_exemplar import (
    _mask_bbox,
    apply_grayscale,
    apply_mask_nms,
    build_exemplar_tokens_for_object,
    generate_detections_train,
    pad_exemplar_batch,
    parse_ref_view_ids,
)


PALETTE_BGR: List[Tuple[int, int, int]] = [
    (0, 220, 0),
    (0, 180, 255),
    (255, 180, 0),
    (180, 0, 255),
    (0, 255, 255),
    (255, 0, 180),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate 6D pose from RealSense using exemplar detection masks.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/kevin/ICL/rendering_prompted_muggled_sam/model_weights/sam3.pt",
        help="Path to SAMv3 checkpoint (.pt).",
    )
    parser.add_argument(
        "--finetune_ckpt",
        type=str,
        default="model_weights/0321_k12_b156_resume_from_preprinte18_s1_e34.pth",
        help="Optional finetuned detector checkpoint.",
    )
    # parser.add_argument(
    #     "--reference_dir",
    #     default="/home/kevin/ICL/rendering_prompted_muggled_sam/assets/renders_2442_0316",
    #     type=str,
    #     help="Directory containing reference images and masks.",
    # )
    parser.add_argument(
        "--reference_dir",
        default="/home/kevin/ICL/rendering_prompted_muggled_sam/assets/wbcd_renders_2442_0316",
        type=str,
        help="Directory containing reference images and masks.",
    )
    
    # parser.add_argument(
    #     "--mesh_dir",
    #     default="/home/kevin/ICL/rendering_prompted_muggled_sam/assets/mesh_0316",
    #     type=str,
    #     help="Directory containing meshes (.stl/.ply/.obj).",
    # )
    parser.add_argument(
        "--mesh_dir",
        default="/home/kevin/ICL/rendering_prompted_muggled_sam/assets/wbcd_meshes",
        type=str,
        help="Directory containing meshes (.stl/.ply/.obj).",
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        default="",
        help="Optional explicit mesh path. Overrides mesh_dir/object_id lookup.",
    )
    parser.add_argument(
        "--object_id",
        type=str,
        default="bowl",
        help="Object name/id to load from the reference directory (and mesh_dir).",
    )
    parser.add_argument(
        "--ref_view_ids",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9,10,11",
        help="Reference view ids to use.",
    )
    parser.add_argument(
        "--max_side_length",
        type=int,
        default=1008,
        help="Image encoder max side length.",
    )
    parser.add_argument(
        "--no_square",
        action="store_true",
        help="Disable square image sizing.",
    )
    parser.add_argument(
        "--num_points_approx",
        type=int,
        default=24,
        help="Approx number of mask points to sample per reference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device string, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "bf16"],
        default="",
        help="Override model dtype.",
    )
    parser.add_argument(
        "--det_filter",
        type=float,
        default=0.0,
        help="Detection score threshold filter.",
    )
    parser.add_argument(
        "--nms_iou",
        type=float,
        default=0.5,
        help="IoU threshold for mask NMS (<=0 disables NMS).",
    )
    parser.add_argument(
        "--max_show",
        type=int,
        default=3,
        help="Max number of masks to visualize.",
    )
    parser.add_argument(
        "--max_objects",
        type=int,
        default=2,
        help="Max number of object poses to compute per frame.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay alpha for masks.",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Convert all input images to grayscale.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="RealSense capture width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="RealSense capture height.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="RealSense capture FPS.",
    )
    parser.add_argument(
        "--visualize_icp",
        default=False,
        action="store_true",
        help="Show Open3D ICP visualization (blocking).",
    )
    parser.add_argument(
        "--to_base",
        action="store_true",
        help="Transform poses to base frame using the hard-coded extrinsic.",
    )
    parser.add_argument(
        "--rs_timeout_ms",
        type=int,
        default=5000,
        help="Realsense wait_for_frames timeout in milliseconds.",
    )
    return parser.parse_args()


def cam_frame_to_base_frame(object_pose_in_camera: np.ndarray) -> np.ndarray:
    pos_only = False
    if object_pose_in_camera.shape == (3,):
        pos_only = True
        translation = object_pose_in_camera
        object_pose_in_camera = np.eye(4)
        object_pose_in_camera[:3, 3] = translation
    T = np.array(
        [
            [0, -0.743, 0.669, 0.047],
            [-1, 0, 0, 0.055],
            [0, -0.669, -0.743, 0.46],
            [0, 0, 0, 1],
        ]
    )
    object_pos_in_base_hom = T @ object_pose_in_camera
    if pos_only:
        object_pos_in_base_hom[:3, :3] = np.eye(3)
    return object_pos_in_base_hom


class Realsense:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(self.config)
        self.device = self.profile.get_device()
        self.depth_sensor = self.device.first_depth_sensor()
        self.depth_scale = float(self.depth_sensor.get_depth_scale())
        self.depth_stream = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        self.color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_intrinsics = self.depth_stream.get_intrinsics()
        color_intrinsics = self.color_stream.get_intrinsics()
        self.align = rs.align(rs.stream.color)
        self.K = np.array(
            [
                [color_intrinsics.fx, 0, color_intrinsics.ppx],
                [0, color_intrinsics.fy, color_intrinsics.ppy],
                [0, 0, 1],
            ]
        )

    def restart(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.profile = self.pipeline.start(self.config)

    def get_frames(self, timeout_ms: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        frames = self.pipeline.wait_for_frames(timeout_ms)
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not color_frame:
            raise RuntimeError("Realsense returned invalid frames (depth/color missing).")
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_image = (depth_image.astype(np.float32) * self.depth_scale / 0.001).astype(int)
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image

    def stop(self) -> None:
        self.pipeline.stop()


def _rot_x(theta: float) -> np.ndarray:
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1],
        ]
    )


def _rot_y(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta), 0],
            [0, 1, 0, 0],
            [-np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1],
        ]
    )


def _rot_z(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def icp_get_6d_pose(
    cad_pc: o3d.geometry.PointCloud,
    cam_pc: o3d.geometry.PointCloud,
    threshold: float = 50.0,
    init_transformations: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, float]:
    rot = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -0.66913, 0.74314],
            [0.0, -0.74314, -0.66913],
        ]
    )
    transformation = np.eye(4)
    transformation[:3, :3] = rot
    if init_transformations is None:
        init_transformations = [
            transformation,
            transformation @ _rot_y(np.pi / 2),
            transformation @ _rot_z(np.pi / 2),
            transformation @ _rot_z(-np.pi / 2),
            transformation @ _rot_x(np.pi / 2),
            transformation @ _rot_x(-np.pi / 2),
        ]

    # Use bounding-box center for alignment instead of centroid.
    cam_pts = np.asarray(cam_pc.points)
    cam_min = np.min(cam_pts, axis=0)
    cam_max = np.max(cam_pts, axis=0)
    pc_center = ((cam_min + cam_max) / 2.0).reshape((3, 1))
    min_rmse = float("inf")
    best_trans = None

    for trans_init in init_transformations:
        ref_pc = copy.deepcopy(cad_pc)
        ref_pts = np.asarray(ref_pc.points)
        ref_min = np.min(ref_pts, axis=0)
        ref_max = np.max(ref_pts, axis=0)
        ref_pc_center = ((ref_min + ref_max) / 2.0).reshape((3, 1))
        t_init = np.identity(4)
        t_init[:3, 3] = -(ref_pc_center - pc_center)[:, 0]
        t_init = t_init @ trans_init
        reg_p2p = o3d.pipelines.registration.registration_icp(
            ref_pc,
            cam_pc,
            threshold,
            t_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20),
        )
        if reg_p2p.inlier_rmse < min_rmse:
            min_rmse = reg_p2p.inlier_rmse
            best_trans = reg_p2p.transformation

    if best_trans is None:
        best_trans = np.eye(4)
    return best_trans, min_rmse


def get_pointcloud_from_mask(mask: np.ndarray, depth_image: np.ndarray, K: np.ndarray, trim: float = 0.05) -> Optional[np.ndarray]:
    ys, xs = np.where(mask > 0)
    zs = depth_image[ys, xs]
    valid = zs > 0
    xs = xs[valid]
    ys = ys[valid]
    zs = zs[valid]
    if len(xs) == 0:
        return None
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    xs_3d = (xs - cx) * zs / fx
    ys_3d = (ys - cy) * zs / fy
    pointcloud = np.stack((xs_3d, ys_3d, zs), axis=-1)
    median = np.median(pointcloud, axis=0)
    distances = np.linalg.norm(pointcloud - median, axis=1)
    threshold = np.percentile(distances, (1 - trim) * 100)
    pointcloud = pointcloud[distances <= threshold]
    return pointcloud


def get_pose_from_mask(
    mask: np.ndarray,
    depth_image: np.ndarray,
    K: np.ndarray,
    mesh: o3d.geometry.TriangleMesh,
    visualize: bool,
) -> Optional[np.ndarray]:
    pointcloud = get_pointcloud_from_mask(mask, depth_image, K)
    if pointcloud is None or pointcloud.size == 0:
        return None
    ref_pc = mesh.sample_points_uniformly(number_of_points=2000)
    ref_arr = np.asarray(ref_pc.points)
    if np.max(ref_arr[:, 0]) - np.min(ref_arr[:, 0]) <= 1:
        ref_pc.points = o3d.utility.Vector3dVector(ref_arr * 1000)
    cam_pc = o3d.geometry.PointCloud()
    cam_pc.points = o3d.utility.Vector3dVector(pointcloud)
    pose, rmse = icp_get_6d_pose(ref_pc, cam_pc, threshold=50.0)
    pose_m = pose.copy()
    pose_m[:3, 3] = pose[:3, 3] / 1000.0
    if visualize:
        _draw_registration_result(ref_pc, cam_pc, pose)
    return pose_m


def _draw_registration_result(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
) -> None:
    source_temp = o3d.geometry.PointCloud(source)
    target_temp = o3d.geometry.PointCloud(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0, origin=[0, 0, 0])
    object_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0, origin=[0, 0, 0])
    object_frame.transform(transformation)
    o3d.visualization.draw_geometries(
        [source_temp, target_temp, coordinate_frame, object_frame],
        zoom=0.4459,
        front=[0.9288, -0.2951, -0.2242],
        lookat=[1.6784, 2.0612, 1.4451],
        up=[-0.3402, -0.9189, -0.1996],
    )


def _draw_overlays(
    frame_bgr: np.ndarray,
    masks_nhw: torch.Tensor,
    scores_n: torch.Tensor,
    poses: List[Optional[np.ndarray]],
    alpha: float,
    max_show: int,
) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    if masks_nhw.shape[0] == 0:
        return out
    num_show = min(max_show, masks_nhw.shape[0])
    masks_cpu = masks_nhw[:num_show].detach().float().cpu().numpy()
    scores_cpu = scores_n[:num_show].detach().float().cpu().numpy()

    for i in range(num_show):
        mask = masks_cpu[i] > 0
        if mask.max() <= 0:
            continue
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        color = PALETTE_BGR[i % len(PALETTE_BGR)]
        overlay = out.copy()
        overlay[mask_resized] = color
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
        bbox = _mask_bbox(mask_resized.astype(np.uint8))
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
            score_val = float(scores_cpu[i]) if i < scores_cpu.size else 0.0
            label = f"{score_val:.2f}"
            if i < len(poses) and poses[i] is not None:
                t = poses[i][:3, 3]
                label = f"{score_val:.2f} ({t[0]:.3f},{t[1]:.3f},{t[2]:.3f})m"
            cv2.putText(
                out,
                label,
                (x0, max(0, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
    return out


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


class ExemplarPosePipeline:
    def __init__(
        self,
        model_path: str,
        finetune_ckpt: str,
        reference_dir: str,
        mesh_dir: str,
        mesh_path: str,
        object_id: str,
        ref_view_ids: str,
        max_side_length: int = 1008,
        no_square: bool = False,
        num_points_approx: int = 24,
        device: str = "cuda:0",
        dtype: str = "",
        det_filter: float = 0.0,
        nms_iou: float = 0.5,
        max_objects: int = 2,
        grayscale: bool = False,
        visualize_icp: bool = False,
        to_base: bool = False,
    ) -> None:
        self.device = torch.device(device if device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        if dtype:
            self.dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
        else:
            self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.max_side_length = max_side_length
        self.no_square = no_square
        self.num_points_approx = num_points_approx
        self.det_filter = det_filter
        self.nms_iou = nms_iou
        self.max_objects = max_objects
        self.grayscale = grayscale
        self.visualize_icp = visualize_icp
        self.to_base = to_base

        reference_dir_path = Path(reference_dir).expanduser().resolve()
        if not reference_dir_path.is_dir():
            raise FileNotFoundError(reference_dir_path)

        mesh_dir_path = Path(mesh_dir).expanduser().resolve()
        mesh_path_resolved = _resolve_mesh_path(mesh_dir_path, mesh_path, object_id)
        self.mesh = o3d.io.read_triangle_mesh(str(mesh_path_resolved))
        if self.mesh.is_empty() or not self.mesh.has_triangles():
            raise ValueError(f"Mesh invalid or empty: {mesh_path_resolved}")

        view_ids = parse_ref_view_ids(ref_view_ids)
        if not view_ids:
            raise ValueError("No reference view ids resolved.")

        _, base_model = make_sam_from_state_dict(model_path)
        base_model.to(device=self.device, dtype=self.dtype)
        self.detmodel = base_model.make_detector_model()
        self.detmodel.to(device=self.device, dtype=self.dtype)
        self.detmodel.eval()

        if finetune_ckpt:
            ckpt = torch.load(finetune_ckpt, map_location="cpu")
            self.detmodel.image_exemplar_fusion.load_state_dict(ckpt["image_exemplar_fusion"])
            self.detmodel.exemplar_detector.load_state_dict(ckpt["exemplar_detector"])
            self.detmodel.exemplar_segmentation.load_state_dict(ckpt["exemplar_segmentation"])
            print("Loaded finetuned detector weights from", finetune_ckpt)

        self.exemplar_ref = build_exemplar_tokens_for_object(
            detmodel=self.detmodel,
            object_id=object_id,
            reference_dir=reference_dir_path,
            ref_view_ids=view_ids,
            max_side_length=max_side_length,
            use_square_sizing=not no_square,
            num_points_approx=num_points_approx,
            device=self.device,
            grayscale=grayscale,
        )
        if self.exemplar_ref is None:
            raise RuntimeError("No exemplar tokens built. Check reference_dir/object_id/ref_view_ids.")

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        depth_image: np.ndarray,
        K: np.ndarray,
        alpha: float = 0.45,
        max_show: int = 3,
        draw_overlays: bool = False,
    ) -> Tuple[List[Optional[np.ndarray]], torch.Tensor, torch.Tensor, Optional[np.ndarray]]:
        if self.grayscale:
            frame_bgr = apply_grayscale(frame_bgr)

        with torch.inference_mode():
            img_t = self.detmodel.image_encoder.prepare_image(
                frame_bgr,
                max_side_length=self.max_side_length,
                use_square_sizing=not self.no_square,
            )
            encoded_img = self.detmodel.image_encoder(img_t)
            encoded_image_features_list = self.detmodel.image_projection.v3_projection(encoded_img)

            exemplar_batch, padding_mask = pad_exemplar_batch([self.exemplar_ref], device=self.device)
            mask_preds, box_preds, det_scores, _ = generate_detections_train(
                self.detmodel,
                encoded_image_features_list,
                exemplar_batch,
                detection_filter_threshold=self.det_filter,
                exemplar_padding_mask_bn=padding_mask,
            )

            masks_nhw = mask_preds[0]
            scores_n = det_scores[0]
            if masks_nhw.shape[0] > 0 and self.nms_iou > 0:
                _, masks_nhw, scores_n = apply_mask_nms(
                    box_preds[0], masks_nhw, scores_n, iou_threshold=self.nms_iou
                )

        poses: List[Optional[np.ndarray]] = []
        if masks_nhw.shape[0] > 0:
            h, w = frame_bgr.shape[:2]
            for i in range(min(self.max_objects, masks_nhw.shape[0])):
                mask = masks_nhw[i].detach().float().cpu().numpy() > 0
                mask_resized = cv2.resize(
                    mask.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                pose = get_pose_from_mask(
                    mask_resized.astype(np.uint8),
                    depth_image,
                    K,
                    self.mesh,
                    visualize=self.visualize_icp,
                )
                if pose is not None and self.to_base:
                    pose = cam_frame_to_base_frame(pose)
                poses.append(pose)

        vis = None
        if draw_overlays:
            vis = _draw_overlays(
                frame_bgr,
                masks_nhw,
                scores_n,
                poses,
                alpha=alpha,
                max_show=max_show,
            )
        return poses, masks_nhw, scores_n, vis


def main() -> None:
    args = parse_args()

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
        to_base=args.to_base,
    )

    realsense = Realsense(width=args.width, height=args.height, fps=args.fps)

    last_time = time.time()
    fps_ema: Optional[float] = None

    try:
        while True:
            try:
                frame_bgr, depth_image = realsense.get_frames(timeout_ms=args.rs_timeout_ms)
                print("Got frames from RealSense.")
            except RuntimeError as exc:
                msg = str(exc)
                if "Frame didn't arrive" in msg or "invalid frames" in msg:
                    print(f"[realsense] {msg} (timeout_ms={args.rs_timeout_ms}); restarting pipeline...")
                    realsense.restart()
                    continue
                raise

            poses, masks_nhw, scores_n, vis = pipeline.process_frame(
                frame_bgr,
                depth_image,
                realsense.K,
                alpha=args.alpha,
                max_show=args.max_show,
                draw_overlays=True,
            )

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
                cv2.imshow("Exemplar RealSense 6D Pose", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        realsense.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
