from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GraspSpec:
    x_offset: float
    y_offset: float
    z_offset: float
    gripper_close_pos: float
    use_object_yaw: bool


class GraspMethods:
    """
    Template grasp method registry.

    For each object id, define grasp offsets (in robot/base frame),
    gripper close position, and whether to use the object's yaw.
    """

    def __init__(self) -> None:
        self._specs: dict[str, GraspSpec] = {}

    def get_spec(self, object_id: str) -> GraspSpec:
        return self._specs.get(
            object_id,
            GraspSpec(0.0, 0.0, 0.0, gripper_close_pos=0.02, use_object_yaw=False),
        )

    def register(self, object_id: str, spec: GraspSpec) -> None:
        self._specs[object_id] = spec

    def apply_to_pose(
        self,
        object_id: str,
        pose_robot: np.ndarray,
        object_yaw_deg: float,
        current_yaw_deg: float,
    ) -> tuple[float, float, float, float, float]:
        """
        Returns (x, y, z, gripper_close_pos, yaw_deg).
        If use_object_yaw is False, keep current_yaw_deg.
        """
        spec = self.get_spec(object_id)
        t = pose_robot[:3, 3].copy()
        t[0] += spec.x_offset
        t[1] += spec.y_offset
        t[2] += spec.z_offset
        if spec.use_object_yaw:
            yaw = object_yaw_deg
        else:
            yaw = 90.0
        return float(t[0]), float(t[1]), float(t[2]), spec.gripper_close_pos, float(yaw)
