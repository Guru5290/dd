"""Transform utilities for camera, bed, and workpiece frames."""

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
from geometry_msgs.msg import Pose, Quaternion, Transform


def _quat_to_matrix(quat: Iterable[float]) -> np.ndarray:
    x, y, z, w = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_to_matrix(translation: Iterable[float], rotation_xyzw: Iterable[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_to_matrix(rotation_xyzw)
    matrix[:3, 3] = np.array(translation, dtype=np.float64)
    return matrix


def matrix_to_translation_quaternion(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotation[2, 1] - rotation[1, 2]) * s
        y = (rotation[0, 2] - rotation[2, 0]) * s
        z = (rotation[1, 0] - rotation[0, 1]) * s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float64)
    quat /= max(np.linalg.norm(quat), 1e-12)
    return translation, quat


def invert_transform(matrix: np.ndarray) -> np.ndarray:
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def pose_to_matrix(pose: Pose) -> np.ndarray:
    return transform_to_matrix(
        [pose.position.x, pose.position.y, pose.position.z],
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
    )


def matrix_to_pose(matrix: np.ndarray) -> Pose:
    translation, quat = matrix_to_translation_quaternion(matrix)
    pose = Pose()
    pose.position.x = float(translation[0])
    pose.position.y = float(translation[1])
    pose.position.z = float(translation[2])
    pose.orientation = Quaternion(
        x=float(quat[0]),
        y=float(quat[1]),
        z=float(quat[2]),
        w=float(quat[3]),
    )
    return pose


def yaw_from_matrix(matrix: np.ndarray) -> float:
    return float(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))


def transform_to_msg(translation: np.ndarray, quat: np.ndarray) -> Transform:
    msg = Transform()
    msg.translation.x = float(translation[0])
    msg.translation.y = float(translation[1])
    msg.translation.z = float(translation[2])
    msg.rotation = Quaternion(
        x=float(quat[0]),
        y=float(quat[1]),
        z=float(quat[2]),
        w=float(quat[3]),
    )
    return msg


def is_pose_center_on_bed(
    t_bed_workpiece: np.ndarray,
    bed_length_m: float,
    bed_width_m: float,
    margin_m: float = 0.015,
) -> bool:
    """Return True when the workpiece center lies inside the CNC bed footprint."""
    x_m = float(t_bed_workpiece[0, 3])
    y_m = float(t_bed_workpiece[1, 3])
    return (
        margin_m <= x_m <= bed_length_m - margin_m
        and margin_m <= y_m <= bed_width_m - margin_m
    )


def surface_normal_tilt_deg(rotation: np.ndarray) -> float:
    """Angle between the object +Z axis and the bed +Z axis (degrees)."""
    normal = rotation[:, 2]
    cos_angle = float(np.clip(normal[2], -1.0, 1.0))
    return float(math.degrees(math.acos(cos_angle)))


def flatten_transform_to_bed_plane(t_bed_workpiece: np.ndarray, thickness_m: float) -> np.ndarray:
    """Keep X, Y, yaw; force Z = top-face height and tilt = 0 for flat stock."""
    yaw_rad = math.radians(yaw_from_matrix(t_bed_workpiece[:3, :3]))
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    flat = np.eye(4, dtype=np.float64)
    flat[0, 0] = cos_yaw
    flat[0, 1] = -sin_yaw
    flat[1, 0] = sin_yaw
    flat[1, 1] = cos_yaw
    flat[0, 3] = float(t_bed_workpiece[0, 3])
    flat[1, 3] = float(t_bed_workpiece[1, 3])
    flat[2, 3] = float(thickness_m)
    return flat


def _angle_lerp_rad(previous: float, current: float, alpha: float) -> float:
    delta = math.atan2(math.sin(current - previous), math.cos(current - previous))
    return previous + (1.0 - alpha) * delta


def smooth_flat_bed_transform(
    previous: Optional[np.ndarray],
    current: np.ndarray,
    alpha: float,
    yaw_alpha: Optional[float] = None,
) -> np.ndarray:
    """Low-pass filter X/Y/Z and yaw for a bed-flat workpiece transform."""
    if previous is None or alpha <= 0.0:
        return current.copy()
    if alpha >= 1.0:
        return previous.copy()

    yaw_filter = alpha if yaw_alpha is None else yaw_alpha
    smoothed = current.copy()
    for axis in range(3):
        smoothed[axis, 3] = alpha * previous[axis, 3] + (1.0 - alpha) * current[axis, 3]

    yaw_prev = math.radians(yaw_from_matrix(previous[:3, :3]))
    yaw_cur = math.radians(yaw_from_matrix(current[:3, :3]))
    yaw_smooth = _angle_lerp_rad(yaw_prev, yaw_cur, yaw_filter)
    cos_yaw = math.cos(yaw_smooth)
    sin_yaw = math.sin(yaw_smooth)
    smoothed[:3, :3] = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return smoothed


def apply_coordinate_reporting_offset(
    t_bed_workpiece: np.ndarray,
    subtract_x_m: float,
    subtract_y_m: float,
) -> np.ndarray:
    """Shift reported pose from calibration origin to ruler origin (e.g. metal corner)."""
    if subtract_x_m == 0.0 and subtract_y_m == 0.0:
        return t_bed_workpiece.copy()
    reported = t_bed_workpiece.copy()
    reported[0, 3] -= subtract_x_m
    reported[1, 3] -= subtract_y_m
    return reported


def angle_diff_deg(measured: float, reference: float) -> float:
    return float((measured - reference + 180.0) % 360.0 - 180.0)


def square_yaw_match_error_deg(measured_yaw: float, reference_yaw: float) -> float:
    """Smallest yaw error allowing 90 deg symmetry for square stock."""
    return min(
        abs(angle_diff_deg(measured_yaw, reference_yaw + 90.0 * k))
        for k in range(4)
    )
