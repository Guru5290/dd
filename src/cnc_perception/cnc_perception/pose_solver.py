"""Pose solving utilities using OpenCV solvePnP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from geometry_msgs.msg import Pose, Quaternion


@dataclass
class PoseEstimate:
    translation: np.ndarray
    rotation_matrix: np.ndarray
    reprojection_error: float


def camera_info_to_matrices(
    camera_matrix_data: list[float],
    distortion_data: list[float],
    image_width: int,
    image_height: int,
    *,
    rectified: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if len(camera_matrix_data) != 9:
        raise ValueError('camera_matrix must contain 9 elements')

    camera_matrix = np.array(camera_matrix_data, dtype=np.float64).reshape(3, 3)
    if camera_matrix[0, 0] <= 0.0 or camera_matrix[1, 1] <= 0.0:
        raise ValueError('Invalid focal lengths in camera_matrix')

    if image_width > 0 and image_height > 0:
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        if cx <= 0.0 or cy <= 0.0:
            camera_matrix[0, 2] = image_width / 2.0
            camera_matrix[1, 2] = image_height / 2.0

    if rectified:
        distortion = np.zeros((5, 1), dtype=np.float64)
    else:
        distortion = np.array(distortion_data, dtype=np.float64).reshape(-1, 1)
    return camera_matrix, distortion


def _reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
    return float(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1).mean())


def _pose_from_rvec_tvec(
    rvec: np.ndarray,
    tvec: np.ndarray,
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> PoseEstimate:
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    error = _reprojection_error(object_points, image_points, rvec, tvec, camera_matrix, distortion)
    return PoseEstimate(
        translation=tvec.reshape(3),
        rotation_matrix=rotation_matrix,
        reprojection_error=error,
    )


def _planar_pose_facing_score(rotation_matrix: np.ndarray, translation: np.ndarray) -> float:
    """Prefer poses with the top face normal pointing toward the camera."""
    if translation[2] <= 0.01:
        return -100.0
    normal = rotation_matrix[:, 2]
    view_toward_camera = -translation / max(np.linalg.norm(translation), 1e-9)
    return float(np.dot(normal, view_toward_camera))


def _pose_rank(pose: PoseEstimate) -> float:
    """Lower is better: tight reprojection + top face toward camera (rejects shadow blobs)."""
    facing = _planar_pose_facing_score(pose.rotation_matrix, pose.translation)
    facing_penalty = max(0.0, 0.35 - facing) * 120.0
    depth_penalty = 80.0 if pose.translation[2] <= 0.01 else 0.0
    return pose.reprojection_error + facing_penalty + depth_penalty


def _pick_best_planar_pose(candidates: list[PoseEstimate]) -> Optional[PoseEstimate]:
    if not candidates:
        return None
    viable = [pose for pose in candidates if _planar_pose_facing_score(pose.rotation_matrix, pose.translation) > 0.0]
    pool = viable if viable else candidates
    pool.sort(key=_pose_rank)
    return pool[0]


def solve_workpiece_pose(
    image_corners: np.ndarray,
    object_corners: list[list[float]],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    use_ippe_for_planar: bool = True,
) -> Optional[PoseEstimate]:
    if image_corners is None or len(image_corners) != 4:
        return None

    object_points = np.array(object_corners, dtype=np.float64)
    image_points = np.asarray(image_corners, dtype=np.float64)
    candidates: list[PoseEstimate] = []

    if use_ippe_for_planar and hasattr(cv2, 'solvePnPGeneric'):
        try:
            success, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                object_points,
                image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_IPPE,
            )
            if success:
                for rvec, tvec in zip(rvecs, tvecs):
                    candidates.append(
                        _pose_from_rvec_tvec(
                            rvec, tvec, object_points, image_points, camera_matrix, distortion
                        )
                    )
        except cv2.error:
            candidates = []

    if not candidates:
        flags = cv2.SOLVEPNP_IPPE if use_ippe_for_planar else cv2.SOLVEPNP_ITERATIVE
        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                distortion,
                flags=flags,
            )
        except cv2.error:
            return None
        if not success:
            return None
        candidates.append(
            _pose_from_rvec_tvec(rvec, tvec, object_points, image_points, camera_matrix, distortion)
        )

    return _pick_best_planar_pose(candidates)


def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> Quaternion:
    matrix = rotation_matrix
    trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]

    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (matrix[2, 1] - matrix[1, 2]) * s
        y = (matrix[0, 2] - matrix[2, 0]) * s
        z = (matrix[1, 0] - matrix[0, 1]) * s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s

    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm > 1e-12:
        w /= norm
        x /= norm
        y /= norm
        z /= norm

    msg = Quaternion()
    msg.x = float(x)
    msg.y = float(y)
    msg.z = float(z)
    msg.w = float(w)
    return msg


def pose_estimate_to_geometry_pose(estimate: PoseEstimate) -> Pose:
    pose = Pose()
    pose.position.x = float(estimate.translation[0])
    pose.position.y = float(estimate.translation[1])
    pose.position.z = float(estimate.translation[2])
    pose.orientation = rotation_matrix_to_quaternion(estimate.rotation_matrix)
    return pose


def _quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def smooth_pose(
    previous: Optional[PoseEstimate],
    current: PoseEstimate,
    alpha: float,
) -> PoseEstimate:
    if previous is None or alpha <= 0.0:
        return current
    if alpha >= 1.0:
        return previous

    translation = alpha * previous.translation + (1.0 - alpha) * current.translation

    previous_quat = rotation_matrix_to_quaternion(previous.rotation_matrix)
    current_quat = rotation_matrix_to_quaternion(current.rotation_matrix)
    previous_vec = np.array([previous_quat.x, previous_quat.y, previous_quat.z, previous_quat.w])
    current_vec = np.array([current_quat.x, current_quat.y, current_quat.z, current_quat.w])
    dot = float(np.dot(previous_vec, current_vec))
    if dot < 0.0:
        current_vec = -current_vec
    blended = alpha * previous_vec + (1.0 - alpha) * current_vec
    norm = np.linalg.norm(blended)
    if norm > 1e-9:
        blended /= norm

    return PoseEstimate(
        translation=translation,
        rotation_matrix=_quaternion_to_rotation_matrix(blended),
        reprojection_error=current.reprojection_error,
    )



