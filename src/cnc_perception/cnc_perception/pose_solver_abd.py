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

    distortion = np.array(distortion_data, dtype=np.float64).reshape(-1, 1)
    return camera_matrix, distortion


def _reprojection_error(object_points, rvec, tvec, camera_matrix, distortion, image_points) -> float:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
    return float(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1).mean())


def _select_planar_solution(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    reproj_errors: Optional[np.ndarray],
    prefer_normal_toward_camera: bool,
) -> int:
    """Pick the best of the (up to two) planar PnP solutions.

    Primary key: reprojection error. Tie-break (when the two errors are close):
    prefer the solution whose object +Z axis points back toward the camera,
    i.e. the part is seen from above lying flat, not mirrored through the bed.
    """
    n = len(rvecs)
    if n == 1:
        return 0

    if reproj_errors is not None and len(reproj_errors) == n:
        errors = np.asarray(reproj_errors, dtype=np.float64).reshape(-1)
    else:
        errors = np.array([
            _reprojection_error(object_points, rvecs[i], tvecs[i], camera_matrix, distortion, image_points)
            for i in range(n)
        ])

    order = np.argsort(errors)
    best, second = int(order[0]), int(order[1])

    if not prefer_normal_toward_camera:
        return best

    # If the two best solutions reproject almost equally well, the choice is
    # genuinely ambiguous -> use the physical prior (top face toward camera).
    if errors[second] <= errors[best] * 1.25 + 1e-9:
        def normal_z(idx: int) -> float:
            rotation, _ = cv2.Rodrigues(rvecs[idx])
            return float(rotation[2, 2])  # object +Z expressed in camera frame
        # In the OpenCV optical frame the camera looks down +Z, so a top face
        # pointing back at the camera has a negative z-component.
        cand = sorted([best, second], key=normal_z)
        return cand[0]

    return best


def solve_workpiece_pose(
    image_corners: np.ndarray,
    object_corners: list[list[float]],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    use_ippe_for_planar: bool = True,
    max_reprojection_error_px: Optional[float] = 5.0,
    prefer_normal_toward_camera: bool = True,
) -> Optional[PoseEstimate]:
    if image_corners is None or len(image_corners) != 4:
        return None

    object_points = np.array(object_corners, dtype=np.float64)
    image_points = np.asarray(image_corners, dtype=np.float64).reshape(-1, 2)

    flags = cv2.SOLVEPNP_IPPE if use_ippe_for_planar else cv2.SOLVEPNP_ITERATIVE
    try:
        retval, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=flags,
        )
    except cv2.error:
        return None

    if retval < 1 or not rvecs:
        return None

    best = _select_planar_solution(
        object_points, image_points, camera_matrix, distortion,
        list(rvecs), list(tvecs),
        np.asarray(reproj_errors) if reproj_errors is not None else None,
        prefer_normal_toward_camera,
    )
    rvec, tvec = rvecs[best], tvecs[best]

    error = _reprojection_error(object_points, rvec, tvec, camera_matrix, distortion, image_points)
    if max_reprojection_error_px is not None and error > max_reprojection_error_px:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rvec)

    return PoseEstimate(
        translation=tvec.reshape(3),
        rotation_matrix=rotation_matrix,
        reprojection_error=error,
    )


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
