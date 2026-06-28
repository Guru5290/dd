"""Pose estimation using an ArUco marker rigidly attached to the workpiece."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from cnc_perception.bed_calibration import _detect_aruco_markers, _get_aruco_dictionary, rvec_tvec_to_matrix
from cnc_perception.pose_solver import PoseEstimate, _reprojection_error


@dataclass(frozen=True)
class WorkpieceMarkerConfig:
    dictionary: str
    marker_id: int
    marker_size_m: float
    exclude_ids: tuple[int, ...]
    center_x_m: float
    center_y_m: float
    yaw_offset_deg: float
    fallback_to_markerless: bool


def marker_transform_in_object_frame(config: WorkpieceMarkerConfig) -> np.ndarray:
    """Transform from workpiece object frame to ArUco marker frame (on top face)."""
    yaw_rad = math.radians(config.yaw_offset_deg)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    transform = np.eye(4, dtype=np.float64)
    transform[0, 0] = cos_yaw
    transform[0, 1] = -sin_yaw
    transform[1, 0] = sin_yaw
    transform[1, 1] = cos_yaw
    transform[0, 3] = config.center_x_m
    transform[1, 3] = config.center_y_m
    transform[2, 3] = 0.0
    return transform


def _invert_transform(matrix: np.ndarray) -> np.ndarray:
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def detect_workpiece_marker_pose(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    marker_config: WorkpieceMarkerConfig,
) -> Optional[PoseEstimate]:
    """Estimate workpiece pose in the camera optical frame from a workpiece-mounted ArUco."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dictionary = _get_aruco_dictionary(marker_config.dictionary)
    corners, ids = _detect_aruco_markers(gray, dictionary)
    if ids is None or len(ids) == 0:
        return None

    half = marker_config.marker_size_m / 2.0
    marker_object_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )
    t_object_marker = marker_transform_in_object_frame(marker_config)
    t_marker_object = _invert_transform(t_object_marker)

    best_pose: Optional[PoseEstimate] = None
    best_error = float('inf')

    for index, detected_id in enumerate(ids.flatten()):
        marker_id = int(detected_id)
        if marker_id in marker_config.exclude_ids:
            continue
        if marker_id != marker_config.marker_id:
            continue

        image_points = corners[index].reshape(4, 2)
        try:
            success, rvec, tvec = cv2.solvePnP(
                marker_object_points,
                image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_IPPE,
            )
        except cv2.error:
            continue
        if not success:
            continue

        t_optical_marker = rvec_tvec_to_matrix(rvec, tvec)
        t_optical_object = t_optical_marker @ t_marker_object
        rotation_matrix = t_optical_object[:3, :3]
        translation = t_optical_object[:3, 3]
        rvec, _ = cv2.Rodrigues(rotation_matrix)

        error = _reprojection_error(
            marker_object_points,
            image_points,
            rvec,
            translation.reshape(3, 1),
            camera_matrix,
            distortion,
        )
        if error < best_error:
            best_error = error
            best_pose = PoseEstimate(
                translation=translation,
                rotation_matrix=rotation_matrix,
                reprojection_error=error,
            )

    return best_pose


def draw_workpiece_marker_debug(image_bgr: np.ndarray, marker_config: WorkpieceMarkerConfig) -> np.ndarray:
    """Overlay detected workpiece marker corners for debug view."""
    debug = image_bgr.copy()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dictionary = _get_aruco_dictionary(marker_config.dictionary)
    corners, ids = _detect_aruco_markers(gray, dictionary)
    if ids is not None and len(ids) > 0 and hasattr(cv2.aruco, 'drawDetectedMarkers'):
        cv2.aruco.drawDetectedMarkers(debug, corners, ids)
    return debug
