"""Bed plane calibration: ArUco origin and manual corner homography."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml

from cnc_perception.bed_config import BedConfig, BedDimensions
from cnc_perception.camera_frames import transform_optical_to_link
from cnc_perception.pose_solver import camera_info_to_matrices
from cnc_perception.transform_utils import matrix_to_translation_quaternion, transform_to_matrix


ARUCO_DICTIONARIES = {
    'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
    'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
    'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
    'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
}


def _get_aruco_dictionary(name: str) -> cv2.aruco_Dictionary:
    if name not in ARUCO_DICTIONARIES:
        raise ValueError(f'Unsupported ArUco dictionary: {name}')
    return cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[name])


def _detect_aruco_markers(gray: np.ndarray, dictionary: cv2.aruco_Dictionary):
    if hasattr(cv2.aruco, 'ArucoDetector'):
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
        return corners, ids
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    return corners, ids


def marker_to_bed_transform(marker_yaw_deg: float, flip_marker_y: bool) -> np.ndarray:
    """
    Map OpenCV ArUco marker frame to cnc_bed_frame at the bed origin.

    ArUco on a horizontal bed: +Z points toward the camera (same as bed +Z).
    ArUco +Y points down on the marker print; bed +Y is the opposite in-plane direction.
    """
    matrix = np.eye(4, dtype=np.float64)
    if flip_marker_y:
        y_flip = np.diag([1.0, -1.0, 1.0])
    else:
        y_flip = np.eye(3, dtype=np.float64)
    yaw = math.radians(marker_yaw_deg)
    yaw_rot = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    matrix[:3, :3] = yaw_rot @ y_flip
    return matrix


def detect_aruco_origin_pose(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    marker_dictionary: str,
    marker_id: int,
    marker_size_m: float,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return (rvec, tvec) of marker frame in camera optical coordinates."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dictionary = _get_aruco_dictionary(marker_dictionary)
    corners, ids = _detect_aruco_markers(gray, dictionary)
    if ids is None or len(ids) == 0:
        return None

    half = marker_size_m / 2.0
    object_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )

    for index, detected_id in enumerate(ids.flatten()):
        if int(detected_id) != marker_id:
            continue
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            corners[index].reshape(4, 2),
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if success:
            return rvec, tvec
    return None


def rvec_tvec_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = tvec.reshape(3)
    return matrix


def calibrate_bed_from_aruco(
    image_bgr: np.ndarray,
    camera_info_k: list[float],
    camera_info_d: list[float],
    image_width: int,
    image_height: int,
    bed_config: BedConfig,
    *,
    rectified_image: bool = True,
) -> dict[str, Any]:
    """
    Compute static TF camera_link -> cnc_bed_frame from ArUco at bed origin.

    solvePnP is performed in camera optical coordinates, then converted to camera_link.
    """
    camera_matrix, distortion = camera_info_to_matrices(
        camera_info_k,
        camera_info_d,
        image_width,
        image_height,
        rectified=rectified_image,
    )
    result = detect_aruco_origin_pose(
        image_bgr,
        camera_matrix,
        distortion,
        bed_config.marker.dictionary,
        bed_config.marker.marker_id,
        bed_config.marker.marker_size_m,
    )
    if result is None:
        raise RuntimeError(
            f'ArUco marker id {bed_config.marker.marker_id} not detected. '
            'Ensure marker is visible, in focus, and matches dictionary/size in cnc_bed.yaml.'
        )

    rvec, tvec = result
    t_optical_marker = rvec_tvec_to_matrix(rvec, tvec)
    t_marker_bed = marker_to_bed_transform(
        bed_config.marker.marker_to_bed_yaw_deg,
        bed_config.marker.flip_marker_y,
    )
    t_optical_bed = t_optical_marker @ t_marker_bed
    t_link_bed = transform_optical_to_link() @ t_optical_bed

    translation, quat = matrix_to_translation_quaternion(t_link_bed)
    return {
        'calibrated': True,
        'method': 'aruco_origin',
        'camera_frame': 'camera_link',
        'bed_frame': 'cnc_bed_frame',
        'translation_m': translation.tolist(),
        'rotation_xyzw': quat.tolist(),
        'marker_id': bed_config.marker.marker_id,
        'marker_size_m': bed_config.marker.marker_size_m,
        'marker_to_bed_yaw_deg': bed_config.marker.marker_to_bed_yaw_deg,
        'rectified_image': rectified_image,
    }


def bed_corner_coordinates(bed: BedDimensions) -> np.ndarray:
    """Bed corners in cnc_bed_frame: BL, BR, TR, TL."""
    return np.array(
        [
            [0.0, 0.0],
            [bed.length_m, 0.0],
            [bed.length_m, bed.width_m],
            [0.0, bed.width_m],
        ],
        dtype=np.float64,
    )


def calibrate_bed_from_corners(
    image_corners_px: np.ndarray,
    bed: BedDimensions,
) -> dict[str, Any]:
    if image_corners_px.shape != (4, 2):
        raise ValueError('image_corners_px must be shape (4, 2)')

    bed_corners = bed_corner_coordinates(bed)
    homography, _ = cv2.findHomography(bed_corners, image_corners_px.astype(np.float64))
    if homography is None:
        raise RuntimeError('Homography estimation failed')

    return {
        'calibrated': True,
        'method': 'manual_corners',
        'camera_frame': 'camera_link',
        'bed_frame': 'cnc_bed_frame',
        'homography_bed_to_image': homography.tolist(),
        'bed_corners_image_px': image_corners_px.tolist(),
        'translation_m': [0.0, 0.0, 0.0],
        'rotation_xyzw': [0.0, 0.0, 0.0, 1.0],
    }


def save_bed_calibration(path: str, data: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        yaml.safe_dump(data, handle, default_flow_style=False)


def load_bed_calibration(path: str) -> dict[str, Any]:
    cal_path = Path(path)
    if not cal_path.is_file():
        raise FileNotFoundError(f'Bed calibration not found: {path}')
    with cal_path.open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}
    if not data.get('calibrated', False):
        raise RuntimeError('Bed is not calibrated yet. Run step02 script first.')
    return data


def calibration_to_transform_matrix(calibration: dict[str, Any]) -> np.ndarray:
    return transform_to_matrix(
        calibration.get('translation_m', [0.0, 0.0, 0.0]),
        calibration.get('rotation_xyzw', [0.0, 0.0, 0.0, 1.0]),
    )
