"""Bed plane calibration: ArUco origin and manual corner homography."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml

from cnc_perception.bed_config import BedConfig, BedDimensions
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


def detect_aruco_origin_pose(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    marker_dictionary: str,
    marker_id: int,
    marker_size_m: float,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return (rvec, tvec) of marker frame relative to camera."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dictionary = _get_aruco_dictionary(marker_dictionary)
    corners, ids = _detect_aruco_markers(gray, dictionary)
    if ids is None or len(ids) == 0:
        return None

    for index, detected_id in enumerate(ids.flatten()):
        if int(detected_id) != marker_id:
            continue
        marker_corners = corners[index]
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
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            marker_corners.reshape(4, 2),
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if success:
            return rvec, tvec
    return None


def _detect_aruco_markers(gray: np.ndarray, dictionary: cv2.aruco_Dictionary):
    if hasattr(cv2.aruco, 'ArucoDetector'):
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
        return corners, ids
    parameters = cv2.aruco.DetectorParameters_create()
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)


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
) -> dict[str, Any]:
    """
    Compute T_camera_bed from ArUco marker at bed origin.

    Marker frame convention (OpenCV ArUco): Z out of marker plane.
  Bed frame: origin at marker center, X along bed length, Y along bed width, Z up.
    """
    camera_matrix, distortion = camera_info_to_matrices(
        camera_info_k, camera_info_d, image_width, image_height
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
    t_camera_marker = rvec_tvec_to_matrix(rvec, tvec)

    # Align marker frame to bed frame: rotate 180 deg about X so marker Z maps to bed -Z
    # then rotate 180 about Z if needed. For flat marker on bed: marker Z points toward camera.
    # Bed Z points up (opposite bed surface normal into bed). R_x(pi) flips Z.
    flip_z = np.eye(4, dtype=np.float64)
    flip_z[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    t_camera_bed = t_camera_marker @ flip_z

    translation, quat = matrix_to_translation_quaternion(t_camera_bed)
    return {
        'calibrated': True,
        'method': 'aruco_origin',
        'camera_frame': 'camera_link',
        'bed_frame': 'cnc_bed_frame',
        'translation_m': translation.tolist(),
        'rotation_xyzw': quat.tolist(),
        'marker_id': bed_config.marker.marker_id,
        'marker_size_m': bed_config.marker.marker_size_m,
    }


def bed_corner_coordinates(bed: BedDimensions) -> np.ndarray:
    """Bed corners in cnc_bed_frame: BL, BR, TR, TL."""
    length = bed.length_m
    width = bed.width_m
    return np.array(
        [
            [0.0, 0.0],
            [length, 0.0],
            [length, width],
            [0.0, width],
        ],
        dtype=np.float64,
    )


def calibrate_bed_from_corners(
    image_corners_px: np.ndarray,
    bed: BedDimensions,
) -> dict[str, Any]:
    """
    Compute homography from bed plane (meters) to image pixels.

    image_corners_px: 4 points [BL, BR, TR, TL] in image pixels.
    """
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


def bed_xy_to_image_px(homography: np.ndarray, x_m: float, y_m: float) -> tuple[float, float]:
    point = np.array([x_m, y_m, 1.0], dtype=np.float64)
    projected = homography @ point
    projected /= projected[2]
    return float(projected[0]), float(projected[1])


def image_px_to_bed_xy(homography: np.ndarray, u: float, v: float) -> tuple[float, float]:
    inv_h = np.linalg.inv(homography)
    point = np.array([u, v, 1.0], dtype=np.float64)
    projected = inv_h @ point
    projected /= projected[2]
    return float(projected[0]), float(projected[1])


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
