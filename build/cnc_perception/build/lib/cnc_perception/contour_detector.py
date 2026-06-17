"""Contour-based workpiece detection for plain rectangular stock."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from cnc_perception.workpiece_config import DetectionConfig, WorkpieceDimensions


@dataclass(frozen=True)
class DetectionResult:
    corners: np.ndarray
    contour: np.ndarray
    score: float


def _order_corners_clockwise(points: np.ndarray) -> np.ndarray:
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = points.sum(axis=1)
    ordered[0] = points[np.argmin(s)]
    ordered[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    ordered[1] = points[np.argmin(diff)]
    ordered[3] = points[np.argmax(diff)]
    return ordered


def _contour_aspect_ratio(rect: tuple) -> float:
    _, (width, height), _ = rect
    short_side = min(width, height)
    long_side = max(width, height)
    if short_side <= 1e-6:
        return 0.0
    return long_side / short_side


def _aspect_ratio_matches(candidate_ratio: float, expected_ratio: float, tolerance: float) -> bool:
    if candidate_ratio <= 0.0:
        return False
    direct_error = abs(candidate_ratio - expected_ratio) / expected_ratio
    inverse_error = abs((1.0 / candidate_ratio) - expected_ratio) / expected_ratio
    return min(direct_error, inverse_error) <= tolerance


def _preprocess(gray: np.ndarray, config: DetectionConfig) -> np.ndarray:
    if config.blur_kernel_size > 1:
        k = config.blur_kernel_size | 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    # Adaptive threshold handles ambient lighting variation better than a fixed threshold.
    block_size = config.adaptive_block_size | 1
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        config.adaptive_c,
    )
    edges = cv2.Canny(thresh, config.canny_low, config.canny_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)


def _score_contour(
    contour: np.ndarray,
    image_area: float,
    expected_aspect_ratio: float,
    config: DetectionConfig,
) -> float:
    area = float(cv2.contourArea(contour))
    if area < config.min_contour_area_px or area > config.max_contour_area_px:
        return -1.0

    area_ratio = area / image_area
    if area_ratio < config.min_area_ratio or area_ratio > config.max_area_ratio:
        return -1.0

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 1e-6:
        return -1.0
    solidity = area / hull_area
    if solidity < config.min_solidity:
        return -1.0

    rect = cv2.minAreaRect(contour)
    aspect_ratio = _contour_aspect_ratio(rect)
    if not _aspect_ratio_matches(aspect_ratio, expected_aspect_ratio, config.aspect_ratio_tolerance):
        return -1.0

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, config.polygon_epsilon_ratio * peri, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return -1.0

    aspect_error = min(
        abs(aspect_ratio - expected_aspect_ratio) / expected_aspect_ratio,
        abs((1.0 / aspect_ratio) - expected_aspect_ratio) / expected_aspect_ratio,
    )
    return float(solidity * (1.0 - aspect_error) * area_ratio)


def _detect_from_template(
    gray: np.ndarray,
    template_path: str,
    threshold: float,
) -> Optional[np.ndarray]:
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        return None

    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val < threshold:
        return None

    h, w = template.shape[:2]
    top_left = np.array(max_loc, dtype=np.float32)
    # Template gives a bounding box; approximate corners for PnP seeding.
    return np.array(
        [
            top_left,
            top_left + [w, 0],
            top_left + [w, h],
            top_left + [0, h],
        ],
        dtype=np.float32,
    )


def detect_workpiece_corners(
    image_bgr: np.ndarray,
    dimensions: WorkpieceDimensions,
    config: DetectionConfig,
) -> Optional[DetectionResult]:
    if image_bgr is None or image_bgr.size == 0:
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_area = float(gray.shape[0] * gray.shape[1])
    expected_aspect_ratio = dimensions.aspect_ratio

    if config.template_enabled and config.template_path:
        template_corners = _detect_from_template(
            gray,
            config.template_path,
            config.template_match_threshold,
        )
        if template_corners is not None:
            return DetectionResult(
                corners=_order_corners_clockwise(template_corners),
                contour=template_corners.reshape(-1, 1, 2).astype(np.int32),
                score=1.0,
            )

    edges = _preprocess(gray, config)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    best_corners = None
    best_score = -1.0

    for contour in contours:
        score = _score_contour(contour, image_area, expected_aspect_ratio, config)
        if score <= best_score:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, config.polygon_epsilon_ratio * peri, True)
        if len(approx) != 4:
            continue

        corners = approx.reshape(4, 2).astype(np.float32)
        best_contour = contour
        best_corners = _order_corners_clockwise(corners)
        best_score = score

    if best_corners is None or best_contour is None:
        return None

    return DetectionResult(corners=best_corners, contour=best_contour, score=best_score)


def draw_detection_debug(image_bgr: np.ndarray, detection: DetectionResult) -> np.ndarray:
    debug = image_bgr.copy()
    cv2.drawContours(debug, [detection.contour], -1, (0, 255, 0), 2)
    for index, corner in enumerate(detection.corners):
        point = tuple(int(v) for v in corner)
        cv2.circle(debug, point, 6, (0, 0, 255), -1)
        cv2.putText(
            debug,
            str(index),
            (point[0] + 5, point[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return debug
