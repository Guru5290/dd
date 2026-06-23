"""Contour-based workpiece detection for plain rectangular stock."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from cnc_perception.workpiece_config import DetectionConfig, WorkpieceDimensions


def _interior_exterior_contrast(gray: np.ndarray, contour: np.ndarray) -> float:
    """Higher when contour interior is darker than the local background (typical workpiece on metal bed)."""
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    inner_mean = cv2.mean(gray, mask=mask)[0]

    dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=2)
    ring = cv2.subtract(dilated, mask)
    if cv2.countNonZero(ring) == 0:
        return 0.0
    outer_mean = cv2.mean(gray, mask=ring)[0]
    return float(max(0.0, outer_mean - inner_mean))


@dataclass(frozen=True)
class DetectionResult:
    corners: np.ndarray
    contour: np.ndarray
    score: float


@dataclass(frozen=True)
class ContourCandidate:
    corners: np.ndarray
    contour: np.ndarray
    score: float
    reason: str


def _order_corners_clockwise(points: np.ndarray) -> np.ndarray:
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
    direct_error = abs(candidate_ratio - expected_ratio) / max(expected_ratio, 1e-6)
    inverse_error = abs((1.0 / candidate_ratio) - expected_ratio) / max(expected_ratio, 1e-6)
    return min(direct_error, inverse_error) <= tolerance


def _edges_from_gray(gray: np.ndarray, config: DetectionConfig, invert: bool) -> np.ndarray:
    if config.blur_kernel_size > 1:
        k = config.blur_kernel_size | 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    block_size = config.adaptive_block_size | 1
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresh_type,
        block_size,
        config.adaptive_c,
    )
    edges = cv2.Canny(thresh, config.canny_low, config.canny_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)


def _edges_otsu(gray: np.ndarray, config: DetectionConfig) -> np.ndarray:
    if config.blur_kernel_size > 1:
        k = config.blur_kernel_size | 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, config.canny_low, config.canny_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)


def _preprocess_variants(gray: np.ndarray, config: DetectionConfig) -> list[tuple[str, np.ndarray]]:
    variants = [('normal', _edges_from_gray(gray, config, invert=False))]
    if config.try_inverted_threshold:
        variants.append(('inverted', _edges_from_gray(gray, config, invert=True)))
    if config.try_otsu_threshold:
        variants.append(('otsu', _edges_otsu(gray, config)))
    return variants


def _score_contour(
    contour: np.ndarray,
    image_area: float,
    expected_aspect_ratio: float,
    config: DetectionConfig,
    gray: Optional[np.ndarray] = None,
) -> tuple[float, str]:
    area = float(cv2.contourArea(contour))
    if area < config.min_contour_area_px:
        return -1.0, f'area {area:.0f} < min {config.min_contour_area_px:.0f}'
    if area > config.max_contour_area_px:
        return -1.0, f'area {area:.0f} > max {config.max_contour_area_px:.0f}'

    area_ratio = area / image_area
    if area_ratio < config.min_area_ratio:
        return -1.0, f'area_ratio {area_ratio:.4f} < min {config.min_area_ratio:.4f}'
    if area_ratio > config.max_area_ratio:
        return -1.0, f'area_ratio {area_ratio:.4f} > max {config.max_area_ratio:.4f}'

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 1e-6:
        return -1.0, 'degenerate hull'
    solidity = area / hull_area
    if solidity < config.min_solidity:
        return -1.0, f'solidity {solidity:.2f} < min {config.min_solidity:.2f}'

    rect = cv2.minAreaRect(contour)
    aspect_ratio = _contour_aspect_ratio(rect)
    if not _aspect_ratio_matches(aspect_ratio, expected_aspect_ratio, config.aspect_ratio_tolerance):
        return -1.0, f'aspect {aspect_ratio:.2f} vs expected {expected_aspect_ratio:.2f}'

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, config.polygon_epsilon_ratio * peri, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return -1.0, f'not convex quad (vertices={len(approx)})'

    aspect_error = min(
        abs(aspect_ratio - expected_aspect_ratio) / max(expected_aspect_ratio, 1e-6),
        abs((1.0 / aspect_ratio) - expected_aspect_ratio) / max(expected_aspect_ratio, 1e-6),
    )
    score = float(solidity * (1.0 - aspect_error) * area_ratio)

    if gray is not None:
        contrast = _interior_exterior_contrast(gray, contour)
        if contrast < 8.0:
            return -1.0, f'low interior contrast {contrast:.1f} (shadow/clutter?)'
        score *= min(1.0, contrast / 25.0)

    return score, 'ok'


def diagnose_contours(
    image_bgr: np.ndarray,
    dimensions: WorkpieceDimensions,
    config: DetectionConfig,
    top_n: int = 10,
) -> tuple[list[ContourCandidate], Optional[np.ndarray]]:
    """Return ranked contour candidates and best edge image for debugging."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_area = float(gray.shape[0] * gray.shape[1])
    expected_aspect_ratio = dimensions.aspect_ratio

    candidates: list[ContourCandidate] = []
    best_edges = None
    best_score = -1.0

    for variant_name, edges in _preprocess_variants(gray, config):
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            score, reason = _score_contour(
                contour, image_area, expected_aspect_ratio, config, gray=gray
            )
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, config.polygon_epsilon_ratio * peri, True)
            if len(approx) == 4:
                corners = _order_corners_clockwise(approx.reshape(4, 2).astype(np.float32))
                candidates.append(
                    ContourCandidate(
                        corners=corners,
                        contour=contour,
                        score=score,
                        reason=f'{variant_name}: {reason}',
                    )
                )
                if score > best_score:
                    best_score = score
                    best_edges = edges

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:top_n], best_edges


def detect_workpiece_corners(
    image_bgr: np.ndarray,
    dimensions: WorkpieceDimensions,
    config: DetectionConfig,
) -> Optional[DetectionResult]:
    if image_bgr is None or image_bgr.size == 0:
        return None

    candidates, _ = diagnose_contours(image_bgr, dimensions, config, top_n=1)
    if not candidates or candidates[0].score <= 0.0:
        return None

    best = candidates[0]
    return DetectionResult(corners=best.corners, contour=best.contour, score=best.score)


def draw_detection_debug(
    image_bgr: np.ndarray,
    detection: DetectionResult,
    label: str = '',
) -> np.ndarray:
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
    if label:
        cv2.putText(
            debug,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return debug


def draw_diagnostic_overlay(
    image_bgr: np.ndarray,
    candidates: list[ContourCandidate],
    edges: Optional[np.ndarray],
) -> np.ndarray:
    debug = image_bgr.copy()
    for index, candidate in enumerate(candidates[:5]):
        color = (0, 255, 0) if candidate.score > 0 else (0, 0, 255)
        cv2.drawContours(debug, [candidate.contour], -1, color, 2)
        moment = cv2.moments(candidate.contour)
        if moment['m00'] > 0:
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
            cv2.putText(
                debug,
                f'#{index} s={candidate.score:.3f}',
                (cx - 40, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    if edges is not None:
        small = cv2.resize(edges, (debug.shape[1] // 4, debug.shape[0] // 4))
        small_bgr = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        h, w = small_bgr.shape[:2]
        debug[0:h, 0:w] = small_bgr
    return debug


