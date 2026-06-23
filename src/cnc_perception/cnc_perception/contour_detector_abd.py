from __future__ import annotations

import os
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


@dataclass(frozen=True)
class ContourCandidate:
    corners: np.ndarray
    contour: np.ndarray
    score: float
    reason: str


# --- small config helper -----------------------------------------------------

def _cfg(config: DetectionConfig, name: str, default):
    """Read an optional config field, falling back to a default if absent."""
    return getattr(config, name, default)


# --- background reference cache ----------------------------------------------

_BACKGROUND_CACHE: dict[str, Optional[np.ndarray]] = {}


def _load_background(path: str) -> Optional[np.ndarray]:
    if not path:
        return None
    if path in _BACKGROUND_CACHE:
        return _BACKGROUND_CACHE[path]
    image = cv2.imread(path, cv2.IMREAD_COLOR) if os.path.exists(path) else None
    _BACKGROUND_CACHE[path] = image
    return image


# --- corner helpers -----------------------------------------------------------

def _order_corners_clockwise(points: np.ndarray) -> np.ndarray:
    """Order 4 points consistently (TL, TR, BR, BL) and robustly to rotation.

    The classic sum/diff trick breaks for quads near 45 deg because two corners
    can share the smallest/largest x+y. Sorting by angle around the centroid
    gives a stable winding at any rotation; we then roll the start to the
    top-left-most corner so the order matches object_corners_centered().
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]            # consistent winding (clockwise on screen)
    start = int(np.argmin(pts.sum(axis=1)))  # top-left-most corner first
    pts = np.roll(pts, -start, axis=0)
    return pts.astype(np.float32)


def _refine_corners_subpix(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Refine corners to sub-pixel accuracy; ignore corners on the image border."""
    h, w = gray.shape[:2]
    pts = corners.reshape(-1, 1, 2).astype(np.float32)
    margin = 6
    if np.any(pts[:, 0, 0] < margin) or np.any(pts[:, 0, 0] > w - margin) \
            or np.any(pts[:, 0, 1] < margin) or np.any(pts[:, 0, 1] > h - margin):
        return corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    try:
        cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1), criteria)
    except cv2.error:
        return corners
    return pts.reshape(4, 2).astype(np.float32)


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


# --- mask building (shadow-resistant) ----------------------------------------

def _flatten_illumination(gray: np.ndarray, ksize: int) -> np.ndarray:
    """Divide by a heavily-blurred copy to remove slow brightness gradients.

    This flattens lighting and suppresses soft/penumbra shadows so that
    thresholding keys on the object, not the shadow.
    """
    if ksize <= 1:
        return gray
    ksize |= 1
    background = cv2.GaussianBlur(gray, (ksize, ksize), 0).astype(np.float32)
    background[background < 1.0] = 1.0
    flat = (gray.astype(np.float32) / background) * 128.0
    return np.clip(flat, 0, 255).astype(np.uint8)


def _clean_mask(mask: np.ndarray, open_k: int, close_k: int) -> np.ndarray:
    """OPEN to break the shadow bridge + remove specks, then CLOSE to fill the part."""
    if open_k > 1:
        open_k |= 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    if close_k > 1:
        close_k |= 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _foreground_masks(image_bgr: np.ndarray, config: DetectionConfig) -> list[tuple[str, np.ndarray]]:
    """Produce several candidate filled foreground masks (white = foreground)."""
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blur_k = _cfg(config, 'blur_kernel_size', 5)
    if blur_k > 1:
        gray = cv2.GaussianBlur(gray, (blur_k | 1, blur_k | 1), 0)

    flatten = _cfg(config, 'illumination_flatten', True)
    illum_k = _cfg(config, 'illumination_kernel', 0)
    if illum_k <= 0:
        illum_k = max(15, (min(h, w) // 4) | 1)
    flat = _flatten_illumination(gray, illum_k) if flatten else gray

    open_k = _cfg(config, 'morph_open_ksize', 7)
    close_k = _cfg(config, 'morph_close_ksize', 7)
    block_size = _cfg(config, 'adaptive_block_size', 51) | 1
    adaptive_c = _cfg(config, 'adaptive_c', 5)

    variants: list[tuple[str, np.ndarray]] = []

    # Otsu (both polarities) on the illumination-flattened image.
    if _cfg(config, 'try_otsu_threshold', True):
        _, m = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(('otsu', _clean_mask(m, open_k, close_k)))
        _, m = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        variants.append(('otsu_inv', _clean_mask(m, open_k, close_k)))

    # Adaptive threshold (normal + optional inverted).
    m = cv2.adaptiveThreshold(flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, block_size, adaptive_c)
    variants.append(('adaptive', _clean_mask(m, open_k, close_k)))
    if _cfg(config, 'try_inverted_threshold', True):
        m = cv2.adaptiveThreshold(flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, block_size, adaptive_c)
        variants.append(('adaptive_inv', _clean_mask(m, open_k, close_k)))

    # Saturation variant: shadow-IMMUNE for coloured stock (shadow keeps hue,
    # only value drops). Harmless for grey stock (yields ~empty mask).
    if _cfg(config, 'try_saturation', True):
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        _, m = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(('saturation', _clean_mask(m, open_k, close_k)))

    # Optional empty-bed background subtraction (strongest when available).
    bg_path = _cfg(config, 'background_image_path', '')
    background = _load_background(bg_path)
    if background is not None and background.shape == image_bgr.shape:
        diff = cv2.absdiff(image_bgr, background)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        thr = _cfg(config, 'background_diff_threshold', 30)
        _, m = cv2.threshold(diff_gray, thr, 255, cv2.THRESH_BINARY)
        variants.append(('background', _clean_mask(m, open_k, close_k)))

    return variants


# --- scoring ------------------------------------------------------------------

def _score_contour(
    contour: np.ndarray,
    image_area: float,
    expected_aspect_ratio: float,
    config: DetectionConfig,
) -> tuple[float, str]:
    area = float(cv2.contourArea(contour))
    if area < _cfg(config, 'min_contour_area_px', 1000.0):
        return -1.0, f'area {area:.0f} < min {_cfg(config, "min_contour_area_px", 1000.0):.0f}'
    if area > _cfg(config, 'max_contour_area_px', 1e9):
        return -1.0, f'area {area:.0f} > max {_cfg(config, "max_contour_area_px", 1e9):.0f}'

    area_ratio = area / image_area
    if area_ratio < _cfg(config, 'min_area_ratio', 0.001):
        return -1.0, f'area_ratio {area_ratio:.4f} < min'
    if area_ratio > _cfg(config, 'max_area_ratio', 0.95):
        return -1.0, f'area_ratio {area_ratio:.4f} > max'

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 1e-6:
        return -1.0, 'degenerate hull'
    solidity = area / hull_area
    # Solidity is the main shadow filter: an object+shadow union is non-convex,
    # so its solidity drops. Default raised to 0.90.
    if solidity < _cfg(config, 'min_solidity', 0.90):
        return -1.0, f'solidity {solidity:.2f} < min {_cfg(config, "min_solidity", 0.90):.2f}'

    rect = cv2.minAreaRect(contour)
    aspect_ratio = _contour_aspect_ratio(rect)
    tol = _cfg(config, 'aspect_ratio_tolerance', 0.2)
    if not _aspect_ratio_matches(aspect_ratio, expected_aspect_ratio, tol):
        return -1.0, f'aspect {aspect_ratio:.2f} vs expected {expected_aspect_ratio:.2f}'

    peri = cv2.arcLength(contour, True)
    eps = _cfg(config, 'polygon_epsilon_ratio', 0.02)
    approx = cv2.approxPolyDP(contour, eps * peri, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return -1.0, f'not convex quad (vertices={len(approx)})'

    # Extent (contour area vs its minAreaRect area) is a second rectangularity
    # check that also penalises shadow-fattened blobs.
    rect_area = max(rect[1][0] * rect[1][1], 1e-6)
    extent = area / rect_area
    if extent < _cfg(config, 'min_extent', 0.80):
        return -1.0, f'extent {extent:.2f} < min'

    aspect_error = min(
        abs(aspect_ratio - expected_aspect_ratio) / max(expected_aspect_ratio, 1e-6),
        abs((1.0 / aspect_ratio) - expected_aspect_ratio) / max(expected_aspect_ratio, 1e-6),
    )
    score = float(solidity * extent * (1.0 - aspect_error) * area_ratio)
    return score, 'ok'


# --- public API ---------------------------------------------------------------

def diagnose_contours(
    image_bgr: np.ndarray,
    dimensions: WorkpieceDimensions,
    config: DetectionConfig,
    top_n: int = 10,
) -> tuple[list[ContourCandidate], Optional[np.ndarray]]:
    """Return ranked contour candidates and the best mask image for debugging."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_area = float(gray.shape[0] * gray.shape[1])
    expected_aspect_ratio = dimensions.aspect_ratio
    eps_ratio = _cfg(config, 'polygon_epsilon_ratio', 0.02)
    subpix = _cfg(config, 'subpixel_refine', True)

    candidates: list[ContourCandidate] = []
    best_mask = None
    best_score = -1.0

    for variant_name, mask in _foreground_masks(image_bgr, config):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            score, reason = _score_contour(contour, image_area, expected_aspect_ratio, config)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, eps_ratio * peri, True)
            if len(approx) == 4:
                corners = _order_corners_clockwise(approx.reshape(4, 2).astype(np.float32))
                if subpix:
                    corners = _refine_corners_subpix(gray, corners)
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
                    best_mask = mask

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:top_n], best_mask


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
