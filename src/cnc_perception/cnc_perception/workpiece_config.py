"""Load and validate workpiece model configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory


@dataclass(frozen=True)
class WorkpieceDimensions:
    width_m: float
    length_m: float
    thickness_m: float

    @property
    def aspect_ratio(self) -> float:
        return self.width_m / self.length_m

    def object_corners_centered(self) -> list[list[float]]:
        """Return 4 top-face corners in object coordinates (meters)."""
        half_w = self.width_m / 2.0
        half_l = self.length_m / 2.0
        return [
            [-half_w, -half_l, 0.0],
            [half_w, -half_l, 0.0],
            [half_w, half_l, 0.0],
            [-half_w, half_l, 0.0],
        ]


@dataclass(frozen=True)
class DetectionRoi:
    enabled: bool
    x_min_ratio: float
    y_min_ratio: float
    x_max_ratio: float
    y_max_ratio: float


@dataclass(frozen=True)
class DetectionConfig:
    blur_kernel_size: int
    adaptive_block_size: int
    adaptive_c: int
    canny_low: int
    canny_high: int
    min_contour_area_px: float
    max_contour_area_px: float
    aspect_ratio_tolerance: float
    polygon_epsilon_ratio: float
    min_solidity: float
    min_interior_contrast: float
    min_area_ratio: float
    max_area_ratio: float
    try_inverted_threshold: bool
    try_otsu_threshold: bool
    use_relaxed_fallback: bool
    roi: DetectionRoi
    template_enabled: bool
    template_path: str
    template_match_threshold: float
    use_ippe_for_planar: bool
    publish_debug_image: bool
    pose_smoothing_alpha: float
    assume_flat_on_bed: bool
    bed_pose_smoothing_alpha: float


def _resolve_package_uri(uri: str) -> str:
    if not uri.startswith('package://'):
        return uri
    remainder = uri[len('package://'):]
    package_name, _, relative_path = remainder.partition('/')
    try:
        share_dir = get_package_share_directory(package_name)
    except PackageNotFoundError as exc:
        raise FileNotFoundError(f'ROS package not found for URI: {uri}') from exc
    return str(Path(share_dir) / relative_path)


def load_workpiece_config(config_path: str) -> tuple[WorkpieceDimensions, DetectionConfig]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f'Workpiece config not found: {config_path}')

    with path.open('r', encoding='utf-8') as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    workpiece = raw.get('workpiece', {})
    detection = raw.get('detection', {})
    size_filter = raw.get('size_filter', {})
    template = raw.get('template', {})
    pose = raw.get('pose', {})

    dimensions = WorkpieceDimensions(
        width_m=float(workpiece['width_m']),
        length_m=float(workpiece['length_m']),
        thickness_m=float(workpiece['thickness_m']),
    )

    template_path = _resolve_package_uri(str(template.get('path', ''))) if template.get('path') else ''
    roi_raw = detection.get('roi', {})

    config = DetectionConfig(
        blur_kernel_size=int(detection.get('blur_kernel_size', 5)),
        adaptive_block_size=int(detection.get('adaptive_block_size', 31)),
        adaptive_c=int(detection.get('adaptive_c', 5)),
        canny_low=int(detection.get('canny_low', 40)),
        canny_high=int(detection.get('canny_high', 120)),
        min_contour_area_px=float(detection.get('min_contour_area_px', 2500)),
        max_contour_area_px=float(detection.get('max_contour_area_px', 500000)),
        aspect_ratio_tolerance=float(detection.get('aspect_ratio_tolerance', 0.12)),
        polygon_epsilon_ratio=float(detection.get('polygon_epsilon_ratio', 0.02)),
        min_solidity=float(detection.get('min_solidity', 0.85)),
        min_interior_contrast=float(detection.get('min_interior_contrast', 4.0)),
        min_area_ratio=float(size_filter.get('min_area_ratio', 0.02)),
        max_area_ratio=float(size_filter.get('max_area_ratio', 0.80)),
        try_inverted_threshold=bool(detection.get('try_inverted_threshold', True)),
        try_otsu_threshold=bool(detection.get('try_otsu_threshold', True)),
        use_relaxed_fallback=bool(detection.get('use_relaxed_fallback', True)),
        roi=DetectionRoi(
            enabled=bool(roi_raw.get('enabled', False)),
            x_min_ratio=float(roi_raw.get('x_min_ratio', 0.0)),
            y_min_ratio=float(roi_raw.get('y_min_ratio', 0.0)),
            x_max_ratio=float(roi_raw.get('x_max_ratio', 1.0)),
            y_max_ratio=float(roi_raw.get('y_max_ratio', 1.0)),
        ),
        template_enabled=bool(template.get('enabled', False)),
        template_path=template_path,
        template_match_threshold=float(template.get('match_threshold', 0.72)),
        use_ippe_for_planar=bool(pose.get('use_ippe_for_planar', True)),
        publish_debug_image=bool(pose.get('publish_debug_image', True)),
        pose_smoothing_alpha=float(pose.get('pose_smoothing_alpha', 0.35)),
        assume_flat_on_bed=bool(pose.get('assume_flat_on_bed', True)),
        bed_pose_smoothing_alpha=float(pose.get('bed_pose_smoothing_alpha', 0.85)),
    )
    return dimensions, config
