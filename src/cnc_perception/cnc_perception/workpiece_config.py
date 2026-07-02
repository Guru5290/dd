"""Load and validate workpiece model configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

from cnc_perception.pose_ekf import EkfConfig
from cnc_perception.workpiece_marker_pose import WorkpieceMarkerConfig


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
    use_clahe: bool
    clahe_clip_limit: float
    clahe_tile_size: int
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
    bed_yaw_smoothing_alpha: float
    square_yaw_stability_weight: float
    lost_frames_to_reset_smoothing: int


@dataclass(frozen=True)
class PoseEstimationSettings:
    mode: str
    shape_mode: str
    is_square_stock: bool
    ekf: EkfConfig
    marker: Optional[WorkpieceMarkerConfig]


def resolve_shape_mode(shape_mode: str, dimensions: WorkpieceDimensions) -> bool:
    normalized = shape_mode.strip().lower()
    if normalized == 'square':
        return True
    if normalized == 'rectangle':
        return False
    return abs(dimensions.aspect_ratio - 1.0) < 0.05


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


def load_workpiece_config(
    config_path: str,
) -> tuple[WorkpieceDimensions, DetectionConfig, PoseEstimationSettings]:
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
    pose_estimation = raw.get('pose_estimation', {})
    ekf_raw = pose_estimation.get('ekf', {})
    recovery_raw = ekf_raw.get('recovery', {})
    marker_raw = raw.get('workpiece_marker', pose_estimation.get('workpiece_marker', {}))

    dimensions = WorkpieceDimensions(
        width_m=float(workpiece['width_m']),
        length_m=float(workpiece['length_m']),
        thickness_m=float(workpiece['thickness_m']),
    )

    template_path = _resolve_package_uri(str(template.get('path', ''))) if template.get('path') else ''
    roi_raw = detection.get('roi', {})

    config = DetectionConfig(
        blur_kernel_size=int(detection.get('blur_kernel_size', 5)),
        use_clahe=bool(detection.get('use_clahe', False)),
        clahe_clip_limit=float(detection.get('clahe_clip_limit', 2.0)),
        clahe_tile_size=int(detection.get('clahe_tile_size', 8)),
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
        bed_pose_smoothing_alpha=float(pose.get('bed_pose_smoothing_alpha', 0.88)),
        bed_yaw_smoothing_alpha=float(
            pose.get('bed_yaw_smoothing_alpha', pose.get('bed_pose_smoothing_alpha', 0.88))
        ),
        square_yaw_stability_weight=float(pose.get('square_yaw_stability_weight', 1.5)),
        lost_frames_to_reset_smoothing=int(pose.get('lost_frames_to_reset_smoothing', 1)),
    )

    exclude_ids_raw = marker_raw.get('exclude_ids', [0])
    if isinstance(exclude_ids_raw, int):
        exclude_ids = (int(exclude_ids_raw),)
    else:
        exclude_ids = tuple(int(value) for value in exclude_ids_raw)

    marker_config = WorkpieceMarkerConfig(
        dictionary=str(marker_raw.get('dictionary', 'DICT_4X4_50')),
        marker_id=int(marker_raw.get('marker_id', 1)),
        marker_size_m=float(marker_raw.get('marker_size_m', 0.020)),
        exclude_ids=exclude_ids,
        center_x_m=float(marker_raw.get('center_x_m', 0.0)),
        center_y_m=float(marker_raw.get('center_y_m', 0.0)),
        yaw_offset_deg=float(marker_raw.get('yaw_offset_deg', 0.0)),
        fallback_to_markerless=bool(marker_raw.get('fallback_to_markerless', True)),
    )

    shape_mode = str(pose_estimation.get('shape_mode', 'auto'))
    pose_settings = PoseEstimationSettings(
        mode=str(pose_estimation.get('mode', 'markerless')).strip().lower(),
        shape_mode=shape_mode,
        is_square_stock=resolve_shape_mode(shape_mode, dimensions),
        ekf=EkfConfig(
            enabled=bool(ekf_raw.get('enabled', True)),
            warmup_sec=float(ekf_raw.get('warmup_sec', 15.0)),
            max_yaw_rate_deg_s=float(ekf_raw.get('max_yaw_rate_deg_s', 90.0)),
            max_position_rate_m_s=float(ekf_raw.get('max_position_rate_m_s', 0.5)),
            process_noise_xy_m=float(ekf_raw.get('process_noise_xy_m', 0.0005)),
            process_noise_z_m=float(ekf_raw.get('process_noise_z_m', 0.0002)),
            process_noise_yaw_deg=float(ekf_raw.get('process_noise_yaw_deg', 2.0)),
            measurement_noise_xy_m=float(ekf_raw.get('measurement_noise_xy_m', 0.002)),
            measurement_noise_z_m=float(ekf_raw.get('measurement_noise_z_m', 0.001)),
            measurement_noise_yaw_deg=float(ekf_raw.get('measurement_noise_yaw_deg', 3.0)),
            recovery_enabled=bool(recovery_raw.get('enabled', True)),
            recovery_gated_frames=int(recovery_raw.get('gated_frames', 30)),
            recovery_off_target_xy_mm=float(recovery_raw.get('off_target_xy_mm', 15.0)),
            recovery_off_target_frames=int(recovery_raw.get('off_target_frames', 45)),
        ),
        marker=marker_config,
    )
    return dimensions, config, pose_settings
