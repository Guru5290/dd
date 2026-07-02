"""Bed-frame pose EKF with timestamp-based gating and square yaw handling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from cnc_perception.transform_utils import angle_diff_deg, yaw_from_matrix


@dataclass(frozen=True)
class EkfConfig:
    enabled: bool
    warmup_sec: float
    max_yaw_rate_deg_s: float
    max_position_rate_m_s: float
    process_noise_xy_m: float
    process_noise_z_m: float
    process_noise_yaw_deg: float
    measurement_noise_xy_m: float
    measurement_noise_z_m: float
    measurement_noise_yaw_deg: float
    recovery_enabled: bool = True
    recovery_gated_frames: int = 30
    recovery_off_target_xy_mm: float = 15.0
    recovery_off_target_frames: int = 45


@dataclass(frozen=True)
class EkfResult:
    transform: np.ndarray
    phase: str
    gated: bool


def _yaw_to_matrix(yaw_rad: float) -> np.ndarray:
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    return np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


class BedPoseEkf:
    """Kalman filter on [x, y, z, yaw] in cnc_bed_frame with timestamp-based gating."""

    def __init__(self, square_mode: bool, config: EkfConfig) -> None:
        self._square_mode = square_mode
        self._config = config
        self._state: Optional[np.ndarray] = None
        self._covariance: Optional[np.ndarray] = None
        self._warmup_start_sec: Optional[float] = None
        self._last_time_sec: Optional[float] = None
        self._filter_active = False

    def reset(self) -> None:
        self._state = None
        self._covariance = None
        self._warmup_start_sec = None
        self._last_time_sec = None
        self._filter_active = False

    def filter_transform(self, t_bed_workpiece: np.ndarray, time_sec: float) -> EkfResult:
        if not self._config.enabled:
            return EkfResult(t_bed_workpiece.copy(), 'RAW', gated=False)

        measurement = self._transform_to_state(t_bed_workpiece)

        if self._warmup_start_sec is None:
            self._warmup_start_sec = time_sec

        warmup_elapsed = time_sec - self._warmup_start_sec
        if warmup_elapsed < self._config.warmup_sec:
            return EkfResult(t_bed_workpiece.copy(), 'WARMUP', gated=False)

        if not self._filter_active:
            self._initialize(measurement)
            self._last_time_sec = time_sec
            self._filter_active = True
            return EkfResult(self._state_to_transform(t_bed_workpiece), 'EKF_INIT', gated=False)

        dt = self._compute_dt(time_sec)
        self._predict(dt)
        gated = self._is_outlier(measurement, dt)
        aligned_measurement = self._align_yaw_measurement(measurement)
        if not gated:
            self._update(aligned_measurement)
        self._last_time_sec = time_sec
        return EkfResult(self._state_to_transform(t_bed_workpiece), 'EKF', gated=gated)

    def _compute_dt(self, time_sec: float) -> float:
        if self._last_time_sec is None:
            return 0.0
        dt = float(time_sec - self._last_time_sec)
        if dt <= 0.0:
            return 0.0
        return min(dt, 0.5)

    def _transform_to_state(self, transform: np.ndarray) -> np.ndarray:
        yaw_rad = math.radians(yaw_from_matrix(transform[:3, :3]))
        return np.array(
            [
                float(transform[0, 3]),
                float(transform[1, 3]),
                float(transform[2, 3]),
                yaw_rad,
            ],
            dtype=np.float64,
        )

    def _state_to_transform(self, template: np.ndarray) -> np.ndarray:
        if self._state is None:
            return template.copy()
        result = template.copy()
        result[:3, :3] = _yaw_to_matrix(float(self._state[3]))
        result[0, 3] = float(self._state[0])
        result[1, 3] = float(self._state[1])
        result[2, 3] = float(self._state[2])
        return result

    def _initialize(self, measurement: np.ndarray) -> None:
        self._state = measurement.copy()
        yaw_var = math.radians(self._config.measurement_noise_yaw_deg) ** 2
        self._covariance = np.diag(
            [
                self._config.measurement_noise_xy_m**2,
                self._config.measurement_noise_xy_m**2,
                self._config.measurement_noise_z_m**2,
                yaw_var,
            ]
        ).astype(np.float64)

    def _predict(self, dt: float) -> None:
        if self._state is None or self._covariance is None or dt <= 0.0:
            return
        q_xy = self._config.process_noise_xy_m**2 * dt
        q_z = self._config.process_noise_z_m**2 * dt
        q_yaw = math.radians(self._config.process_noise_yaw_deg) ** 2 * dt
        self._covariance[0, 0] += q_xy
        self._covariance[1, 1] += q_xy
        self._covariance[2, 2] += q_z
        self._covariance[3, 3] += q_yaw

    def _align_yaw_measurement(self, measurement: np.ndarray) -> np.ndarray:
        if self._state is None:
            return measurement.copy()
        aligned = measurement.copy()
        predicted_yaw_deg = math.degrees(float(self._state[3]))
        measured_yaw_deg = math.degrees(float(measurement[3]))
        if self._square_mode:
            best_yaw_deg = measured_yaw_deg
            best_dist = abs(angle_diff_deg(measured_yaw_deg, predicted_yaw_deg))
            for k in range(-3, 4):
                candidate = measured_yaw_deg + 90.0 * k
                dist = abs(angle_diff_deg(candidate, predicted_yaw_deg))
                if dist < best_dist:
                    best_dist = dist
                    best_yaw_deg = candidate
            aligned[3] = math.radians(best_yaw_deg)
        else:
            aligned[3] = float(self._state[3]) + math.radians(
                angle_diff_deg(measured_yaw_deg, predicted_yaw_deg)
            )
        return aligned

    def _is_outlier(self, measurement: np.ndarray, dt: float) -> bool:
        if self._state is None or dt <= 0.0:
            return False
        max_pos_delta = self._config.max_position_rate_m_s * dt
        max_yaw_delta = math.radians(self._config.max_yaw_rate_deg_s * dt)

        dx = abs(float(measurement[0] - self._state[0]))
        dy = abs(float(measurement[1] - self._state[1]))
        dz = abs(float(measurement[2] - self._state[2]))
        measured_yaw_deg = math.degrees(float(measurement[3]))
        predicted_yaw_deg = math.degrees(float(self._state[3]))
        dyaw = abs(angle_diff_deg(measured_yaw_deg, predicted_yaw_deg))

        if dyaw > math.degrees(max_yaw_delta):
            return True
        return dx > max_pos_delta or dy > max_pos_delta or dz > max_pos_delta

    def _update(self, measurement: np.ndarray) -> None:
        if self._state is None or self._covariance is None:
            return
        yaw_var = math.radians(self._config.measurement_noise_yaw_deg) ** 2
        measurement_noise = np.diag(
            [
                self._config.measurement_noise_xy_m**2,
                self._config.measurement_noise_xy_m**2,
                self._config.measurement_noise_z_m**2,
                yaw_var,
            ]
        ).astype(np.float64)

        for index in range(4):
            innovation = float(measurement[index] - self._state[index])
            if index == 3:
                innovation = math.radians(
                    angle_diff_deg(
                        math.degrees(float(measurement[3])),
                        math.degrees(float(self._state[3])),
                    )
                )
            variance = float(self._covariance[index, index] + measurement_noise[index, index])
            if variance <= 1e-12:
                continue
            gain = float(self._covariance[index, index] / variance)
            self._state[index] += gain * innovation
            self._covariance[index, index] *= max(0.0, 1.0 - gain)


@dataclass
class RecoveryTracker:
    consecutive_gated: int = 0
    consecutive_off_target: int = 0

    def reset_counters(self) -> None:
        self.consecutive_gated = 0
        self.consecutive_off_target = 0


def check_auto_recovery(
    *,
    filter_phase: str,
    gated: bool,
    x_mm: float,
    y_mm: float,
    target_x_mm: float,
    target_y_mm: float,
    config: EkfConfig,
    tracker: RecoveryTracker,
) -> Optional[str]:
    """Return a reset reason when the filter should re-warmup, else None."""
    if not config.enabled or not config.recovery_enabled:
        return None
    if filter_phase in ('WARMUP', 'RAW', 'EKF_INIT'):
        tracker.reset_counters()
        return None

    if gated:
        tracker.consecutive_gated += 1
    else:
        tracker.consecutive_gated = 0

    distance_mm = math.hypot(x_mm - target_x_mm, y_mm - target_y_mm)
    if distance_mm > config.recovery_off_target_xy_mm:
        tracker.consecutive_off_target += 1
    else:
        tracker.consecutive_off_target = 0

    if tracker.consecutive_gated >= config.recovery_gated_frames:
        return (
            f'EKF rejected {tracker.consecutive_gated} consecutive measurements '
            f'(threshold {config.recovery_gated_frames})'
        )
    if tracker.consecutive_off_target >= config.recovery_off_target_frames:
        return (
            f'pose {distance_mm:.1f} mm from target for '
            f'{tracker.consecutive_off_target} frames '
            f'(threshold {config.recovery_off_target_xy_mm:.1f} mm)'
        )
    return None
