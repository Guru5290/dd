# Copyright 2026 CNC Perception Team

import math

import numpy as np

from cnc_perception.pose_ekf import BedPoseEkf, EkfConfig
from cnc_perception.transform_utils import yaw_from_matrix


def _transform(x: float, y: float, z: float, yaw_deg: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    yaw_rad = math.radians(yaw_deg)
    transform[0, 0] = math.cos(yaw_rad)
    transform[0, 1] = -math.sin(yaw_rad)
    transform[1, 0] = math.sin(yaw_rad)
    transform[1, 1] = math.cos(yaw_rad)
    transform[0, 3] = x
    transform[1, 3] = y
    transform[2, 3] = z
    return transform


def test_ekf_warmup_passes_raw_measurement() -> None:
    ekf = BedPoseEkf(
        square_mode=True,
        config=EkfConfig(
            enabled=True,
            warmup_sec=2.0,
            max_yaw_rate_deg_s=90.0,
            max_position_rate_m_s=0.5,
            process_noise_xy_m=0.001,
            process_noise_z_m=0.001,
            process_noise_yaw_deg=2.0,
            measurement_noise_xy_m=0.002,
            measurement_noise_z_m=0.001,
            measurement_noise_yaw_deg=3.0,
        ),
    )
    first = _transform(0.10, 0.10, 0.01, 10.0)
    result = ekf.filter_transform(first, 0.0)
    assert result.phase == 'WARMUP'
    assert abs(result.transform[0, 3] - 0.10) < 1e-9


def test_ekf_rejects_square_yaw_jump_after_warmup() -> None:
    ekf = BedPoseEkf(
        square_mode=True,
        config=EkfConfig(
            enabled=True,
            warmup_sec=0.0,
            max_yaw_rate_deg_s=90.0,
            max_position_rate_m_s=0.5,
            process_noise_xy_m=0.001,
            process_noise_z_m=0.001,
            process_noise_yaw_deg=2.0,
            measurement_noise_xy_m=0.002,
            measurement_noise_z_m=0.001,
            measurement_noise_yaw_deg=3.0,
        ),
    )
    first = _transform(0.10, 0.10, 0.01, 0.0)
    ekf.filter_transform(first, 1.0)
    jumped = _transform(0.10, 0.10, 0.01, 90.0)
    result = ekf.filter_transform(jumped, 1.033)
    assert result.phase == 'EKF'
    assert result.gated is True
    assert abs(yaw_from_matrix(result.transform[:3, :3])) < 5.0
