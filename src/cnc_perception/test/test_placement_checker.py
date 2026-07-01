# Copyright 2026 CNC Perception Team

import math

import numpy as np

from cnc_perception.bed_config import TargetPlacement
from cnc_perception.placement_checker import check_placement


def _transform(x_m: float, y_m: float, z_m: float, yaw_deg: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    yaw_rad = math.radians(yaw_deg)
    transform[0, 0] = math.cos(yaw_rad)
    transform[0, 1] = -math.sin(yaw_rad)
    transform[1, 0] = math.sin(yaw_rad)
    transform[1, 1] = math.cos(yaw_rad)
    transform[0, 3] = x_m
    transform[1, 3] = y_m
    transform[2, 3] = z_m
    return transform


def _target(**overrides) -> TargetPlacement:
    defaults = dict(
        x_m=0.100,
        y_m=0.100,
        z_m=0.0,
        yaw_deg=0.0,
        tolerance_x_mm=2.0,
        tolerance_y_mm=3.0,
        tolerance_yaw_deg=5.0,
    )
    defaults.update(overrides)
    return TargetPlacement(**defaults)


def test_placement_ok_within_separate_xy_yaw_tolerances() -> None:
    result = check_placement(
        _transform(0.101, 0.102, 0.012, 1.0),
        workpiece_thickness_m=0.010,
        target=_target(),
    )
    assert result.ok is True
    assert 'CORRECT POSITION' in result.message


def test_placement_fails_when_only_x_out_of_tolerance() -> None:
    result = check_placement(
        _transform(0.1035, 0.1005, 0.010, 0.0),
        workpiece_thickness_m=0.010,
        target=_target(tolerance_x_mm=2.0, tolerance_y_mm=3.0),
    )
    assert result.ok is False
    assert abs(result.dx_mm) > 2.0
    assert abs(result.dy_mm) <= 3.0


def test_placement_ignores_z_height() -> None:
    result = check_placement(
        _transform(0.100, 0.100, 0.020, 0.0),
        workpiece_thickness_m=0.010,
        target=_target(),
    )
    assert result.ok is True


def test_placement_fails_on_yaw_only() -> None:
    result = check_placement(
        _transform(0.100, 0.100, 0.010, 8.0),
        workpiece_thickness_m=0.010,
        target=_target(tolerance_yaw_deg=5.0),
    )
    assert result.ok is False
    assert abs(result.dyaw_deg) > 5.0
