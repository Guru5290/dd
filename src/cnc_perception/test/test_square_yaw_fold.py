# Copyright 2026 CNC Perception Team

from cnc_perception.transform_utils import fold_square_yaw_deg


def test_fold_square_yaw_deg_near_zero_for_slight_tilt() -> None:
    assert abs(fold_square_yaw_deg(0.0) - 0.0) < 1e-9
    assert abs(fold_square_yaw_deg(5.0) - 5.0) < 1e-9
    assert abs(fold_square_yaw_deg(85.0) - 5.0) < 1e-9
    assert abs(fold_square_yaw_deg(87.0) - 3.0) < 1e-9
    assert abs(fold_square_yaw_deg(90.0) - 0.0) < 1e-9
    assert abs(fold_square_yaw_deg(118.0) - 28.0) < 1e-9
    assert abs(fold_square_yaw_deg(180.0) - 0.0) < 1e-9
    assert abs(fold_square_yaw_deg(270.0) - 0.0) < 1e-9
    assert abs(fold_square_yaw_deg(360.0) - 0.0) < 1e-9


def test_fold_square_yaw_deg_tracks_reference() -> None:
    assert abs(fold_square_yaw_deg(85.0, reference_yaw_deg=3.0) - 5.0) < 1e-9
    assert abs(fold_square_yaw_deg(-2.0, reference_yaw_deg=4.0) - 2.0) < 1e-9
