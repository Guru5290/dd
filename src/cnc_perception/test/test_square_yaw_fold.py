# Copyright 2026 CNC Perception Team

from cnc_perception.transform_utils import fold_square_yaw_deg


def test_fold_square_yaw_deg_examples() -> None:
    assert abs(fold_square_yaw_deg(0.0) - 0.0) < 1e-9
    assert abs(fold_square_yaw_deg(90.0) - 0.0) < 1e-9
    assert abs(fold_square_yaw_deg(118.0) - 28.0) < 1e-9
    assert abs(fold_square_yaw_deg(180.0) - 0.0) < 1e-9
    assert abs(fold_square_yaw_deg(270.0) - 0.0) < 1e-9
    assert abs(fold_square_yaw_deg(360.0) - 0.0) < 1e-9
