# Copyright 2026 CNC Perception Team
#
# Licensed under the Apache License, Version 2.0

from cnc_perception.workpiece_config import WorkpieceDimensions


def test_workpiece_aspect_ratio() -> None:
    dims = WorkpieceDimensions(width_m=0.12, length_m=0.08, thickness_m=0.01)
    assert abs(dims.aspect_ratio - 1.5) < 1e-6


def test_object_corners_centered() -> None:
    dims = WorkpieceDimensions(width_m=0.10, length_m=0.04, thickness_m=0.005)
    corners = dims.object_corners_centered()
    assert len(corners) == 4
    assert corners[0] == [-0.05, -0.02, 0.0]
    assert corners[2] == [0.05, 0.02, 0.0]
