# Copyright 2026 CNC Perception Team
#
# Licensed under the Apache License, Version 2.0

import numpy as np

from cnc_perception.pose_solver import rotation_matrix_to_quaternion


def test_rotation_matrix_to_quaternion_identity() -> None:
    identity = np.eye(3)
    quat = rotation_matrix_to_quaternion(identity)
    assert abs(quat.w - 1.0) < 1e-3
    assert abs(quat.x) < 1e-3
    assert abs(quat.y) < 1e-3
    assert abs(quat.z) < 1e-3
