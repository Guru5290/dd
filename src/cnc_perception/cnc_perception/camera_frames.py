"""ROS camera_link vs camera_optical_frame transforms (REP-103 / REP-105)."""

from __future__ import annotations

import numpy as np

# OpenCV / camera_optical_frame: +X right, +Y down, +Z forward (into scene).
# camera_link (REP-103): +X forward, +Y left, +Z up.
OPTICAL_FRAME = 'camera_optical_frame'
LINK_FRAME = 'camera_link'


def rotation_link_to_optical() -> np.ndarray:
    """3x3 rotation: vectors expressed in optical frame -> link frame is R @ v_opt."""
    return np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )


def transform_link_to_optical() -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_link_to_optical()
    return matrix


def transform_optical_to_link() -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_link_to_optical().T
    return matrix
