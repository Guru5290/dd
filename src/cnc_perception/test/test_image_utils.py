"""Tests for image_utils."""

import numpy as np
from sensor_msgs.msg import CameraInfo, Image

from cnc_perception.image_utils import (
    bgr_to_image_msg,
    distortion_from_camera_info,
    image_msg_to_bgr,
    rectification_matrix_from_camera_info,
)


def test_distortion_defaults_to_zeros() -> None:
    msg = CameraInfo()
    msg.distortion_model = 'plumb_bob'
    msg.d = []
    coeffs = distortion_from_camera_info(msg)
    assert coeffs.shape == (5, 1)
    assert np.allclose(coeffs, 0.0)


def test_rectification_matrix_handles_numpy_like_r() -> None:
    msg = CameraInfo()
    msg.r = np.eye(3, dtype=np.float64).reshape(-1)
    matrix = rectification_matrix_from_camera_info(msg)
    assert len(matrix) == 9


def test_bgr_round_trip() -> None:
    source = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    image_msg = Image()
    image_msg.height = 2
    image_msg.width = 2
    image_msg.encoding = 'bgr8'
    image_msg.step = 6
    image_msg.data = source.tobytes()

    decoded = image_msg_to_bgr(image_msg)
    assert decoded.shape == (2, 2, 3)
    assert np.array_equal(decoded, source)

    round_trip = bgr_to_image_msg(decoded, image_msg.header, 'camera_optical_frame')
    assert round_trip.encoding == 'bgr8'
    assert len(round_trip.data) == 12
