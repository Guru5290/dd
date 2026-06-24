"""Convert sensor_msgs/Image without cv_bridge (avoids NumPy 2.x / cv_bridge crashes)."""

from __future__ import annotations

import cv2
import numpy as np
from sensor_msgs.msg import CameraInfo, Image


SUPPORTED_ENCODINGS = frozenset({
    'bgr8',
    'rgb8',
    'mono8',
    '8uc1',
    'yuv422',
    'yuv422_yuy2',
    'uyvy',
})


def distortion_from_camera_info(msg: CameraInfo) -> np.ndarray:
    if msg.d:
        coeffs = np.array(msg.d, dtype=np.float64).reshape(-1, 1)
        if coeffs.size > 0:
            return coeffs
    if msg.distortion_model == 'plumb_bob':
        return np.zeros((5, 1), dtype=np.float64)
    return np.zeros((4, 1), dtype=np.float64)


def image_msg_to_bgr(msg: Image) -> np.ndarray:
    encoding = (msg.encoding or 'bgr8').lower()
    if msg.height <= 0 or msg.width <= 0 or not msg.data:
        raise ValueError(f'Invalid image message ({msg.width}x{msg.height}, encoding={msg.encoding})')

    buffer = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding == 'bgr8':
        return buffer.reshape((msg.height, msg.width, 3))
    if encoding == 'rgb8':
        rgb = buffer.reshape((msg.height, msg.width, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if encoding in ('mono8', '8uc1'):
        gray = buffer.reshape((msg.height, msg.width))
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if encoding in ('yuv422', 'yuv422_yuy2'):
        yuy2 = buffer.reshape((msg.height, msg.width, 2))
        return cv2.cvtColor(yuy2, cv2.COLOR_YUV2BGR_YUY2)
    if encoding == 'uyvy':
        uyvy = buffer.reshape((msg.height, msg.width, 2))
        return cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)

    raise ValueError(
        f'Unsupported image encoding: {msg.encoding!r}. '
        f'Set usb_cam pixel_format to mjpeg2rgb in camera_params.yaml.'
    )


def bgr_to_image_msg(image_bgr: np.ndarray, header, frame_id: str) -> Image:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError('Expected BGR image with shape (H, W, 3)')

    contiguous = np.ascontiguousarray(image_bgr)
    msg = Image()
    msg.header = header
    msg.header.frame_id = frame_id
    msg.height = int(contiguous.shape[0])
    msg.width = int(contiguous.shape[1])
    msg.encoding = 'bgr8'
    msg.is_bigendian = 0
    msg.step = int(contiguous.shape[1] * 3)
    msg.data = contiguous.tobytes()
    return msg
