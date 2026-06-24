#!/usr/bin/env python3
"""Rectify /image_raw with OpenCV and publish /image_rect_color (+ rectified camera_info)."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from cnc_perception.camera_frames import OPTICAL_FRAME


class RectifyImageNode(Node):
    """Reliable alternative to image_proc::RectifyNode for usb_cam pipelines."""

    def __init__(self) -> None:
        super().__init__('image_rectifier')
        self.declare_parameter('input_topic', '/image_raw')
        self.declare_parameter('output_topic', '/image_rect_color')
        self.declare_parameter('raw_camera_info_topic', '/camera_info_raw')
        self.declare_parameter('camera_info_topic', '/camera_info')

        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        raw_info_topic = self.get_parameter('raw_camera_info_topic').get_parameter_value().string_value
        info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value

        self._bridge = CvBridge()
        self._map1: Optional[np.ndarray] = None
        self._map2: Optional[np.ndarray] = None
        self._rect_camera_info: Optional[CameraInfo] = None

        self._image_pub = self.create_publisher(Image, output_topic, qos_profile_sensor_data)
        self._info_pub = self.create_publisher(CameraInfo, info_topic, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, raw_info_topic, self._info_cb, qos_profile_sensor_data)
        self.create_subscription(Image, input_topic, self._image_cb, qos_profile_sensor_data)

        self.get_logger().info(
            f'Rectifier: {input_topic} + {raw_info_topic} -> {output_topic} + {info_topic}'
        )

    def _info_cb(self, msg: CameraInfo) -> None:
        if len(msg.k) != 9:
            self.get_logger().warn('Invalid camera_info K matrix', throttle_duration_sec=5.0)
            return

        camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        distortion = np.array(msg.d, dtype=np.float64).reshape(-1, 1)
        image_size = (int(msg.width), int(msg.height))

        if np.allclose(distortion, 0.0):
            new_camera_matrix = camera_matrix.copy()
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                camera_matrix,
                distortion,
                None,
                new_camera_matrix,
                image_size,
                cv2.CV_16SC2,
            )
        else:
            new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                camera_matrix,
                distortion,
                image_size,
                alpha=0.0,
            )
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                camera_matrix,
                distortion,
                None,
                new_camera_matrix,
                image_size,
                cv2.CV_16SC2,
            )

        rect_info = CameraInfo()
        rect_info.header = msg.header
        rect_info.height = msg.height
        rect_info.width = msg.width
        rect_info.distortion_model = msg.distortion_model
        rect_info.d = [0.0] * max(len(msg.d), 5)
        rect_info.k = new_camera_matrix.reshape(-1).tolist()
        rect_info.r = msg.r if msg.r else [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        rect_info.p = [
            new_camera_matrix[0, 0], 0.0, new_camera_matrix[0, 2], 0.0,
            0.0, new_camera_matrix[1, 1], new_camera_matrix[1, 2], 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        rect_info.binning_x = msg.binning_x
        rect_info.binning_y = msg.binning_y
        self._rect_camera_info = rect_info
        self.get_logger().info(
            f'Rectification maps ready ({msg.width}x{msg.height}, D cleared in output camera_info)'
        )

    def _image_cb(self, msg: Image) -> None:
        if self._map1 is None or self._map2 is None or self._rect_camera_info is None:
            self.get_logger().warn(
                'Waiting for /camera_info_raw before rectifying...',
                throttle_duration_sec=4.0,
            )
            return

        try:
            cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as exc:
            self.get_logger().warn(f'cv_bridge failed: {exc}', throttle_duration_sec=4.0)
            return

        rectified = cv2.remap(cv_image, self._map1, self._map2, cv2.INTER_LINEAR)
        encoding = msg.encoding
        if len(rectified.shape) == 2:
            encoding = 'mono8'
        elif rectified.shape[2] == 3 and encoding in ('', 'passthrough'):
            encoding = 'bgr8'

        try:
            out_msg = self._bridge.cv2_to_imgmsg(rectified, encoding=encoding)
        except CvBridgeError as exc:
            self.get_logger().warn(f'Failed to convert rectified image: {exc}', throttle_duration_sec=4.0)
            return

        out_msg.header = msg.header
        out_msg.header.frame_id = OPTICAL_FRAME
        self._image_pub.publish(out_msg)

        info_msg = CameraInfo()
        info_msg.header = out_msg.header
        info_msg.height = self._rect_camera_info.height
        info_msg.width = self._rect_camera_info.width
        info_msg.distortion_model = self._rect_camera_info.distortion_model
        info_msg.d = list(self._rect_camera_info.d)
        info_msg.k = list(self._rect_camera_info.k)
        info_msg.r = list(self._rect_camera_info.r)
        info_msg.p = list(self._rect_camera_info.p)
        info_msg.binning_x = self._rect_camera_info.binning_x
        info_msg.binning_y = self._rect_camera_info.binning_y
        self._info_pub.publish(info_msg)


def main() -> None:
    rclpy.init()
    node = RectifyImageNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
