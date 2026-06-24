#!/usr/bin/env python3
"""Rectify /image_raw with OpenCV and publish /image_rect_color (+ rectified camera_info)."""

from __future__ import annotations

import traceback
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo, Image

from cnc_perception.camera_frames import OPTICAL_FRAME
from cnc_perception.image_utils import (
    bgr_to_image_msg,
    distortion_from_camera_info,
    image_msg_to_bgr,
    rectification_matrix_from_camera_info,
    zero_distortion_list,
)


# Reliable output so RViz (default Reliable subscriber) receives images.
# Perception nodes may subscribe with sensor_data (Best Effort) — still compatible.
QOS_PUBLISH = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)


class RectifyImageNode(Node):
    """Rectify camera images without cv_bridge (NumPy 2.x safe)."""

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

        self._map1: Optional[np.ndarray] = None
        self._map2: Optional[np.ndarray] = None
        self._rect_camera_info: Optional[CameraInfo] = None
        self._warned_waiting_for_info = False

        self._image_pub = self.create_publisher(Image, output_topic, QOS_PUBLISH)
        self._info_pub = self.create_publisher(CameraInfo, info_topic, QOS_PUBLISH)
        self.create_subscription(CameraInfo, raw_info_topic, self._info_cb, qos_profile_sensor_data)
        self.create_subscription(Image, input_topic, self._image_cb, qos_profile_sensor_data)

        self.get_logger().info(
            f'Ready: {input_topic} + {raw_info_topic} -> {output_topic} + {info_topic}'
        )

    def _info_cb(self, msg: CameraInfo) -> None:
        try:
            if len(msg.k) != 9:
                self.get_logger().warn('camera_info.k must have 9 elements', throttle_duration_sec=5.0)
                return
            if msg.width <= 0 or msg.height <= 0:
                self.get_logger().warn('camera_info has invalid image size', throttle_duration_sec=5.0)
                return

            camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            distortion = distortion_from_camera_info(msg)
            image_size = (int(msg.width), int(msg.height))

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
            rect_info.distortion_model = msg.distortion_model or 'plumb_bob'
            rect_info.d = zero_distortion_list(msg)
            rect_info.k = new_camera_matrix.reshape(-1).tolist()
            rect_info.r = rectification_matrix_from_camera_info(msg)
            rect_info.p = [
                new_camera_matrix[0, 0], 0.0, new_camera_matrix[0, 2], 0.0,
                0.0, new_camera_matrix[1, 1], new_camera_matrix[1, 2], 0.0,
                0.0, 0.0, 1.0, 0.0,
            ]
            rect_info.binning_x = msg.binning_x
            rect_info.binning_y = msg.binning_y
            self._rect_camera_info = rect_info
            self._warned_waiting_for_info = False
            self.get_logger().info(
                f'Rectification maps ready ({msg.width}x{msg.height}); publishing rectified camera_info'
            )
        except Exception as exc:
            self.get_logger().error(f'Failed to build rectification maps: {exc}')

    def _image_cb(self, msg: Image) -> None:
        if self._map1 is None or self._map2 is None or self._rect_camera_info is None:
            if not self._warned_waiting_for_info:
                self.get_logger().warn(
                    'Waiting for valid /camera_info_raw before rectifying images...'
                )
                self._warned_waiting_for_info = True
            return

        try:
            bgr = image_msg_to_bgr(msg)
            rectified = cv2.remap(bgr, self._map1, self._map2, cv2.INTER_LINEAR)
            out_msg = bgr_to_image_msg(rectified, msg.header, OPTICAL_FRAME)
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
        except Exception as exc:
            self.get_logger().error(f'Rectification failed: {exc}', throttle_duration_sec=2.0)


def main() -> None:
    node: Optional[RectifyImageNode] = None
    try:
        rclpy.init()
        node = RectifyImageNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except rclpy.executors.ExternalShutdownException:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
