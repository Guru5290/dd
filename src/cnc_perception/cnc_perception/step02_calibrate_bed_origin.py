#!/usr/bin/env python3
"""Step 02: Calibrate cnc_bed_frame using ArUco marker at bed origin."""

from __future__ import annotations

import os

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from cnc_perception.bed_calibration import calibrate_bed_from_aruco, save_bed_calibration
from cnc_perception.bed_config import load_bed_config


class CalibrateBedOriginNode(Node):
    def __init__(self) -> None:
        super().__init__('step02_calibrate_bed_origin')
        self.declare_parameter('bed_config_path', '')
        self.declare_parameter('output_path', '')
        self._bridge = CvBridge()
        self._camera_info: CameraInfo | None = None
        self._done = False

        bed_path = self._resolve_path('bed_config_path', 'config/cnc_bed.yaml')
        self._output_path = self._resolve_path('output_path', 'config/bed_calibration.yaml')
        self._bed_config = load_bed_config(bed_path)

        self.create_subscription(CameraInfo, '/camera_info', self._info_cb, qos_profile_sensor_data)
        self.create_subscription(Image, '/image_raw', self._image_cb, qos_profile_sensor_data)

        self.get_logger().info(
            'Step 02: Place ArUco marker (DICT_4X4_50 id=0) at bed origin. '
            'Ensure only the origin marker is visible or matches configured id.'
        )

    def _resolve_path(self, param_name: str, default_rel: str) -> str:
        value = self.get_parameter(param_name).get_parameter_value().string_value
        if value:
            return value
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('cnc_perception'), default_rel)

    def _info_cb(self, msg: CameraInfo) -> None:
        self._camera_info = msg

    def _image_cb(self, msg: Image) -> None:
        if self._done or self._camera_info is None:
            return
        try:
            image = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
            calibration = calibrate_bed_from_aruco(
                image,
                list(self._camera_info.k),
                list(self._camera_info.d),
                self._camera_info.width,
                self._camera_info.height,
                self._bed_config,
            )
            save_bed_calibration(self._output_path, calibration)
            self._done = True
            self.get_logger().info(f'Bed calibration saved to {self._output_path}')
            self.get_logger().info(
                f'T_camera_bed translation (m): {calibration["translation_m"]}'
            )
            self.get_logger().info('Next: run step03_publish_bed_tf.py')
        except Exception as exc:
            self.get_logger().warn(f'Calibration attempt failed: {exc}')


def main() -> None:
    rclpy.init()
    node = CalibrateBedOriginNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
