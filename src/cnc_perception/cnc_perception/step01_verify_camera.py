#!/usr/bin/env python3
"""Step 01: Verify camera stream and camera_info."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class VerifyCameraNode(Node):
    def __init__(self) -> None:
        super().__init__('step01_verify_camera')
        self._image_count = 0
        self._camera_info_ok = False
        self.create_subscription(Image, '/image_rect_color', self._image_cb, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/camera_info', self._info_cb, qos_profile_sensor_data)
        self.create_timer(5.0, self._report)
        self.get_logger().info(
            'Step 01 running. Start usb_cam in another terminal, then watch this output.'
        )

    def _image_cb(self, msg: Image) -> None:
        self._image_count += 1
        if self._image_count == 1:
            self.get_logger().info(
                f'First image: {msg.width}x{msg.height} encoding={msg.encoding} frame={msg.header.frame_id}'
            )

    def _info_cb(self, msg: CameraInfo) -> None:
        if not self._camera_info_ok:
            self._camera_info_ok = True
            self.get_logger().info(
                f'camera_info OK: fx={msg.k[0]:.1f} fy={msg.k[4]:.1f} '
                f'cx={msg.k[2]:.1f} cy={msg.k[5]:.1f}'
            )

    def _report(self) -> None:
        if self._image_count == 0:
            self.get_logger().warn('No /image_rect_color yet. Run image_proc_pipeline.launch.py first.')
        else:
            self.get_logger().info(f'Received {self._image_count} images so far.')
        if not self._camera_info_ok:
            self.get_logger().warn('No /camera_info yet.')


def main() -> None:
    rclpy.init()
    node = VerifyCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
