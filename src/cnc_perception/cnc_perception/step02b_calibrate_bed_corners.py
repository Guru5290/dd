#!/usr/bin/env python3
"""Step 02b: Calibrate bed plane by clicking 4 bed corners (BL, BR, TR, TL)."""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from cnc_perception.bed_calibration import calibrate_bed_from_corners, save_bed_calibration
from cnc_perception.bed_config import load_bed_config

CLICKS: list[tuple[int, int]] = []
WINDOW = 'step02b - click BL, BR, TR, TL then press ENTER'


def _mouse(event: int, x: int, y: int, _flags, _param) -> None:
    if event == cv2.EVENT_LBUTTONDOWN and len(CLICKS) < 4:
        CLICKS.append((x, y))


class CalibrateBedCornersNode(Node):
    def __init__(self) -> None:
        super().__init__('step02b_calibrate_bed_corners')
        self.declare_parameter('bed_config_path', '')
        self.declare_parameter('output_path', '')
        self._bridge = CvBridge()
        self._image = None
        self._output_path = self._resolve_path('output_path', 'config/bed_calibration.yaml')
        self._bed_config = load_bed_config(self._resolve_path('bed_config_path', 'config/cnc_bed.yaml'))
        self.create_subscription(Image, '/image_raw', self._image_cb, qos_profile_sensor_data)
        self.create_timer(0.1, self._ui_tick)
        self.get_logger().info(
            'Step 02b: A window will open. Click bed corners: bottom-left, bottom-right, '
            'top-right, top-left. Press ENTER to save, ESC to cancel.'
        )

    def _resolve_path(self, param_name: str, default_rel: str) -> str:
        value = self.get_parameter(param_name).get_parameter_value().string_value
        if value:
            return value
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('cnc_perception'), default_rel)

    def _image_cb(self, msg: Image) -> None:
        self._image = self._bridge.imgmsg_to_cv2(msg, 'bgr8')

    def _ui_tick(self) -> None:
        if self._image is None:
            return
        display = self._image.copy()
        for index, (x, y) in enumerate(CLICKS):
            cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(
                display,
                str(index),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            self.get_logger().info('Cancelled.')
            rclpy.shutdown()
            sys.exit(0)
        if key == 13 and len(CLICKS) == 4:
            corners = np.array(CLICKS, dtype=np.float64)
            data = calibrate_bed_from_corners(corners, self._bed_config.bed)
            save_bed_calibration(self._output_path, data)
            self.get_logger().info(f'Saved homography calibration to {self._output_path}')
            cv2.destroyAllWindows()
            rclpy.shutdown()
            sys.exit(0)


def main() -> None:
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, _mouse)
    rclpy.init()
    node = CalibrateBedCornersNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
