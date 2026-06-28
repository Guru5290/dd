#!/usr/bin/env python3
"""Step 04: Diagnose workpiece detection and save debug images."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from cnc_perception.contour_detector import (
    detect_workpiece_corners,
    diagnose_contours,
    draw_detection_debug,
    draw_diagnostic_overlay,
)
from cnc_perception.image_utils import QOS_IMAGE_SUB, image_msg_to_bgr
from cnc_perception.workpiece_config import load_workpiece_config


class DebugWorkpieceDetectionNode(Node):
    def __init__(self) -> None:
        super().__init__('step04_debug_workpiece_detection')
        self.declare_parameter('workpiece_config_path', '')
        self.declare_parameter('output_dir', '/tmp/cnc_perception_debug')
        config_path = self._resolve_config()
        self._dimensions, self._detection, _ = load_workpiece_config(config_path)
        self._output_dir = Path(
            self.get_parameter('output_dir').get_parameter_value().string_value
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._frame = 0
        self.create_subscription(Image, '/image_rect_color', self._image_cb, QOS_IMAGE_SUB)
        self.get_logger().info(
            f'Step 04: workpiece {self._dimensions.width_m*1000:.0f}x'
            f'{self._dimensions.length_m*1000:.0f}x{self._dimensions.thickness_m*1000:.0f} mm. '
            f'Debug images -> {self._output_dir}'
        )

    def _resolve_config(self) -> str:
        value = self.get_parameter('workpiece_config_path').get_parameter_value().string_value
        if value:
            return value
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('cnc_perception'), 'config/workpiece_model.yaml')

    def _image_cb(self, msg: Image) -> None:
        self._frame += 1
        if self._frame % 10 != 0:
            return
        try:
            image = image_msg_to_bgr(msg)
        except (ValueError, RuntimeError) as exc:
            self.get_logger().warn(str(exc))
            return

        candidates, edges = diagnose_contours(image, self._dimensions, self._detection)
        detection = detect_workpiece_corners(image, self._dimensions, self._detection)

        if detection is None:
            self.get_logger().warn('No workpiece detected.')
            if candidates:
                self.get_logger().info('Top candidate rejections:')
                for index, candidate in enumerate(candidates[:3]):
                    self.get_logger().info(f'  #{index} score={candidate.score:.3f} {candidate.reason}')
        else:
            self.get_logger().info(f'Detected workpiece score={detection.score:.3f}')

        overlay = draw_diagnostic_overlay(image, candidates, edges)
        if detection is not None:
            overlay = draw_detection_debug(overlay, detection, 'DETECTED')

        out = self._output_dir / f'frame_{self._frame:06d}.jpg'
        cv2.imwrite(str(out), overlay)
        self.get_logger().info(f'Saved {out}')


def main() -> None:
    rclpy.init()
    node = DebugWorkpieceDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
