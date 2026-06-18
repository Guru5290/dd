#!/usr/bin/env python3
"""Step 03: Publish static TF camera_link -> cnc_bed_frame from saved calibration."""

from __future__ import annotations

import os

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster

from cnc_perception.bed_calibration import load_bed_calibration


class PublishBedTfNode(Node):
    def __init__(self) -> None:
        super().__init__('step03_publish_bed_tf')
        self.declare_parameter('calibration_path', '')
        path = self._resolve_path()
        calibration = load_bed_calibration(path)
        if calibration.get('method') == 'manual_corners':
            self.get_logger().warn(
                'Homography-only calibration: full 3D TF is approximate. '
                'Prefer ArUco step02 for accurate 6D bed frame.'
            )

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = calibration.get('camera_frame', 'camera_link')
        transform.child_frame_id = calibration.get('bed_frame', 'cnc_bed_frame')
        translation = calibration.get('translation_m', [0.0, 0.0, 0.0])
        rotation = calibration.get('rotation_xyzw', [0.0, 0.0, 0.0, 1.0])
        transform.transform.translation.x = float(translation[0])
        transform.transform.translation.y = float(translation[1])
        transform.transform.translation.z = float(translation[2])
        transform.transform.rotation.x = float(rotation[0])
        transform.transform.rotation.y = float(rotation[1])
        transform.transform.rotation.z = float(rotation[2])
        transform.transform.rotation.w = float(rotation[3])

        self._broadcaster = StaticTransformBroadcaster(self)
        self._broadcaster.sendTransform(transform)
        self.get_logger().info(
            f'Publishing static TF {transform.header.frame_id} -> {transform.child_frame_id}'
        )
        self.get_logger().info('Keep this node running. Open RViz with Fixed Frame=cnc_bed_frame.')

    def _resolve_path(self) -> str:
        value = self.get_parameter('calibration_path').get_parameter_value().string_value
        if value:
            return value
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('cnc_perception'), 'config/bed_calibration.yaml')


def main() -> None:
    rclpy.init()
    node = PublishBedTfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
