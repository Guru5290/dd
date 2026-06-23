#!/usr/bin/env python3
"""Step 03: Publish static TF camera_link -> cnc_bed_frame from saved calibration."""

from __future__ import annotations

import os

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster

from cnc_perception.bed_calibration import load_bed_calibration
from cnc_perception.camera_frames import LINK_FRAME, OPTICAL_FRAME
from cnc_perception.camera_frames import transform_link_to_optical
from cnc_perception.transform_utils import matrix_to_translation_quaternion


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

        transforms: list[TransformStamped] = []
        stamp = self.get_clock().now().to_msg()

        bed_tf = TransformStamped()
        bed_tf.header.stamp = stamp
        bed_tf.header.frame_id = calibration.get('camera_frame', LINK_FRAME)
        bed_tf.child_frame_id = calibration.get('bed_frame', 'cnc_bed_frame')
        translation = calibration.get('translation_m', [0.0, 0.0, 0.0])
        rotation = calibration.get('rotation_xyzw', [0.0, 0.0, 0.0, 1.0])
        bed_tf.transform.translation.x = float(translation[0])
        bed_tf.transform.translation.y = float(translation[1])
        bed_tf.transform.translation.z = float(translation[2])
        bed_tf.transform.rotation.x = float(rotation[0])
        bed_tf.transform.rotation.y = float(rotation[1])
        bed_tf.transform.rotation.z = float(rotation[2])
        bed_tf.transform.rotation.w = float(rotation[3])
        transforms.append(bed_tf)

        optical_tf = TransformStamped()
        optical_tf.header.stamp = stamp
        optical_tf.header.frame_id = LINK_FRAME
        optical_tf.child_frame_id = OPTICAL_FRAME
        opt_translation, opt_quat = matrix_to_translation_quaternion(transform_link_to_optical())
        optical_tf.transform.translation.x = float(opt_translation[0])
        optical_tf.transform.translation.y = float(opt_translation[1])
        optical_tf.transform.translation.z = float(opt_translation[2])
        optical_tf.transform.rotation.x = float(opt_quat[0])
        optical_tf.transform.rotation.y = float(opt_quat[1])
        optical_tf.transform.rotation.z = float(opt_quat[2])
        optical_tf.transform.rotation.w = float(opt_quat[3])
        transforms.append(optical_tf)

        self._broadcaster = StaticTransformBroadcaster(self)
        self._broadcaster.sendTransform(transforms)
        self.get_logger().info(
            f'Publishing static TF {bed_tf.header.frame_id} -> {bed_tf.child_frame_id}'
        )
        self.get_logger().info(
            f'Publishing static TF {optical_tf.header.frame_id} -> {optical_tf.child_frame_id}'
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
