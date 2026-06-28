#!/usr/bin/env python3
"""Step 06: Compare workpiece pose to target — CORRECT / NOT CORRECT output."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from cnc_perception.bed_config import load_bed_config
from cnc_perception.bed_visualization import make_placement_status_marker
from cnc_perception.placement_checker import check_placement
from cnc_perception.transform_utils import pose_to_matrix
from cnc_perception.workpiece_config import load_workpiece_config


class CheckPlacementNode(Node):
    def __init__(self) -> None:
        super().__init__('step06_check_placement')
        self.declare_parameter('bed_config_path', '')
        self.declare_parameter('workpiece_config_path', '')
        self._bed_config = load_bed_config(self._share_path('bed_config_path', 'config/cnc_bed.yaml'))
        self._dimensions, _, _ = load_workpiece_config(
            self._share_path('workpiece_config_path', 'config/workpiece_model.yaml')
        )
        self._status_pub = self.create_publisher(String, '/workpiece/placement_status', 10)
        self._marker_pub = self.create_publisher(MarkerArray, '/workpiece/placement_markers', 10)
        self.create_subscription(
            PoseStamped, '/workpiece/pose_in_bed_frame', self._pose_cb, 10
        )
        self.get_logger().info('Step 06: Run step05 first. Listening to /workpiece/pose_in_bed_frame.')

    def _share_path(self, param: str, rel: str) -> str:
        value = self.get_parameter(param).get_parameter_value().string_value
        if value:
            return value
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('cnc_perception'), rel)

    def _pose_cb(self, msg: PoseStamped) -> None:
        if msg.header.frame_id != 'cnc_bed_frame':
            self.get_logger().warn(
                f'Expected cnc_bed_frame, got {msg.header.frame_id}', throttle_duration_sec=5.0
            )
        t_bed_workpiece = pose_to_matrix(msg.pose)
        result = check_placement(
            t_bed_workpiece,
            self._dimensions.thickness_m,
            self._bed_config.target,
        )

        status = String()
        status.data = result.message
        self._status_pub.publish(status)

        level = 'info' if result.ok else 'warn'
        getattr(self.get_logger(), level)(result.message)

        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()
        markers.markers.append(
            make_placement_status_marker(
                stamp,
                'cnc_bed_frame',
                'CORRECT POSITION' if result.ok else 'NOT CORRECT POSITION',
                result.ok,
            )
        )
        detail = Marker()
        detail.header.stamp = stamp
        detail.header.frame_id = 'cnc_bed_frame'
        detail.ns = 'status'
        detail.id = 1
        detail.type = Marker.TEXT_VIEW_FACING
        detail.action = Marker.ADD
        detail.pose.position.x = 0.0
        detail.pose.position.y = 0.05
        detail.pose.position.z = 0.08
        detail.scale.z = 0.025
        detail.color.r = 1.0
        detail.color.g = 1.0
        detail.color.b = 1.0
        detail.color.a = 1.0
        detail.text = (
            f'X={result.x_mm:.1f} Y={result.y_mm:.1f} Z={result.z_mm:.1f} mm '
            f'yaw={result.yaw_deg:.1f} deg | '
            f'dX={result.dx_mm:+.1f} dY={result.dy_mm:+.1f} dZ={result.dz_mm:+.1f} mm '
            f'dYaw={result.dyaw_deg:+.1f} deg'
        )
        markers.markers.append(detail)
        self._marker_pub.publish(markers)


def main() -> None:
    rclpy.init()
    node = CheckPlacementNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
