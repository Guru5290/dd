#!/usr/bin/env python3
"""Markerless 6D pose estimation node for plain CNC workpieces."""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import MarkerArray

from cnc_perception.contour_detector import (
    detect_workpiece_corners,
    draw_detection_debug,
)
from cnc_perception.pose_solver import (
    PoseEstimate,
    camera_info_to_matrices,
    pose_estimate_to_geometry_pose,
    smooth_pose,
    solve_workpiece_pose,
)
from cnc_perception.visualization import make_workpiece_markers
from cnc_perception.workpiece_config import load_workpiece_config


class PoseEstimatorNode(Node):
    """Estimate workpiece pose from overhead camera images without fiducials."""

    def __init__(self) -> None:
        super().__init__('pose_estimator_node')

        self.declare_parameter('workpiece_config_path', '')
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('camera_info_topic', '/camera_info')
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('workpiece_frame', 'workpiece_frame')
        self.declare_parameter('max_reprojection_error_px', 8.0)

        config_path = self._resolve_config_path(
            self.get_parameter('workpiece_config_path').get_parameter_value().string_value
        )
        try:
            self._dimensions, self._detection_config = load_workpiece_config(config_path)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            self.get_logger().fatal(f'Failed to load workpiece config: {exc}')
            raise

        self._object_corners = self._dimensions.object_corners_centered()
        self._bridge = CvBridge()
        self._tf_broadcaster = TransformBroadcaster(self)

        self._camera_matrix: Optional[np.ndarray] = None
        self._distortion: Optional[np.ndarray] = None
        self._camera_info_received = False
        self._latest_pose: Optional[PoseEstimate] = None
        self._consecutive_failures = 0

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value

        self._image_sub = self.create_subscription(
            Image,
            image_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )
        self._camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self._camera_info_callback,
            qos_profile_sensor_data,
        )
        self._pose_pub = self.create_publisher(PoseStamped, '/workpiece/actual_pose', 10)
        self._marker_pub = self.create_publisher(MarkerArray, '/workpiece/markers', 10)
        if self._detection_config.publish_debug_image:
            self._debug_pub = self.create_publisher(Image, '/workpiece/debug_image', 10)
        else:
            self._debug_pub = None

        self.get_logger().info(
            'Pose estimator ready. Workpiece: '
            f'{self._dimensions.width_m:.3f} x {self._dimensions.length_m:.3f} x '
            f'{self._dimensions.thickness_m:.3f} m'
        )

    def _resolve_config_path(self, configured_path: str) -> str:
        if configured_path:
            return configured_path
        from ament_index_python.packages import get_package_share_directory

        share_dir = get_package_share_directory('cnc_perception')
        return os.path.join(share_dir, 'config', 'workpiece_model.yaml')

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        try:
            camera_matrix, distortion = camera_info_to_matrices(
                list(msg.k),
                list(msg.d),
                msg.width,
                msg.height,
            )
        except ValueError as exc:
            if self._consecutive_failures % 30 == 0:
                self.get_logger().warn(f'Invalid camera_info received: {exc}')
            return

        self._camera_matrix = camera_matrix
        self._distortion = distortion
        if not self._camera_info_received:
            self.get_logger().info(
                f'Camera calibration loaded ({msg.width}x{msg.height}, frame={msg.header.frame_id})'
            )
            self._camera_info_received = True

    def _image_callback(self, msg: Image) -> None:
        if self._camera_matrix is None or self._distortion is None:
            if self._consecutive_failures % 30 == 0:
                self.get_logger().warn(
                    'Waiting for valid /camera_info before pose estimation.'
                )
            self._consecutive_failures += 1
            return

        try:
            image_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as exc:
            if self._consecutive_failures % 30 == 0:
                self.get_logger().warn(f'cv_bridge conversion failed: {exc}')
            self._consecutive_failures += 1
            return

        try:
            detection = detect_workpiece_corners(
                image_bgr,
                self._dimensions,
                self._detection_config,
            )
        except cv2.error as exc:
            if self._consecutive_failures % 30 == 0:
                self.get_logger().warn(f'OpenCV detection error: {exc}')
            self._consecutive_failures += 1
            return

        if detection is None:
            if self._consecutive_failures % 30 == 0:
                self.get_logger().warn(
                    'Workpiece not detected. Check lighting, contrast, and workpiece_model.yaml.'
                )
            self._consecutive_failures += 1
            return

        pose = solve_workpiece_pose(
            detection.corners,
            self._object_corners,
            self._camera_matrix,
            self._distortion,
            use_ippe_for_planar=self._detection_config.use_ippe_for_planar,
        )
        if pose is None:
            if self._consecutive_failures % 30 == 0:
                self.get_logger().warn('solvePnP failed for detected contour.')
            self._consecutive_failures += 1
            return

        max_error = self.get_parameter('max_reprojection_error_px').get_parameter_value().double_value
        if pose.reprojection_error > max_error:
            if self._consecutive_failures % 30 == 0:
                self.get_logger().warn(
                    f'Reprojection error too high: {pose.reprojection_error:.2f}px '
                    f'(max {max_error:.2f}px)'
                )
            self._consecutive_failures += 1
            return

        pose = smooth_pose(
            self._latest_pose,
            pose,
            self._detection_config.pose_smoothing_alpha,
        )
        self._latest_pose = pose
        self._consecutive_failures = 0

        stamp = msg.header.stamp
        camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        workpiece_frame = self.get_parameter('workpiece_frame').get_parameter_value().string_value
        source_frame = msg.header.frame_id or camera_frame

        self._publish_pose(stamp, source_frame, pose)
        self._publish_tf(stamp, source_frame, workpiece_frame, pose)
        self._publish_markers(stamp, workpiece_frame)

        if self._debug_pub is not None:
            debug_image = draw_detection_debug(image_bgr, detection)
            try:
                debug_msg = self._bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
                debug_msg.header = msg.header
                self._debug_pub.publish(debug_msg)
            except CvBridgeError as exc:
                self.get_logger().debug(f'Failed to publish debug image: {exc}')

    def _publish_pose(self, stamp, frame_id: str, pose: PoseEstimate) -> None:
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.pose = pose_estimate_to_geometry_pose(pose)
        self._pose_pub.publish(msg)

    def _publish_tf(
        self,
        stamp,
        parent_frame: str,
        child_frame: str,
        pose: PoseEstimate,
    ) -> None:
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame
        transform.transform.translation.x = float(pose.translation[0])
        transform.transform.translation.y = float(pose.translation[1])
        transform.transform.translation.z = float(pose.translation[2])

        from cnc_perception.pose_solver import rotation_matrix_to_quaternion

        quat = rotation_matrix_to_quaternion(pose.rotation_matrix)
        transform.transform.rotation = quat
        self._tf_broadcaster.sendTransform(transform)

    def _publish_markers(self, stamp, frame_id: str) -> None:
        markers = make_workpiece_markers(stamp, frame_id, self._dimensions)
        self._marker_pub.publish(markers)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = PoseEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
