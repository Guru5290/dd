#!/usr/bin/env python3
"""Step 05: Workpiece 6D pose in cnc_bed_frame + RViz bed/workpiece markers."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from visualization_msgs.msg import MarkerArray

from cnc_perception.bed_config import load_bed_config
from cnc_perception.bed_visualization import make_bed_markers
from cnc_perception.contour_detector import detect_workpiece_corners, draw_detection_debug
from cnc_perception.pose_solver import (
    camera_info_to_matrices,
    pose_estimate_to_geometry_pose,
    solve_workpiece_pose,
)
from cnc_perception.transform_utils import matrix_to_pose, transform_to_matrix
from cnc_perception.visualization import make_workpiece_markers
from cnc_perception.workpiece_config import load_workpiece_config


class WorkpiecePoseBedFrameNode(Node):
    def __init__(self) -> None:
        super().__init__('step05_workpiece_pose_bed_frame')
        self.declare_parameter('workpiece_config_path', '')
        self.declare_parameter('bed_config_path', '')
        self._dimensions, self._detection = load_workpiece_config(self._share_path('workpiece_config_path', 'config/workpiece_model.yaml'))
        self._bed_config = load_bed_config(self._share_path('bed_config_path', 'config/cnc_bed.yaml'))
        self._bridge = CvBridge()
        self._camera_matrix: Optional[np.ndarray] = None
        self._distortion: Optional[np.ndarray] = None
        self._tf_broadcaster = TransformBroadcaster(self)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._pose_cam_pub = self.create_publisher(PoseStamped, '/workpiece/actual_pose', 10)
        self._pose_bed_pub = self.create_publisher(PoseStamped, '/workpiece/pose_in_bed_frame', 10)
        self._status_pub = self.create_publisher(String, '/workpiece/bed_coordinates', 10)
        self._marker_pub = self.create_publisher(MarkerArray, '/workpiece/markers', 10)
        self._bed_marker_pub = self.create_publisher(MarkerArray, '/cnc_bed/markers', 10)
        self._debug_pub = self.create_publisher(Image, '/workpiece/debug_image', 10)

        self.create_subscription(CameraInfo, '/camera_info', self._info_cb, qos_profile_sensor_data)
        self.create_subscription(Image, '/image_raw', self._image_cb, qos_profile_sensor_data)
        self.create_timer(1.0, self._publish_bed_markers)
        self.get_logger().info('Step 05: Requires step03_publish_bed_tf running (cnc_bed_frame in TF tree).')

    def _share_path(self, param: str, rel: str) -> str:
        value = self.get_parameter(param).get_parameter_value().string_value
        if value:
            return value
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('cnc_perception'), rel)

    def _info_cb(self, msg: CameraInfo) -> None:
        self._camera_matrix, self._distortion = camera_info_to_matrices(
            list(msg.k), list(msg.d), msg.width, msg.height
        )

    def _publish_bed_markers(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self._bed_marker_pub.publish(make_bed_markers(stamp, 'cnc_bed_frame', self._bed_config))

    def _image_cb(self, msg: Image) -> None:
        if self._camera_matrix is None:
            return
        try:
            image = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError:
            return

        detection = detect_workpiece_corners(image, self._dimensions, self._detection)
        if detection is None:
            self.get_logger().warn('Workpiece not detected', throttle_duration_sec=2.0)
            return

        pose = solve_workpiece_pose(
            detection.corners,
            self._dimensions.object_corners_centered(),
            self._camera_matrix,
            self._distortion,
            use_ippe_for_planar=True,
        )
        if pose is None:
            return

        stamp = msg.header.stamp
        camera_frame = msg.header.frame_id or 'camera_link'

        t_camera_workpiece = transform_to_matrix(pose.translation, self._rotation_to_quat(pose.rotation_matrix))

        pose_cam = PoseStamped()
        pose_cam.header.stamp = stamp
        pose_cam.header.frame_id = camera_frame
        pose_cam.pose = pose_estimate_to_geometry_pose(pose)
        self._pose_cam_pub.publish(pose_cam)

        tf_cam_wp = TransformStamped()
        tf_cam_wp.header.stamp = stamp
        tf_cam_wp.header.frame_id = camera_frame
        tf_cam_wp.child_frame_id = 'workpiece_frame'
        tf_cam_wp.transform.translation.x = float(pose.translation[0])
        tf_cam_wp.transform.translation.y = float(pose.translation[1])
        tf_cam_wp.transform.translation.z = float(pose.translation[2])
        from cnc_perception.pose_solver import rotation_matrix_to_quaternion
        tf_cam_wp.transform.rotation = rotation_matrix_to_quaternion(pose.rotation_matrix)
        self._tf_broadcaster.sendTransform(tf_cam_wp)

        # Lookup T_camera_bed via TF would be ideal; here we assume step03 published cnc_bed_frame child of camera.
        # Publish workpiece relative to bed using tf2 would need listener — compute if bed TF is camera<-bed:
        # T_bed_workpiece = inv(T_camera_bed) * T_camera_workpiece
        # For now publish pose in camera frame + bed coords message from listener alternative:
        # Use static composition: user must have bed TF. We use tf2 buffer.
        try:
            tf_bed_cam_msg = self._tf_buffer.lookup_transform(
                'cnc_bed_frame', camera_frame, rclpy.time.Time()
            )
            t_bed_cam = transform_to_matrix(
                [
                    tf_bed_cam_msg.transform.translation.x,
                    tf_bed_cam_msg.transform.translation.y,
                    tf_bed_cam_msg.transform.translation.z,
                ],
                [
                    tf_bed_cam_msg.transform.rotation.x,
                    tf_bed_cam_msg.transform.rotation.y,
                    tf_bed_cam_msg.transform.rotation.z,
                    tf_bed_cam_msg.transform.rotation.w,
                ],
            )
            t_bed_workpiece = t_bed_cam @ t_camera_workpiece

            pose_bed = PoseStamped()
            pose_bed.header.stamp = stamp
            pose_bed.header.frame_id = 'cnc_bed_frame'
            pose_bed.pose = matrix_to_pose(t_bed_workpiece)
            self._pose_bed_pub.publish(pose_bed)

            x_mm = t_bed_workpiece[0, 3] * 1000.0
            y_mm = t_bed_workpiece[1, 3] * 1000.0
            z_mm = t_bed_workpiece[2, 3] * 1000.0
            status = String()
            status.data = (
                f'Workpiece center in bed frame: X={x_mm:.1f} mm Y={y_mm:.1f} mm Z={z_mm:.1f} mm '
                f'(Z≈{self._dimensions.thickness_m*1000:.1f} mm when flat on bed)'
            )
            self._status_pub.publish(status)
            self.get_logger().info(status.data, throttle_duration_sec=2.0)
        except Exception as exc:
            self.get_logger().warn(f'Waiting for cnc_bed_frame TF: {exc}', throttle_duration_sec=3.0)

        markers = make_workpiece_markers(stamp, 'workpiece_frame', self._dimensions)
        self._marker_pub.publish(markers)

        debug = draw_detection_debug(image, detection, 'OK')
        try:
            debug_msg = self._bridge.cv2_to_imgmsg(debug, 'bgr8')
            debug_msg.header = msg.header
            self._debug_pub.publish(debug_msg)
        except CvBridgeError:
            pass

    @staticmethod
    def _rotation_to_quat(rotation_matrix: np.ndarray) -> list[float]:
        from cnc_perception.pose_solver import rotation_matrix_to_quaternion
        q = rotation_matrix_to_quaternion(rotation_matrix)
        return [q.x, q.y, q.z, q.w]


def main() -> None:
    rclpy.init()
    node = WorkpiecePoseBedFrameNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
