#!/usr/bin/env python3
"""Step 05: Workpiece 6D pose in cnc_bed_frame + RViz bed/workpiece markers."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import rclpy
from rclpy import duration
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
from cnc_perception.camera_frames import LINK_FRAME, OPTICAL_FRAME, transform_optical_to_link
from cnc_perception.contour_detector import detect_workpiece_corners, draw_detection_debug
from cnc_perception.pose_solver import (
    PoseEstimate,
    camera_info_to_matrices,
    pose_estimate_to_geometry_pose,
    rotation_matrix_to_quaternion,
    smooth_pose,
    solve_workpiece_pose,
)
from cnc_perception.transform_utils import (
    is_pose_center_on_bed,
    matrix_to_pose,
    surface_normal_tilt_deg,
    transform_to_matrix,
)
from cnc_perception.visualization import (
    make_delete_workpiece_markers,
    make_workpiece_markers_at_pose,
)
from cnc_perception.workpiece_config import load_workpiece_config


class WorkpiecePoseBedFrameNode(Node):
    def __init__(self) -> None:
        super().__init__('step05_workpiece_pose_bed_frame')
        self.declare_parameter('workpiece_config_path', '')
        self.declare_parameter('bed_config_path', '')
        self.declare_parameter('image_topic', '/image_rect_color')
        self.declare_parameter('camera_info_topic', '/camera_info')
        self.declare_parameter('max_reprojection_error_px', 6.0)
        self.declare_parameter('bed_margin_m', 0.012)
        self.declare_parameter('max_surface_tilt_deg', 25.0)
        self.declare_parameter('use_rectified_camera_info', True)

        self._dimensions, self._detection = load_workpiece_config(
            self._share_path('workpiece_config_path', 'config/workpiece_model.yaml')
        )
        self._bed_config = load_bed_config(
            self._share_path('bed_config_path', 'config/cnc_bed.yaml')
        )
        self._object_corners = self._dimensions.object_corners_centered()
        self._bridge = CvBridge()
        self._camera_matrix: Optional[np.ndarray] = None
        self._distortion: Optional[np.ndarray] = None
        self._tf_broadcaster = TransformBroadcaster(self)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._latest_pose: Optional[PoseEstimate] = None
        self._consecutive_failures = 0
        self._detection_active = False

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value

        self._pose_cam_pub = self.create_publisher(PoseStamped, '/workpiece/actual_pose', 10)
        self._pose_bed_pub = self.create_publisher(PoseStamped, '/workpiece/pose_in_bed_frame', 10)
        self._status_pub = self.create_publisher(String, '/workpiece/bed_coordinates', 10)
        self._marker_pub = self.create_publisher(MarkerArray, '/workpiece/markers', 10)
        self._bed_marker_pub = self.create_publisher(MarkerArray, '/cnc_bed/markers', 10)
        self._debug_pub = self.create_publisher(Image, '/workpiece/debug_image', 10)

        self.create_subscription(
            CameraInfo, camera_info_topic, self._info_cb, qos_profile_sensor_data
        )
        self.create_subscription(Image, image_topic, self._image_cb, qos_profile_sensor_data)
        self.create_timer(1.0, self._publish_bed_markers)
        self.get_logger().info(
            f'Step 05: Using {image_topic} (color) + {camera_info_topic} with rectified intrinsics.'
        )

    def _share_path(self, param: str, rel: str) -> str:
        value = self.get_parameter(param).get_parameter_value().string_value
        if value:
            return value
        from ament_index_python.packages import get_package_share_directory

        return os.path.join(get_package_share_directory('cnc_perception'), rel)

    def _info_cb(self, msg: CameraInfo) -> None:
        rectified = self.get_parameter('use_rectified_camera_info').get_parameter_value().bool_value
        self._camera_matrix, self._distortion = camera_info_to_matrices(
            list(msg.k),
            list(msg.d),
            msg.width,
            msg.height,
            rectified=rectified,
        )

    def _publish_bed_markers(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self._bed_marker_pub.publish(make_bed_markers(stamp, 'cnc_bed_frame', self._bed_config))

    def _lookup_bed_from_link(self, stamp) -> Optional[np.ndarray]:
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                'cnc_bed_frame',
                LINK_FRAME,
                stamp,
                timeout=duration.Duration(seconds=0.05),
            )
        except Exception:
            return None
        return transform_to_matrix(
            [
                tf_msg.transform.translation.x,
                tf_msg.transform.translation.y,
                tf_msg.transform.translation.z,
            ],
            [
                tf_msg.transform.rotation.x,
                tf_msg.transform.rotation.y,
                tf_msg.transform.rotation.z,
                tf_msg.transform.rotation.w,
            ],
        )

    def _handle_detection_lost(self, stamp) -> None:
        self._consecutive_failures += 1
        if self._detection_active:
            self._marker_pub.publish(make_delete_workpiece_markers(stamp, 'cnc_bed_frame'))
            self._detection_active = False
        if self._consecutive_failures >= 3:
            self._latest_pose = None
        if self._consecutive_failures % 30 == 1:
            self.get_logger().warn(
                'Workpiece not detected (or rejected). Cube hidden until redetected.',
                throttle_duration_sec=2.0,
            )

    def _image_cb(self, msg: Image) -> None:
        if self._camera_matrix is None:
            self.get_logger().warn('Waiting for CameraInfo...', throttle_duration_sec=4.0)
            return

        try:
            image = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError:
            self._handle_detection_lost(msg.header.stamp)
            return

        detection = detect_workpiece_corners(image, self._dimensions, self._detection)
        if detection is None:
            self._handle_detection_lost(msg.header.stamp)
            return

        pose = solve_workpiece_pose(
            detection.corners,
            self._object_corners,
            self._camera_matrix,
            self._distortion,
            use_ippe_for_planar=self._detection.use_ippe_for_planar,
        )
        if pose is None:
            self._handle_detection_lost(msg.header.stamp)
            return

        max_error = self.get_parameter('max_reprojection_error_px').get_parameter_value().double_value
        if pose.reprojection_error > max_error:
            self.get_logger().warn(
                f'Reprojection error {pose.reprojection_error:.1f}px > {max_error:.1f}px',
                throttle_duration_sec=2.0,
            )
            self._handle_detection_lost(msg.header.stamp)
            return

        pose = smooth_pose(self._latest_pose, pose, self._detection.pose_smoothing_alpha)
        self._latest_pose = pose
        self._consecutive_failures = 0

        stamp = msg.header.stamp
        optical_frame = msg.header.frame_id or OPTICAL_FRAME

        t_optical_workpiece = transform_to_matrix(pose.translation, self._rotation_to_quat(pose.rotation_matrix))
        t_link_workpiece = transform_optical_to_link() @ t_optical_workpiece

        pose_cam = PoseStamped()
        pose_cam.header.stamp = stamp
        pose_cam.header.frame_id = optical_frame
        pose_cam.pose = pose_estimate_to_geometry_pose(pose)
        self._pose_cam_pub.publish(pose_cam)

        tf_optical_wp = TransformStamped()
        tf_optical_wp.header.stamp = stamp
        tf_optical_wp.header.frame_id = optical_frame
        tf_optical_wp.child_frame_id = 'workpiece_frame'
        tf_optical_wp.transform.translation.x = float(pose.translation[0])
        tf_optical_wp.transform.translation.y = float(pose.translation[1])
        tf_optical_wp.transform.translation.z = float(pose.translation[2])
        tf_optical_wp.transform.rotation = rotation_matrix_to_quaternion(pose.rotation_matrix)

        t_bed_link = self._lookup_bed_from_link(stamp)
        if t_bed_link is None:
            self.get_logger().warn(
                'Waiting for TF cnc_bed_frame <- camera_link. Run step03_publish_bed_tf.',
                throttle_duration_sec=3.0,
            )
            self._tf_broadcaster.sendTransform(tf_optical_wp)
            return

        t_bed_workpiece = t_bed_link @ t_link_workpiece
        bed_margin = self.get_parameter('bed_margin_m').get_parameter_value().double_value
        max_tilt = self.get_parameter('max_surface_tilt_deg').get_parameter_value().double_value

        if not is_pose_center_on_bed(
            t_bed_workpiece,
            self._bed_config.bed.length_m,
            self._bed_config.bed.width_m,
            margin_m=bed_margin,
        ):
            self.get_logger().warn(
                'Workpiece center outside CNC bed — not publishing bed pose.',
                throttle_duration_sec=2.0,
            )
            self._handle_detection_lost(stamp)
            return

        tilt_deg = surface_normal_tilt_deg(t_bed_workpiece[:3, :3])
        if tilt_deg > max_tilt:
            self.get_logger().warn(
                f'Workpiece appears tilted ({tilt_deg:.1f} deg) — check calibration or shadow.',
                throttle_duration_sec=2.0,
            )
            self._handle_detection_lost(stamp)
            return

        pose_bed = PoseStamped()
        pose_bed.header.stamp = stamp
        pose_bed.header.frame_id = 'cnc_bed_frame'
        pose_bed.pose = matrix_to_pose(t_bed_workpiece)
        self._pose_bed_pub.publish(pose_bed)

        tf_bed_wp = TransformStamped()
        tf_bed_wp.header.stamp = stamp
        tf_bed_wp.header.frame_id = 'cnc_bed_frame'
        tf_bed_wp.child_frame_id = 'workpiece_frame'
        tf_bed_wp.transform.translation.x = float(t_bed_workpiece[0, 3])
        tf_bed_wp.transform.translation.y = float(t_bed_workpiece[1, 3])
        tf_bed_wp.transform.translation.z = float(t_bed_workpiece[2, 3])
        bed_quat = rotation_matrix_to_quaternion(t_bed_workpiece[:3, :3])
        tf_bed_wp.transform.rotation = bed_quat
        self._tf_broadcaster.sendTransform([tf_bed_wp, tf_optical_wp])

        x_mm = float(t_bed_workpiece[0, 3]) * 1000.0
        y_mm = float(t_bed_workpiece[1, 3]) * 1000.0
        z_mm = float(t_bed_workpiece[2, 3]) * 1000.0
        status = String()
        status.data = (
            f'Bed frame: X={x_mm:.1f} mm Y={y_mm:.1f} mm Z_top={z_mm:.1f} mm '
            f'(expect Z≈{self._dimensions.thickness_m * 1000:.1f} mm when flat) '
            f'reproj={pose.reprojection_error:.1f}px tilt={tilt_deg:.1f}deg'
        )
        self._status_pub.publish(status)
        self.get_logger().info(status.data, throttle_duration_sec=2.0)

        self._marker_pub.publish(
            make_workpiece_markers_at_pose(stamp, 'cnc_bed_frame', pose_bed.pose, self._dimensions)
        )
        self._detection_active = True

        debug = draw_detection_debug(image, detection, 'OK')
        try:
            debug_msg = self._bridge.cv2_to_imgmsg(debug, 'bgr8')
            debug_msg.header = msg.header
            self._debug_pub.publish(debug_msg)
        except CvBridgeError:
            pass

    @staticmethod
    def _rotation_to_quat(rotation_matrix: np.ndarray) -> list[float]:
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
