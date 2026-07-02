#!/usr/bin/env python3
"""Step 05: Workpiece 6D pose in cnc_bed_frame + RViz bed/workpiece markers."""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy import duration
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from visualization_msgs.msg import MarkerArray

from cnc_perception.bed_config import load_bed_config
from cnc_perception.bed_visualization import make_bed_markers, make_target_placement_markers
from cnc_perception.camera_frames import LINK_FRAME, OPTICAL_FRAME, transform_optical_to_link
from cnc_perception.contour_detector import (
    detect_workpiece_corners,
    diagnose_contours,
    draw_detection_debug,
)
from cnc_perception.pose_ekf import BedPoseEkf, RecoveryTracker, check_auto_recovery
from cnc_perception.pose_solver import (
    PoseEstimate,
    camera_info_to_matrices,
    pose_estimate_to_geometry_pose,
    rotation_matrix_to_quaternion,
    smooth_pose,
    solve_workpiece_pose_in_bed_frame,
)
from cnc_perception.transform_utils import (
    apply_square_yaw_fold_to_transform,
    apply_coordinate_reporting_offset,
    flatten_transform_to_bed_plane,
    is_pose_center_on_bed,
    matrix_to_pose,
    smooth_flat_bed_transform,
    surface_normal_tilt_deg,
    transform_to_matrix,
    yaw_from_matrix,
)
from cnc_perception.visualization import (
    make_delete_workpiece_markers,
    make_pose_status_marker,
    make_workpiece_markers_at_pose,
)
from cnc_perception.image_utils import QOS_IMAGE_SUB, bgr_to_image_msg, image_msg_to_bgr
from cnc_perception.workpiece_marker_pose import detect_workpiece_marker_pose, draw_workpiece_marker_debug
from cnc_perception.workpiece_config import load_workpiece_config


class WorkpiecePoseBedFrameNode(Node):
    def __init__(self) -> None:
        super().__init__('step05_workpiece_pose_bed_frame')
        self.declare_parameter('workpiece_config_path', '')
        self.declare_parameter('bed_config_path', '')
        self.declare_parameter('image_topic', '/image_rect_color')
        self.declare_parameter('camera_info_topic', '/camera_info')
        self.declare_parameter('max_reprojection_error_px', 10.0)
        self.declare_parameter('bed_margin_m', 0.005)
        self.declare_parameter('max_surface_tilt_deg', 35.0)
        self.declare_parameter('publish_pose_only_when_flat', True)
        self.declare_parameter('use_rectified_camera_info', True)

        self._dimensions, self._detection, self._pose_settings = load_workpiece_config(
            self._share_path('workpiece_config_path', 'config/workpiece_model.yaml')
        )
        self._bed_config = load_bed_config(
            self._share_path('bed_config_path', 'config/cnc_bed.yaml')
        )
        self._object_corners = self._dimensions.object_corners_centered()
        self._camera_matrix: Optional[np.ndarray] = None
        self._distortion: Optional[np.ndarray] = None
        self._tf_broadcaster = TransformBroadcaster(self)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._latest_pose: Optional[PoseEstimate] = None
        self._latest_bed_transform: Optional[np.ndarray] = None
        self._latest_bed_yaw_deg: Optional[float] = None
        self._bed_ekf = BedPoseEkf(
            square_mode=self._pose_settings.is_square_stock,
            config=self._pose_settings.ekf,
        )
        self._recovery_tracker = RecoveryTracker()
        self._consecutive_failures = 0
        self._detection_active = False
        self._estimation_source = 'none'

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value

        self._pose_cam_pub = self.create_publisher(PoseStamped, '/workpiece/actual_pose', 10)
        self._pose_bed_pub = self.create_publisher(PoseStamped, '/workpiece/pose_in_bed_frame', 10)
        self._status_pub = self.create_publisher(String, '/workpiece/bed_coordinates', 10)
        self._marker_pub = self.create_publisher(MarkerArray, '/workpiece/markers', 10)
        self._bed_marker_pub = self.create_publisher(MarkerArray, '/cnc_bed/markers', 10)
        self._target_marker_pub = self.create_publisher(MarkerArray, '/cnc_bed/target_markers', 10)
        self._debug_pub = self.create_publisher(Image, '/workpiece/debug_image', 10)

        self.create_subscription(
            CameraInfo, camera_info_topic, self._info_cb, qos_profile_sensor_data
        )
        self.create_subscription(Image, image_topic, self._image_cb, QOS_IMAGE_SUB)
        self.create_service(Trigger, '/workpiece/reset_filter', self._reset_filter_cb)
        self.create_timer(0.2, self._publish_bed_markers)
        self._publish_bed_markers()
        self.get_logger().info(
            f'Step 05: mode={self._pose_settings.mode} shape={self._pose_settings.shape_mode} '
            f'ekf={"on" if self._pose_settings.ekf.enabled else "off"} '
            f'warmup={self._pose_settings.ekf.warmup_sec:.0f}s '
            f'workpiece {self._dimensions.width_m*1000:.0f}x{self._dimensions.length_m*1000:.0f} mm. '
            f'Reset filter: ros2 service call /workpiece/reset_filter std_srvs/srv/Trigger. '
            f'RViz: /cnc_bed/target_markers + /workpiece/markers'
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
        self._bed_marker_pub.publish(
            make_bed_markers(stamp, 'cnc_bed_frame', self._bed_config, self._dimensions)
        )
        if self._bed_config.show_target_placement:
            target_only = MarkerArray()
            target_only.markers = make_target_placement_markers(
                stamp,
                'cnc_bed_frame',
                self._bed_config.target,
                self._dimensions,
                self._bed_config.coordinate_reporting,
            )
            self._target_marker_pub.publish(target_only)

    def _lookup_bed_from_link(self, stamp) -> Optional[np.ndarray]:
        for query_time in (Time(), stamp):
            try:
                tf_msg = self._tf_buffer.lookup_transform(
                    'cnc_bed_frame',
                    LINK_FRAME,
                    query_time,
                    timeout=duration.Duration(seconds=0.2),
                )
            except Exception:
                continue
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
        return None

    def _reset_pose_filter(self, reason: str) -> None:
        self._latest_pose = None
        self._latest_bed_transform = None
        self._latest_bed_yaw_deg = None
        self._bed_ekf.reset()
        self._recovery_tracker.reset_counters()
        self.get_logger().warn(f'Pose filter reset: {reason}. EKF re-warming.')

    def _reset_filter_cb(self, _request, response):
        self._reset_pose_filter('manual service request')
        response.success = True
        response.message = 'Pose filter reset; EKF will re-warmup from raw measurements.'
        return response

    def _handle_detection_lost(self, stamp) -> None:
        self._consecutive_failures += 1
        if self._detection_active:
            self._marker_pub.publish(make_delete_workpiece_markers(stamp, 'cnc_bed_frame'))
            self._detection_active = False
        reset_after = max(1, self._detection.lost_frames_to_reset_smoothing)
        if self._consecutive_failures >= reset_after:
            self._reset_pose_filter('workpiece detection lost')
        if self._consecutive_failures == 1:
            self.get_logger().info('Workpiece lost — cube hidden.', throttle_duration_sec=1.0)

    def _reported_transform(self, t_bed_workpiece: np.ndarray) -> np.ndarray:
        reporting = self._bed_config.coordinate_reporting
        return apply_coordinate_reporting_offset(
            t_bed_workpiece,
            reporting.subtract_x_m,
            reporting.subtract_y_m,
        )

    def _format_status_lines(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        yaw_deg: float,
        reproj_px: float,
        tilt_deg: float,
        phase: str,
    ) -> list[str]:
        return [
            f'X={x_mm:.1f} mm  Y={y_mm:.1f} mm  Z={z_mm:.1f} mm',
            f'yaw_Z={yaw_deg:.1f} deg  src={self._estimation_source} filter={phase}',
            f'reproj={reproj_px:.1f}px  tilt={tilt_deg:.1f} deg',
        ]

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _estimate_markerless(
        self,
        image,
        t_bed_from_optical: np.ndarray,
        max_tilt: float,
    ) -> tuple[Optional[PoseEstimate], Optional[object]]:
        detection = detect_workpiece_corners(image, self._dimensions, self._detection)
        if detection is None:
            return None, None
        pose = solve_workpiece_pose_in_bed_frame(
            detection.corners,
            self._object_corners,
            self._camera_matrix,
            self._distortion,
            t_bed_from_optical,
            self._dimensions.thickness_m,
            use_ippe_for_planar=self._detection.use_ippe_for_planar,
            max_tilt_deg=max_tilt,
            try_corner_permutations=self._pose_settings.is_square_stock,
            reference_bed_yaw_deg=self._latest_bed_yaw_deg,
            square_yaw_stability_weight=self._detection.square_yaw_stability_weight,
        )
        return pose, detection

    def _estimate_marked(self, image) -> Optional[PoseEstimate]:
        return detect_workpiece_marker_pose(
            image,
            self._camera_matrix,
            self._distortion,
            self._pose_settings.marker,
        )

    def _estimate_pose(
        self,
        image,
        t_bed_from_optical: np.ndarray,
        max_tilt: float,
    ) -> tuple[Optional[PoseEstimate], Optional[object], str]:
        mode = self._pose_settings.mode
        if mode == 'marked':
            pose = self._estimate_marked(image)
            if pose is None:
                return None, None, 'marked'
            return pose, None, 'marked'
        if mode == 'marked_fallback':
            pose = self._estimate_marked(image)
            if pose is not None:
                return pose, None, 'marked'
            if not self._pose_settings.marker.fallback_to_markerless:
                return None, None, 'marked'
            pose, detection = self._estimate_markerless(image, t_bed_from_optical, max_tilt)
            return pose, detection, 'markerless'
        pose, detection = self._estimate_markerless(image, t_bed_from_optical, max_tilt)
        return pose, detection, 'markerless'

    def _apply_bed_pose_filter(
        self, t_bed_workpiece: np.ndarray, stamp_sec: float
    ) -> tuple[np.ndarray, str, bool]:
        if self._detection.assume_flat_on_bed:
            t_bed_workpiece = flatten_transform_to_bed_plane(
                t_bed_workpiece,
                self._dimensions.thickness_m,
            )
        if self._pose_settings.ekf.enabled:
            ekf_result = self._bed_ekf.filter_transform(t_bed_workpiece, stamp_sec)
            self._latest_bed_transform = ekf_result.transform.copy()
            self._latest_bed_yaw_deg = yaw_from_matrix(ekf_result.transform[:3, :3])
            return ekf_result.transform, ekf_result.phase, ekf_result.gated

        if self._detection.assume_flat_on_bed:
            t_bed_workpiece = smooth_flat_bed_transform(
                self._latest_bed_transform,
                t_bed_workpiece,
                self._detection.bed_pose_smoothing_alpha,
                yaw_alpha=self._detection.bed_yaw_smoothing_alpha,
            )
        self._latest_bed_transform = t_bed_workpiece.copy()
        self._latest_bed_yaw_deg = yaw_from_matrix(t_bed_workpiece[:3, :3])
        return t_bed_workpiece, 'LEGACY', False

    def _image_cb(self, msg: Image) -> None:
        if self._camera_matrix is None:
            self.get_logger().warn('Waiting for CameraInfo...', throttle_duration_sec=4.0)
            return

        try:
            image = image_msg_to_bgr(msg)
        except (ValueError, RuntimeError):
            self._handle_detection_lost(msg.header.stamp)
            return

        stamp = msg.header.stamp
        stamp_sec = self._stamp_to_sec(stamp)
        t_bed_link = self._lookup_bed_from_link(stamp)
        if t_bed_link is None:
            self.get_logger().warn(
                'Waiting for TF cnc_bed_frame <- camera_link. Run step03_publish_bed_tf.',
                throttle_duration_sec=3.0,
            )
            self._publish_debug_frame(image, msg, None, 'NO TF')
            self._handle_detection_lost(stamp)
            return

        t_bed_from_optical = t_bed_link @ transform_optical_to_link()
        max_tilt = self.get_parameter('max_surface_tilt_deg').get_parameter_value().double_value

        detection = None
        pose, detection, self._estimation_source = self._estimate_pose(
            image, t_bed_from_optical, max_tilt
        )
        if pose is None:
            if self._consecutive_failures % 15 == 0 and self._pose_settings.mode != 'marked':
                candidates, _ = diagnose_contours(image, self._dimensions, self._detection, top_n=3)
                if candidates:
                    self.get_logger().warn('Contour detection failed. Top candidates:')
                    for index, candidate in enumerate(candidates[:3]):
                        self.get_logger().warn(
                            f'  #{index} score={candidate.score:.3f} {candidate.reason}'
                        )
                else:
                    self.get_logger().warn(
                        'No workpiece pose. Check mode, lighting, and workpiece_model.yaml.',
                        throttle_duration_sec=2.0,
                    )
            elif self._consecutive_failures % 15 == 0 and self._pose_settings.mode in (
                'marked',
                'marked_fallback',
            ):
                self.get_logger().warn(
                    f'Workpiece ArUco id={self._pose_settings.marker.marker_id} not detected.',
                    throttle_duration_sec=2.0,
                )
            self._publish_debug_frame(image, msg, detection, 'NO DETECTION')
            self._handle_detection_lost(stamp)
            return

        max_error = self.get_parameter('max_reprojection_error_px').get_parameter_value().double_value
        if pose.reprojection_error > max_error:
            self.get_logger().warn(
                f'Reprojection error {pose.reprojection_error:.1f}px > {max_error:.1f}px',
                throttle_duration_sec=2.0,
            )
            self._handle_detection_lost(stamp)
            return

        pose = smooth_pose(self._latest_pose, pose, self._detection.pose_smoothing_alpha)
        self._latest_pose = pose
        self._consecutive_failures = 0

        optical_frame = OPTICAL_FRAME
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

        t_bed_workpiece = t_bed_link @ t_link_workpiece
        t_bed_workpiece, filter_phase, measurement_gated = self._apply_bed_pose_filter(
            t_bed_workpiece, stamp_sec
        )

        t_bed_reported = self._reported_transform(t_bed_workpiece)
        if self._pose_settings.shape_mode.strip().lower() == 'square':
            t_bed_reported = apply_square_yaw_fold_to_transform(
                t_bed_reported,
                reference_yaw_deg=self._latest_bed_yaw_deg,
            )
            self._latest_bed_yaw_deg = yaw_from_matrix(t_bed_reported[:3, :3])

        x_mm = float(t_bed_reported[0, 3]) * 1000.0
        y_mm = float(t_bed_reported[1, 3]) * 1000.0
        target_x_mm = self._bed_config.target.x_m * 1000.0
        target_y_mm = self._bed_config.target.y_m * 1000.0
        recovery_reason = check_auto_recovery(
            filter_phase=filter_phase,
            gated=measurement_gated,
            x_mm=x_mm,
            y_mm=y_mm,
            target_x_mm=target_x_mm,
            target_y_mm=target_y_mm,
            config=self._pose_settings.ekf,
            tracker=self._recovery_tracker,
        )
        if recovery_reason is not None:
            self._reset_pose_filter(recovery_reason)
            self._publish_debug_frame(image, msg, detection, 'FILTER RESET')
            return

        bed_margin = self.get_parameter('bed_margin_m').get_parameter_value().double_value

        if not is_pose_center_on_bed(
            t_bed_workpiece,
            self._bed_config.bed.length_m,
            self._bed_config.bed.width_m,
            margin_m=bed_margin,
        ):
            z_mm = float(t_bed_reported[2, 3]) * 1000.0
            self.get_logger().warn(
                f'Workpiece center outside CNC bed — not publishing bed pose. '
                f'Computed X={x_mm:.1f} mm Y={y_mm:.1f} mm Z={z_mm:.1f} mm',
                throttle_duration_sec=2.0,
            )
            self._publish_debug_frame(image, msg, detection, f'OFF BED {x_mm:.0f},{y_mm:.0f}')
            self._handle_detection_lost(stamp)
            return

        tilt_deg = surface_normal_tilt_deg(t_bed_workpiece[:3, :3])
        require_flat = self.get_parameter('publish_pose_only_when_flat').get_parameter_value().bool_value
        if tilt_deg > max_tilt and not self._detection.assume_flat_on_bed:
            self.get_logger().warn(
                f'Workpiece tilt {tilt_deg:.1f} deg > {max_tilt:.1f} deg',
                throttle_duration_sec=2.0,
            )
            self._publish_debug_frame(image, msg, detection, f'TILT {tilt_deg:.0f}deg')
            if require_flat:
                self._handle_detection_lost(stamp)
                return

        pose_bed = PoseStamped()
        pose_bed.header.stamp = stamp
        pose_bed.header.frame_id = 'cnc_bed_frame'
        pose_bed.pose = matrix_to_pose(t_bed_reported)
        self._pose_bed_pub.publish(pose_bed)

        tf_bed_wp = TransformStamped()
        tf_bed_wp.header.stamp = stamp
        tf_bed_wp.header.frame_id = 'cnc_bed_frame'
        tf_bed_wp.child_frame_id = 'workpiece_frame'
        tf_bed_wp.transform.translation.x = float(t_bed_reported[0, 3])
        tf_bed_wp.transform.translation.y = float(t_bed_reported[1, 3])
        tf_bed_wp.transform.translation.z = float(t_bed_reported[2, 3])
        bed_quat = rotation_matrix_to_quaternion(t_bed_reported[:3, :3])
        tf_bed_wp.transform.rotation = bed_quat
        self._tf_broadcaster.sendTransform([tf_bed_wp, tf_optical_wp])

        z_mm = float(t_bed_reported[2, 3]) * 1000.0
        yaw_deg = yaw_from_matrix(t_bed_reported[:3, :3])
        status_lines = self._format_status_lines(
            x_mm, y_mm, z_mm, yaw_deg, pose.reprojection_error, tilt_deg, filter_phase
        )
        status = String()
        status.data = 'Bed frame: ' + ' | '.join(status_lines)
        self._status_pub.publish(status)
        self.get_logger().info(status.data, throttle_duration_sec=1.0)

        markers = make_workpiece_markers_at_pose(
            stamp, 'cnc_bed_frame', pose_bed.pose, self._dimensions
        )
        markers.markers.append(
            make_pose_status_marker(stamp, 'cnc_bed_frame', pose_bed.pose, status_lines)
        )
        self._marker_pub.publish(markers)
        self._detection_active = True

        debug_label = f'OK {self._estimation_source.upper()} {filter_phase}'
        self._publish_debug_frame(
            image, msg, detection, debug_label, status_lines=status_lines
        )

    def _publish_debug_frame(
        self,
        image,
        msg: Image,
        detection,
        label: str,
        status_lines: Optional[list[str]] = None,
    ) -> None:
        if not self._detection.publish_debug_image:
            return
        if detection is None:
            candidates, edges = diagnose_contours(image, self._dimensions, self._detection, top_n=3)
            from cnc_perception.contour_detector import draw_diagnostic_overlay
            debug = draw_diagnostic_overlay(image, candidates, edges)
            if self._pose_settings.mode in ('marked', 'marked_fallback'):
                debug = draw_workpiece_marker_debug(debug, self._pose_settings.marker)
            cv2.putText(
                debug, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA
            )
        else:
            debug = draw_detection_debug(image, detection, label, status_lines=status_lines)
        if self._pose_settings.mode in ('marked', 'marked_fallback') and detection is None:
            debug = draw_workpiece_marker_debug(debug, self._pose_settings.marker)
        elif self._estimation_source == 'marked':
            debug = draw_workpiece_marker_debug(debug, self._pose_settings.marker)
        debug_msg = bgr_to_image_msg(debug, msg.header, msg.header.frame_id or OPTICAL_FRAME)
        self._debug_pub.publish(debug_msg)

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
