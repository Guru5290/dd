"""RViz markers for CNC bed, target pose, and placement status."""

from __future__ import annotations

import math
from typing import Optional

from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from cnc_perception.bed_config import BedConfig, CoordinateReporting, TargetPlacement
from cnc_perception.workpiece_config import WorkpieceDimensions


def _color(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    return ColorRGBA(r=r, g=g, b=b, a=a)


def _identity_pose() -> Pose:
    pose = Pose()
    pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    return pose


def make_bed_markers(
    stamp,
    frame_id: str,
    bed_config: BedConfig,
    workpiece_dimensions: Optional[WorkpieceDimensions] = None,
) -> MarkerArray:
    """Bed outline, grid lines, origin, and target placement in cnc_bed_frame."""
    length = bed_config.bed.length_m
    width = bed_config.bed.width_m
    markers = MarkerArray()
    marker_id = 0

    plate = Marker()
    plate.header.stamp = stamp
    plate.header.frame_id = frame_id
    plate.ns = 'cnc_bed'
    plate.id = marker_id
    marker_id += 1
    plate.type = Marker.CUBE
    plate.action = Marker.ADD
    plate.pose = _identity_pose()
    plate.pose.position.x = length / 2.0
    plate.pose.position.y = width / 2.0
    plate.pose.position.z = -0.002
    plate.scale = Vector3(x=length, y=width, z=0.004)
    plate.color = _color(0.35, 0.35, 0.38, 0.55)
    markers.markers.append(plate)

    outline = Marker()
    outline.header.stamp = stamp
    outline.header.frame_id = frame_id
    outline.ns = 'cnc_bed'
    outline.id = marker_id
    marker_id += 1
    outline.type = Marker.LINE_STRIP
    outline.action = Marker.ADD
    outline.pose = _identity_pose()
    outline.scale.x = 0.004
    outline.color = _color(1.0, 1.0, 1.0, 1.0)
    outline.points = [
        Point(x=0.0, y=0.0, z=0.0),
        Point(x=length, y=0.0, z=0.0),
        Point(x=length, y=width, z=0.0),
        Point(x=0.0, y=width, z=0.0),
        Point(x=0.0, y=0.0, z=0.0),
    ]
    markers.markers.append(outline)

    origin = Marker()
    origin.header.stamp = stamp
    origin.header.frame_id = frame_id
    origin.ns = 'cnc_bed'
    origin.id = marker_id
    marker_id += 1
    origin.type = Marker.SPHERE
    origin.action = Marker.ADD
    origin.pose = _identity_pose()
    origin.scale = Vector3(x=0.015, y=0.015, z=0.015)
    origin.color = _color(1.0, 0.2, 0.2, 1.0)
    markers.markers.append(origin)

    axis_len = min(length, width) * 0.2
    axes = Marker()
    axes.header.stamp = stamp
    axes.header.frame_id = frame_id
    axes.ns = 'cnc_bed'
    axes.id = marker_id
    marker_id += 1
    axes.type = Marker.LINE_LIST
    axes.action = Marker.ADD
    axes.pose = _identity_pose()
    axes.scale.x = 0.006
    axes.points = [
        Point(x=0.0, y=0.0, z=0.0),
        Point(x=axis_len, y=0.0, z=0.0),
        Point(x=0.0, y=0.0, z=0.0),
        Point(x=0.0, y=axis_len, z=0.0),
        Point(x=0.0, y=0.0, z=0.0),
        Point(x=0.0, y=0.0, z=axis_len),
    ]

    axes.colors = [
        _color(1.0, 0.0, 0.0), _color(1.0, 0.0, 0.0),
        _color(0.0, 1.0, 0.0), _color(0.0, 1.0, 0.0),
        _color(0.0, 0.4, 1.0), _color(0.0, 0.4, 1.0),
    ]

    markers.markers.append(axes)

    if bed_config.show_target_placement and workpiece_dimensions is not None:
        target_markers = make_target_placement_markers(
            stamp,
            frame_id,
            bed_config.target,
            workpiece_dimensions,
            bed_config.coordinate_reporting,
            marker_id,
        )
        markers.markers.extend(target_markers)
        marker_id += len(target_markers)

    label = Marker()
    label.header.stamp = stamp
    label.header.frame_id = frame_id
    label.ns = 'cnc_bed'
    label.id = marker_id
    label.type = Marker.TEXT_VIEW_FACING
    label.action = Marker.ADD
    label.pose = _identity_pose()
    label.pose.position.x = 0.01
    label.pose.position.y = 0.01
    label.pose.position.z = 0.02
    label.scale.z = 0.025
    label.color = _color(1.0, 1.0, 1.0, 1.0)
    label.text = (
        f'Bed origin (0,0,0)  L={length*1000:.0f}mm W={width*1000:.0f}mm'
        + (
            f'  Target center=({bed_config.target.x_m*1000:.0f},{bed_config.target.y_m*1000:.0f})mm'
            f' yaw={bed_config.target.yaw_deg:.0f}deg'
            if bed_config.show_target_placement
            else ''
        )
    )
    markers.markers.append(label)

    return markers


def _rotated_rectangle_corners(
    center_x: float,
    center_y: float,
    half_width: float,
    half_length: float,
    yaw_deg: float,
    z: float,
) -> list[Point]:
    yaw_rad = math.radians(yaw_deg)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    local_corners = [
        (-half_width, -half_length),
        (half_width, -half_length),
        (half_width, half_length),
        (-half_width, half_length),
    ]
    corners: list[Point] = []
    for local_x, local_y in local_corners:
        world_x = center_x + cos_yaw * local_x - sin_yaw * local_y
        world_y = center_y + sin_yaw * local_x + cos_yaw * local_y
        corners.append(Point(x=world_x, y=world_y, z=z))
    return corners


def _dotted_edge_points(
    start: Point,
    end: Point,
    dash_length_m: float,
    gap_length_m: float,
) -> list[Point]:
    dx = end.x - start.x
    dy = end.y - start.y
    dz = end.z - start.z
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length <= 1e-9:
        return []

    ux = dx / length
    uy = dy / length
    uz = dz / length
    points: list[Point] = []
    traveled = 0.0
    drawing = True
    while traveled < length:
        step = dash_length_m if drawing else gap_length_m
        next_traveled = min(traveled + step, length)
        if drawing:
            points.append(
                Point(
                    x=start.x + ux * traveled,
                    y=start.y + uy * traveled,
                    z=start.z + uz * traveled,
                )
            )
            points.append(
                Point(
                    x=start.x + ux * next_traveled,
                    y=start.y + uy * next_traveled,
                    z=start.z + uz * next_traveled,
                )
            )
        traveled = next_traveled
        drawing = not drawing
    return points


def _dotted_rectangle_points(
    corners: list[Point],
    dash_length_m: float,
    gap_length_m: float,
) -> list[Point]:
    if len(corners) != 4:
        return []
    points: list[Point] = []
    for index in range(4):
        start = corners[index]
        end = corners[(index + 1) % 4]
        points.extend(_dotted_edge_points(start, end, dash_length_m, gap_length_m))
    return points


def _target_center_in_bed_frame(
    target: TargetPlacement,
    reporting: CoordinateReporting,
) -> tuple[float, float]:
    """Target uses the same coordinates as /workpiece/pose_in_bed_frame (ruler frame)."""
    _ = reporting
    return target.x_m, target.y_m


def make_target_placement_markers(
    stamp,
    frame_id: str,
    target: TargetPlacement,
    dimensions: WorkpieceDimensions,
    reporting: CoordinateReporting,
    marker_id: int = 0,
) -> list[Marker]:
    """Green target footprint (dotted + solid outline) centered at target X/Y."""
    center_x, center_y = _target_center_in_bed_frame(target, reporting)
    return _make_target_outline_markers(
        stamp,
        frame_id,
        target,
        dimensions,
        marker_id,
        center_x=center_x,
        center_y=center_y,
    )


def _make_target_outline_markers(
    stamp,
    frame_id: str,
    target: TargetPlacement,
    dimensions: WorkpieceDimensions,
    marker_id: int,
    *,
    center_x: float,
    center_y: float,
) -> list[Marker]:
    """Green dotted footprint centered at target X/Y with target yaw."""
    half_w = dimensions.width_m / 2.0
    half_l = dimensions.length_m / 2.0
    z = 0.012
    corners = _rotated_rectangle_corners(
        center_x,
        center_y,
        half_w,
        half_l,
        target.yaw_deg,
        z,
    )

    solid = Marker()
    solid.header.stamp = stamp
    solid.header.frame_id = frame_id
    solid.ns = 'target'
    solid.id = marker_id
    solid.type = Marker.LINE_STRIP
    solid.action = Marker.ADD
    solid.pose = _identity_pose()
    solid.scale.x = 0.005
    solid.color = _color(0.72, 0.28, 0.95, 0.95)
    solid.points = corners + [corners[0]]

    outline = Marker()
    outline.header.stamp = stamp
    outline.header.frame_id = frame_id
    outline.ns = 'target'
    outline.id = marker_id + 1
    outline.type = Marker.LINE_LIST
    outline.action = Marker.ADD
    outline.pose = _identity_pose()
    outline.scale.x = 0.006
    outline.color = _color(0.72, 0.28, 0.95, 1.0)
    outline.points = _dotted_rectangle_points(corners, dash_length_m=0.008, gap_length_m=0.005)

    center = Marker()
    center.header.stamp = stamp
    center.header.frame_id = frame_id
    center.ns = 'target'
    center.id = marker_id + 2
    center.type = Marker.SPHERE
    center.action = Marker.ADD
    center.pose = _identity_pose()
    center.pose.position.x = center_x
    center.pose.position.y = center_y
    center.pose.position.z = z
    center.scale = Vector3(x=0.008, y=0.008, z=0.008)
    center.color = _color(0.72, 0.28, 0.95, 1.0)

    yaw_rad = math.radians(target.yaw_deg)
    axis_len = min(dimensions.width_m, dimensions.length_m) * 0.45
    yaw_axis = Marker()
    yaw_axis.header.stamp = stamp
    yaw_axis.header.frame_id = frame_id
    yaw_axis.ns = 'target'
    yaw_axis.id = marker_id + 3
    yaw_axis.type = Marker.LINE_LIST
    yaw_axis.action = Marker.ADD
    yaw_axis.pose = _identity_pose()
    yaw_axis.scale.x = 0.005
    yaw_axis.color = _color(1.0, 1.0, 0.15, 1.0)
    yaw_axis.points = [
        Point(x=center_x, y=center_y, z=z),
        Point(
            x=center_x + math.cos(yaw_rad) * axis_len,
            y=center_y + math.sin(yaw_rad) * axis_len,
            z=z,
        ),
    ]

    label = Marker()
    label.header.stamp = stamp
    label.header.frame_id = frame_id
    label.ns = 'target'
    label.id = marker_id + 4
    label.type = Marker.TEXT_VIEW_FACING
    label.action = Marker.ADD
    label.pose = _identity_pose()
    label.pose.position.x = center_x
    label.pose.position.y = center_y
    label.pose.position.z = z + 0.02
    label.scale.z = 0.014
    label.color = _color(0.78, 0.45, 1.0, 1.0)
    label.text = (
        f'TARGET {target.x_m*1000:.0f},{target.y_m*1000:.0f} mm '
        f'yaw {target.yaw_deg:.0f} deg'
    )

    return [solid, outline, center, yaw_axis, label]


def make_placement_status_marker(
    stamp,
    frame_id: str,
    text: str,
    ok: bool,
) -> Marker:
    marker = Marker()
    marker.header.stamp = stamp
    marker.header.frame_id = frame_id
    marker.ns = 'status'
    marker.id = 0
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD
    marker.pose = _identity_pose()
    marker.pose.position.z = 0.12
    marker.scale.z = 0.04
    marker.color = _color(0.1, 1.0, 0.2, 1.0) if ok else _color(1.0, 0.2, 0.1, 1.0)
    marker.text = text
    return marker
