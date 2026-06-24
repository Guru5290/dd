"""RViz markers for CNC bed, target pose, and placement status."""

from __future__ import annotations

from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from cnc_perception.bed_config import BedConfig, TargetPlacement


def _color(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    return ColorRGBA(r=r, g=g, b=b, a=a)


def _identity_pose() -> Pose:
    pose = Pose()
    pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    return pose


def make_bed_markers(stamp, frame_id: str, bed_config: BedConfig) -> MarkerArray:
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

    if bed_config.show_target_placement:
        target = _make_target_marker(stamp, frame_id, bed_config.target, marker_id, ok=True)
        markers.markers.append(target)
        marker_id += 1

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
            f'  Target=({bed_config.target.x_m*1000:.0f},{bed_config.target.y_m*1000:.0f})mm'
            if bed_config.show_target_placement
            else ''
        )
    )
    markers.markers.append(label)

    return markers


def _make_target_marker(
    stamp,
    frame_id: str,
    target: TargetPlacement,
    marker_id: int,
    ok: bool,
) -> Marker:
    marker = Marker()
    marker.header.stamp = stamp
    marker.header.frame_id = frame_id
    marker.ns = 'target'
    marker.id = marker_id
    marker.type = Marker.CYLINDER
    marker.action = Marker.ADD
    marker.pose = _identity_pose()
    marker.pose.position.x = target.x_m
    marker.pose.position.y = target.y_m
    marker.pose.position.z = 0.005
    marker.scale = Vector3(x=0.03, y=0.03, z=0.01)
    marker.color = _color(0.2, 0.9, 0.2, 0.9) if ok else _color(0.9, 0.2, 0.2, 0.9)
    return marker


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
