"""RViz marker helpers for workpiece pose visualization."""

from __future__ import annotations

from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from cnc_perception.workpiece_config import WorkpieceDimensions


def _identity_pose() -> Pose:
    pose = Pose()
    pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    return pose


def _color(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    return ColorRGBA(r=r, g=g, b=b, a=a)


def make_workpiece_markers(
    stamp,
    frame_id: str,
    dimensions: WorkpieceDimensions,
) -> MarkerArray:
    """Build markers expressed in workpiece_frame (origin at top-face center)."""
    half_w = dimensions.width_m / 2.0
    half_l = dimensions.length_m / 2.0
    thickness = dimensions.thickness_m

    markers = MarkerArray()

    body = Marker()
    body.header.stamp = stamp
    body.header.frame_id = frame_id
    body.ns = 'workpiece'
    body.id = 0
    body.type = Marker.CUBE
    body.action = Marker.ADD
    body.pose = _identity_pose()
    body.pose.position.z = -thickness / 2.0
    body.scale = Vector3(
        x=dimensions.width_m,
        y=dimensions.length_m,
        z=max(thickness, 0.001),
    )
    body.color = _color(0.2, 0.65, 0.95, 0.45)
    markers.markers.append(body)

    outline = Marker()
    outline.header.stamp = stamp
    outline.header.frame_id = frame_id
    outline.ns = 'workpiece'
    outline.id = 1
    outline.type = Marker.LINE_STRIP
    outline.action = Marker.ADD
    outline.pose = _identity_pose()
    outline.scale.x = 0.003
    outline.color = _color(0.1, 1.0, 0.2, 1.0)
    outline.points = [
        Point(x=-half_w, y=-half_l, z=0.0),
        Point(x=half_w, y=-half_l, z=0.0),
        Point(x=half_w, y=half_l, z=0.0),
        Point(x=-half_w, y=half_l, z=0.0),
        Point(x=-half_w, y=-half_l, z=0.0),
    ]
    markers.markers.append(outline)

    axes = Marker()
    axes.header.stamp = stamp
    axes.header.frame_id = frame_id
    axes.ns = 'workpiece'
    axes.id = 2
    axes.type = Marker.LINE_LIST
    axes.action = Marker.ADD
    axes.pose = _identity_pose()
    axes.scale.x = 0.004
    axis_len = min(dimensions.width_m, dimensions.length_m) * 0.35
    axes.points = [
        Point(x=0.0, y=0.0, z=0.0),
        Point(x=axis_len, y=0.0, z=0.0),
        Point(x=0.0, y=0.0, z=0.0),
        Point(x=0.0, y=axis_len, z=0.0),
        Point(x=0.0, y=0.0, z=0.0),
        Point(x=0.0, y=0.0, z=axis_len),
    ]
    axes.colors = [
        _color(1.0, 0.1, 0.1),
        _color(1.0, 0.1, 0.1),
        _color(0.1, 1.0, 0.1),
        _color(0.1, 1.0, 0.1),
        _color(0.1, 0.4, 1.0),
        _color(0.1, 0.4, 1.0),
    ]
    markers.markers.append(axes)

    return markers
