# Copyright 2026 CNC Perception Team
#
# Licensed under the Apache License, Version 2.0

from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker

from cnc_perception.visualization import make_workpiece_markers
from cnc_perception.workpiece_config import WorkpieceDimensions


def test_make_workpiece_markers() -> None:
    dims = WorkpieceDimensions(width_m=0.12, length_m=0.08, thickness_m=0.01)
    stamp = Time(sec=1, nanosec=0)
    markers = make_workpiece_markers(stamp=stamp, frame_id='workpiece_frame', dimensions=dims)
    assert len(markers.markers) == 3
    assert markers.markers[0].type == Marker.CUBE
    assert markers.markers[1].type == Marker.LINE_STRIP
    assert len(markers.markers[1].points) == 5
