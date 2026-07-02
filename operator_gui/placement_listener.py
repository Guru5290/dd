"""Background ROS listener for placement status."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


@dataclass(frozen=True)
class PlacementStatus:
    ok: Optional[bool]
    message: str
    dx_mm: Optional[float] = None
    dy_mm: Optional[float] = None
    dyaw_deg: Optional[float] = None


_DELTA_RE = re.compile(
    r'delta=\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)\s*mm\s*'
    r'dyaw=\s*([+-]?\d+(?:\.\d+)?)\s*deg'
)


def parse_placement_message(text: str) -> PlacementStatus:
    if 'CORRECT POSITION' in text:
        return PlacementStatus(ok=True, message=text)
    if 'NOT CORRECT POSITION' in text:
        match = _DELTA_RE.search(text)
        if match:
            return PlacementStatus(
                ok=False,
                message=text,
                dx_mm=float(match.group(1)),
                dy_mm=float(match.group(2)),
                dyaw_deg=float(match.group(3)),
            )
        return PlacementStatus(ok=False, message=text)
    return PlacementStatus(ok=None, message=text)


class PlacementListener:
    def __init__(self, callback: Callable[[PlacementStatus], None]) -> None:
        self._callback = callback
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._node: Optional[Node] = None

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.active:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._node is not None:
            rclpy.try_shutdown()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._node = None

    def _run(self) -> None:
        try:
            if not rclpy.ok():
                rclpy.init()
            node = Node('cnc_operator_gui_placement_listener')
            self._node = node

            def _on_msg(msg: String) -> None:
                self._callback(parse_placement_message(msg.data))

            node.create_subscription(String, '/workpiece/placement_status', _on_msg, 10)
            while not self._stop_event.is_set() and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)
        finally:
            if rclpy.ok():
                rclpy.try_shutdown()
