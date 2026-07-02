"""Run ROS commands in sourced bash subprocesses."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional


class ManagedProcess:
    def __init__(self, name: str) -> None:
        self.name = name
        self._process: Optional[subprocess.Popen[str]] = None
        self._output_thread: Optional[threading.Thread] = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(
        self,
        command: str,
        workspace_root: Path,
        ros_setup: Path,
        on_output: Callable[[str], None],
        on_exit: Callable[[int], None],
    ) -> None:
        if self.running:
            return
        bash_cmd = (
            f'source "{ros_setup}" && '
            f'source "{workspace_root / "install" / "setup.bash"}" && '
            f'{command}'
        )
        self._process = subprocess.Popen(
            ['bash', '-lc', bash_cmd],
            cwd=str(workspace_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        def _reader() -> None:
            assert self._process is not None
            assert self._process.stdout is not None
            for line in self._process.stdout:
                on_output(f'[{self.name}] {line.rstrip()}')
            code = self._process.wait()
            self._process = None
            on_exit(code)

        self._output_thread = threading.Thread(target=_reader, daemon=True)
        self._output_thread.start()

    def stop(self) -> None:
        if not self.running or self._process is None:
            return
        try:
            os.killpg(self._process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self._process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self._process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self._process = None


def find_ros_setup() -> Path:
    for distro in ('jazzy', 'iron', 'humble'):
        candidate = Path(f'/opt/ros/{distro}/setup.bash')
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('No ROS setup.bash found under /opt/ros')


def run_build(
    workspace_root: Path,
    ros_setup: Path,
    on_output: Callable[[str], None],
    on_done: Callable[[int], None],
) -> threading.Thread:
    def _worker() -> None:
        bash_cmd = (
            f'source "{ros_setup}" && '
            f'cd "{workspace_root}" && '
            'colcon build --packages-select cnc_perception'
        )
        process = subprocess.Popen(
            ['bash', '-lc', bash_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            on_output(line.rstrip())
        on_done(process.wait())

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread
