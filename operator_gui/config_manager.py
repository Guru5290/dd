"""Manage runtime copies of config files (does not touch package configs)."""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml


class RuntimeConfigManager:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self.runtime_dir = workspace_root / 'operator_gui' / 'runtime'
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

        self.source_bed_config = workspace_root / 'src' / 'cnc_perception' / 'config' / 'cnc_bed.yaml'
        self.source_bed_calibration = (
            workspace_root / 'src' / 'cnc_perception' / 'config' / 'bed_calibration.yaml'
        )

        self.runtime_bed_config = self.runtime_dir / 'cnc_bed_gui.yaml'
        self.runtime_bed_calibration = self.runtime_dir / 'bed_calibration_gui.yaml'

        self._ensure_runtime_files()

    def _ensure_runtime_files(self) -> None:
        if not self.runtime_bed_config.exists() and self.source_bed_config.is_file():
            shutil.copy2(self.source_bed_config, self.runtime_bed_config)
        if not self.runtime_bed_calibration.exists() and self.source_bed_calibration.is_file():
            shutil.copy2(self.source_bed_calibration, self.runtime_bed_calibration)

    def read_target_mm_deg(self) -> tuple[float, float, float]:
        self._ensure_runtime_files()
        with self.runtime_bed_config.open('r', encoding='utf-8') as handle:
            data = yaml.safe_load(handle) or {}
        target = data.get('target_placement', {})
        return (
            float(target.get('x_m', 0.1)) * 1000.0,
            float(target.get('y_m', 0.1)) * 1000.0,
            float(target.get('yaw_deg', 0.0)),
        )

    def write_target_mm_deg(self, x_mm: float, y_mm: float, yaw_deg: float) -> None:
        self._ensure_runtime_files()
        with self.runtime_bed_config.open('r', encoding='utf-8') as handle:
            data = yaml.safe_load(handle) or {}
        target = data.setdefault('target_placement', {})
        target['x_m'] = float(x_mm) / 1000.0
        target['y_m'] = float(y_mm) / 1000.0
        target['yaw_deg'] = float(yaw_deg)
        with self.runtime_bed_config.open('w', encoding='utf-8') as handle:
            yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)
