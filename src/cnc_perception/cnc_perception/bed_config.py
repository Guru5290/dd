"""CNC bed configuration loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BedDimensions:
    length_m: float
    width_m: float


@dataclass(frozen=True)
class TargetPlacement:
    x_m: float
    y_m: float
    z_m: float
    yaw_deg: float
    tolerance_xy_mm: float
    tolerance_z_mm: float
    tolerance_yaw_deg: float


@dataclass(frozen=True)
class ReferenceMarker:
    dictionary: str
    marker_id: int
    marker_size_m: float
    placement: str


@dataclass(frozen=True)
class BedConfig:
    bed: BedDimensions
    marker: ReferenceMarker
    target: TargetPlacement
    mesh_enabled: bool
    mesh_stl_path: str
    mesh_scale: float


def load_bed_config(config_path: str) -> BedConfig:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f'CNC bed config not found: {config_path}')

    with path.open('r', encoding='utf-8') as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    bed = raw.get('cnc_bed', {})
    marker = raw.get('reference_marker', {})
    target = raw.get('target_placement', {})
    tol = target.get('tolerance', {})
    mesh = raw.get('workpiece_mesh', {})

    return BedConfig(
        bed=BedDimensions(
            length_m=float(bed.get('length_m', 0.25)),
            width_m=float(bed.get('width_m', 0.18)),
        ),
        marker=ReferenceMarker(
            dictionary=str(marker.get('dictionary', 'DICT_4X4_50')),
            marker_id=int(marker.get('marker_id', 0)),
            marker_size_m=float(marker.get('marker_size_m', 0.04)),
            placement=str(marker.get('placement', 'center_at_origin')),
        ),
        target=TargetPlacement(
            x_m=float(target.get('x_m', 0.1)),
            y_m=float(target.get('y_m', 0.1)),
            z_m=float(target.get('z_m', 0.0)),
            yaw_deg=float(target.get('yaw_deg', 0.0)),
            tolerance_xy_mm=float(tol.get('xy_mm', 2.0)),
            tolerance_z_mm=float(tol.get('z_mm', 1.5)),
            tolerance_yaw_deg=float(tol.get('yaw_deg', 3.0)),
        ),
        mesh_enabled=bool(mesh.get('enabled', False)),
        mesh_stl_path=str(mesh.get('stl_path', '')),
        mesh_scale=float(mesh.get('scale', 1.0)),
    )
