"""Check whether workpiece pose meets target placement on CNC bed."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cnc_perception.bed_config import TargetPlacement
from cnc_perception.transform_utils import fold_square_yaw_deg, yaw_from_matrix


@dataclass(frozen=True)
class PlacementResult:
    ok: bool
    message: str
    x_mm: float
    y_mm: float
    z_mm: float
    yaw_deg: float
    dx_mm: float
    dy_mm: float
    dz_mm: float
    dyaw_deg: float


def check_placement(
    t_bed_workpiece: np.ndarray,
    workpiece_thickness_m: float,
    target: TargetPlacement,
    *,
    square_yaw_fold: bool = False,
) -> PlacementResult:
    """
    Evaluate workpiece center pose in cnc_bed_frame.

    Z of workpiece top face should be approximately thickness when sitting on bed.
    """
    x_m = float(t_bed_workpiece[0, 3])
    y_m = float(t_bed_workpiece[1, 3])
    z_m = float(t_bed_workpiece[2, 3])
    yaw_deg = yaw_from_matrix(t_bed_workpiece)
    target_yaw_deg = target.yaw_deg
    if square_yaw_fold:
        target_yaw_deg = fold_square_yaw_deg(target_yaw_deg)
        yaw_deg = fold_square_yaw_deg(yaw_deg, reference_yaw_deg=target_yaw_deg)

    dx_mm = (x_m - target.x_m) * 1000.0
    dy_mm = (y_m - target.y_m) * 1000.0
    dz_mm = (z_m - workpiece_thickness_m) * 1000.0
    dyaw_deg = yaw_deg - target_yaw_deg

    ok = (
        abs(dx_mm) <= target.tolerance_xy_mm
        and abs(dy_mm) <= target.tolerance_xy_mm
        and abs(dz_mm) <= target.tolerance_z_mm
        and abs(dyaw_deg) <= target.tolerance_yaw_deg
    )

    if ok:
        message = (
            f'CORRECT POSITION | center=({x_m*1000:.1f}, {y_m*1000:.1f}) mm '
            f'z_top={z_m*1000:.1f} mm yaw={yaw_deg:.1f} deg'
        )
    else:
        message = (
            f'NOT CORRECT POSITION | delta=({dx_mm:+.1f}, {dy_mm:+.1f}, {dz_mm:+.1f}) mm '
            f'dyaw={dyaw_deg:+.1f} deg | target=({target.x_m*1000:.0f}, {target.y_m*1000:.0f}) mm'
        )

    return PlacementResult(
        ok=ok,
        message=message,
        x_mm=x_m * 1000.0,
        y_mm=y_m * 1000.0,
        z_mm=z_m * 1000.0,
        yaw_deg=yaw_deg,
        dx_mm=dx_mm,
        dy_mm=dy_mm,
        dz_mm=dz_mm,
        dyaw_deg=dyaw_deg,
    )


def _angle_diff_deg(measured: float, target: float) -> float:
    diff = (measured - target + 180.0) % 360.0 - 180.0
    return float(diff)
