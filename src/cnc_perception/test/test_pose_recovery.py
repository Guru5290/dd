# Copyright 2026 CNC Perception Team

from cnc_perception.pose_ekf import EkfConfig, RecoveryTracker, check_auto_recovery


def _config(**overrides) -> EkfConfig:
    defaults = dict(
        enabled=True,
        warmup_sec=15.0,
        max_yaw_rate_deg_s=10.0,
        max_position_rate_m_s=0.05,
        process_noise_xy_m=0.0002,
        process_noise_z_m=0.0001,
        process_noise_yaw_deg=0.5,
        measurement_noise_xy_m=0.001,
        measurement_noise_z_m=0.001,
        measurement_noise_yaw_deg=2.0,
        recovery_enabled=True,
        recovery_gated_frames=3,
        recovery_off_target_xy_mm=5.0,
        recovery_off_target_frames=3,
    )
    defaults.update(overrides)
    return EkfConfig(**defaults)


def test_recovery_ignored_during_warmup() -> None:
    tracker = RecoveryTracker()
    tracker.consecutive_gated = 2
    reason = check_auto_recovery(
        filter_phase='WARMUP',
        gated=True,
        x_mm=50.0,
        y_mm=50.0,
        target_x_mm=100.0,
        target_y_mm=100.0,
        config=_config(),
        tracker=tracker,
    )
    assert reason is None
    assert tracker.consecutive_gated == 0


def test_recovery_on_consecutive_gated_frames() -> None:
    tracker = RecoveryTracker()
    config = _config(recovery_gated_frames=3)
    for _ in range(2):
        assert (
            check_auto_recovery(
                filter_phase='EKF',
                gated=True,
                x_mm=100.0,
                y_mm=100.0,
                target_x_mm=100.0,
                target_y_mm=100.0,
                config=config,
                tracker=tracker,
            )
            is None
        )
    reason = check_auto_recovery(
        filter_phase='EKF',
        gated=True,
        x_mm=100.0,
        y_mm=100.0,
        target_x_mm=100.0,
        target_y_mm=100.0,
        config=config,
        tracker=tracker,
    )
    assert reason is not None
    assert 'rejected 3 consecutive' in reason


def test_recovery_on_pose_far_from_target() -> None:
    tracker = RecoveryTracker()
    config = _config(recovery_off_target_frames=2, recovery_off_target_xy_mm=5.0)
    assert (
        check_auto_recovery(
            filter_phase='EKF',
            gated=False,
            x_mm=120.0,
            y_mm=100.0,
            target_x_mm=100.0,
            target_y_mm=100.0,
            config=config,
            tracker=tracker,
        )
        is None
    )
    reason = check_auto_recovery(
        filter_phase='EKF',
        gated=False,
        x_mm=120.0,
        y_mm=100.0,
        target_x_mm=100.0,
        target_y_mm=100.0,
        config=config,
        tracker=tracker,
    )
    assert reason is not None
    assert 'from target' in reason
