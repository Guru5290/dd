# Copyright 2026 CNC Perception Team

from placement_listener import parse_placement_message


def test_not_correct_is_not_parsed_as_correct() -> None:
    message = (
        'NOT CORRECT POSITION | delta=(+1.2, -0.8) mm '
        'dyaw=+2.5 deg | target=(100, 100) mm'
    )
    status = parse_placement_message(message)
    assert status.ok is False
    assert status.dx_mm == 1.2
    assert status.dy_mm == -0.8
    assert status.dyaw_deg == 2.5


def test_correct_position() -> None:
    message = 'CORRECT POSITION | center=(100.0, 100.0) mm z_top=10.0 mm yaw=0.0 deg'
    status = parse_placement_message(message)
    assert status.ok is True
