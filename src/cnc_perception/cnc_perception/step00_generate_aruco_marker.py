#!/usr/bin/env python3
"""Step 00: Generate printable ArUco marker for bed origin (DICT_4X4_50 id 0)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate ArUco bed origin marker PNG')
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--size-px', type=int, default=400, help='Marker image size in pixels')
    parser.add_argument('--border-bits', type=int, default=1)
    parser.add_argument(
        '--output',
        type=str,
        default='/tmp/aruco_bed_origin_id0.png',
        help='Output PNG path',
    )
    args = parser.parse_args()

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = np.zeros((args.size_px, args.size_px), dtype=np.uint8)
    if hasattr(cv2.aruco, 'generateImageMarker'):
        cv2.aruco.generateImageMarker(dictionary, args.id, args.size_px, marker, args.border_bits)
    else:
        cv2.aruco.drawMarker(dictionary, args.id, args.size_px, marker, args.border_bits)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), marker)
    print(f'Saved ArUco marker id={args.id} to {output}')
    print('Print at exactly the size configured in cnc_bed.yaml (marker_size_m).')
    print('Place FLAT at FRONT-LEFT bed corner (bed origin).')
    print('Orient so the marker looks UPRIGHT in the camera — one edge along the front bed edge (+X).')
    print('See SETUP_AND_CALIBRATION_GUIDE.md for a diagram.')


if __name__ == '__main__':
    main()
