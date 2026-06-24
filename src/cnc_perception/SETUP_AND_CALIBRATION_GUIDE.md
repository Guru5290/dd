# CNC Perception — Setup and Calibration Guide (Genmitsu 3018-PRO + ELP-USBGS1200P01)

Camera on the **left** (gooseneck, tilted down), bed labels **+X right / +Y back**, **open metal bed** (no corner jig).

Your saved `bed_calibration.yaml` is kept as-is. Re-run step02 only when you move the camera or change marker orientation.

---

## 1. Bed coordinates (same as labels on the machine)

Stand in **front** of the CNC (loading position):

```
                 BACK (+Y = 180 mm)
         ┌─────────────────────────────┐
         │                             │
  LEFT   │         METAL BED           │  RIGHT
 (camera│                             │
  side) │                             │
         └─────────────────────────────┘
                FRONT (+X = 284 mm →)

  ORIGIN (0,0,0) = front-left corner (ArUco marker center at calibration)
```

| Axis | Direction |
|------|-----------|
| **+X** | Along front edge → **right** (284 mm) |
| **+Y** | Along left edge → **back** (180 mm) |
| **+Z** | Up from bed surface |

---

## 2. ArUco ID 0 — how to place it

Use the PNG from `step00_generate_aruco_marker.py` (48 mm print = `marker_size_m: 0.048`).

1. **Position:** Marker **center** on the **front-left** bed corner.
2. **Flat:** Tape marker flat on the bed surface (no curl).
3. **Orientation:** Place so it looks **upright in the camera image** (same as the PNG on screen):
   - One marker edge runs along the **front edge** of the bed (parallel to **+X**).

4. **Default yaml** (for next calibration):
   ```yaml
   marker_to_bed_yaw_deg: 0.0
   flip_marker_y: true
   ```
   If after step02 the RViz **red +X** does not point along the real front edge toward the right, try `marker_to_bed_yaw_deg: 90` or `180` and recalibrate.

**Tip:** Remove or cover the ArUco marker during workpiece detection if it keeps winning contour scoring (it is also a square).

---

## 3. What `bed_calibration.yaml` means

Example:

```yaml
translation_m: [-0.113, -0.323, 0.031]
rotation_xyzw: [0.551, 0.385, 0.593, -0.444]
```

### Translation (NOT camera height)

Position of the **bed origin** in **`camera_link`** (meters). The small Z (~3 cm) is **not** camera height — tilt is mostly in **rotation**.

### Rotation

How `cnc_bed_frame` axes orient relative to `camera_link`. Different directions in RViz are **expected**.

---

## 4. Run order (every session)

```bash
cd ~/dd   # or your workspace path
colcon build --packages-select cnc_perception
source install/setup.bash
```

| Step | Command |
|------|---------|
| 1 | `ros2 launch cnc_perception image_proc_pipeline.launch.py` |
| 2 | `ros2 run cnc_perception step03_publish_bed_tf` |
| 3 | `ros2 run cnc_perception step04_debug_workpiece_detection` |
| 4 | `ros2 run cnc_perception step05_workpiece_pose_bed_frame` |

Recalibrate only when camera or marker changes: step02 then step03.

---

## 5. Workpiece detection (50×50×10 mm)

### Lighting

Use **diffuse light from above**. Strong side light creates shadows on the aluminum bed that merge with the part and break contour detection.

### Verify step04 first

```bash
ros2 run cnc_perception step04_debug_workpiece_detection
```

Check `/tmp/cnc_perception_debug/frame_*.jpg`:

| Overlay | Meaning |
|---------|---------|
| **Green** box + corners | Detection OK — proceed to step05 |
| **Red** candidates, no green | Contours found but rejected — read terminal reasons |
| No quads at all | Weak edges — improve lighting or lower `canny_low` slightly |

Also view `/workpiece/debug_image` in RViz while step05 runs.

### Expected pose when flat on bed

| Quantity | Expected |
|----------|----------|
| Z in bed frame | ≈ **10 mm** (workpiece thickness) |
| Surface tilt | **< 35°** |
| Reprojection error | **< 10 px** |

### Do NOT tighten these (common mistake)

Avoid values like `max_contour_area_px: 15000` or `canny_low: 70` — they caused today's failures (part + shadow rejected).

Current defaults in `workpiece_model.yaml` are tuned for your setup. Only tune one parameter at a time using step04.

### If ArUco steals detection

Cover the marker during production runs, or enable a small ROI in `workpiece_model.yaml` to mask only the marker corner (not the whole left side).

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| step04: no detection | Shadows, dark part, over-tight yaml | Diffuse lighting; rebuild after config change |
| step04: red contours, `area > max` | Part + shadow merged | Better lighting; `max_area_ratio` already limits huge blobs |
| step05: tilt ~70° | Bad contour (shadow) passed briefly | Fix detection first; check debug image |
| Cube floats / wrong place | Bed calibration or marker yaw | Re-run step02 with correct marker orientation |
| Z ≈ 10 mm wrong | Bed TF or wrong contour | step04 green box must be on the part only |
