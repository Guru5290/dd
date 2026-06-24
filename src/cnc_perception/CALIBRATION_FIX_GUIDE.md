# CNC Bed Calibration Fix — Guide

This guide explains what was wrong with the camera→CNC bed transformation, what was fixed in the codebase, and the exact steps to get correct workpiece pose in RViz.

---

## What was wrong

### 1. Incorrect bed calibration math
The original `bed_calibration.py` applied a `flip_z` matrix that inverted the bed Z axis. That made workpieces appear **tilted or floating** above the RViz grid even when lying flat on the bed.

### 2. Missing `camera_optical_frame` transform
OpenCV `solvePnP` returns poses in the **camera optical frame** (+X right, +Y down, +Z forward). The bed calibration and pose pipeline mixed optical and `camera_link` frames without the standard ROS rotation between them.

### 3. Distortion applied twice
You correctly switched to `/image_rect_color` (undistorted), but the code still used raw distortion coefficients from `/camera_info`. That skewed 3D pose and reprojection.

### 4. Bad launch remapping
`image_proc_pipeline.launch.py` remapped `image_rect_color` directly from `usb_cam`, **bypassing** `image_proc`. Rectification was not actually applied in the pipeline.

### 5. Shadow false detections
Grayscale contour detection often merged the workpiece shadow with the part. The cube in RViz was offset because the 2D contour included shadow area.

### 6. Ghost cube in RViz
When detection failed, the last pose and markers were kept. The cube stayed visible after removing the workpiece.

---

## What was fixed

| Component | Fix |
|-----------|-----|
| `bed_calibration.py` | Proper chain: optical → marker → bed → camera_link; removed `flip_z` |
| `camera_frames.py` | Standard `camera_link` ↔ `camera_optical_frame` rotation |
| `step03_publish_bed_tf.py` | Publishes both `camera_link→cnc_bed_frame` and `camera_link→camera_optical_frame` |
| `step05_workpiece_pose_bed_frame.py` | Rectified intrinsics, optical→link→bed chain, smoothing, reprojection gate, bed bounds, DELETE markers on loss, color debug image |
| `contour_detector.py` | Interior/exterior contrast filter rejects shadow-heavy contours |
| `image_proc_pipeline.launch.py` | usb_cam → image_proc → `/image_rect_color` |
| `cnc_bed.yaml` | Marker yaw/flip options, fixed target placement |
| `perception.rviz` | Fixed frame = `cnc_bed_frame`, bed markers, bed-frame pose |

---

## Coordinate conventions (Genmitsu 3018)

```
cnc_bed_frame:
  Origin (0,0,0) = bottom-left corner of bed (as in your setup photo)
  +X = 0.284 m along bottom edge (BL → BR)
  +Y = 0.180 m along left edge (BL → back)
  +Z = up; top surface of bed is Z = 0

Workpiece pose:
  Origin = center of top face
  When flat on bed: Z ≈ thickness (10 mm for 50×50×10 mm stock)
```

---

## Prerequisites

- ROS 2 Jazzy
- Workspace: `~/cnc_pose_ws_fix/dd` (or your extracted `dd` folder)
- Camera calibrated (`config/camera_info.yaml`)
- ArUco marker printed: `step00_generate_aruco_marker.py` → DICT_4X4_50 id=0, 48 mm

---

## Step-by-step commands

Open **4–5 terminals**. Source the workspace in each:

```bash
cd ~/cnc_pose_ws_fix/dd
source /opt/ros/jazzy/setup.bash
colcon build --packages-select cnc_perception
source install/setup.bash
```

### Terminal 1 — Camera + rectification

```bash
ros2 launch cnc_perception image_proc_pipeline.launch.py
```

If the camera is not on `/dev/video0`:
```bash
ros2 launch cnc_perception image_proc_pipeline.launch.py video_device:=/dev/video2
```

**Diagnose in order** (while launch is running in another terminal):

```bash
# 1) Is the camera driver publishing?
ros2 topic list | grep image
ros2 topic hz /image_raw

# 2) Is image_proc running?
ros2 node list | grep -E 'camera|rectify|image_proc'

# 3) Rectified color topic (after fix: RectifyNode -> /image_rect_color)
ros2 topic hz /image_rect_color
```

If `/image_raw` has no data, fix the camera first (see troubleshooting below).  
If `/image_raw` works but `/image_rect_color` does not:
```bash
ros2 node list | grep rectifier
ros2 topic hz /camera_info_raw
ros2 topic hz /camera_info
```
The launch file now uses **`rectify_image_node`** (OpenCV) instead of `image_proc`. Rebuild and restart the launch.

Verify rectified camera info (distortion should be zero):
```bash
ros2 topic echo /camera_info --once
```

Optional color preview:
```bash
ros2 run rqt_image_view rqt_image_view /image_rect_color
```

### Terminal 2 — Bed calibration (ArUco at origin)

1. Place ArUco id=0 so its **center** is at the **bottom-left** bed corner.
2. Align marker +X with bed +X (toward bottom-right) if possible.
3. Ensure marker is flat on the bed surface.

```bash
ros2 run cnc_perception step02_calibrate_bed_origin
```

Wait for: `Bed calibration saved to .../bed_calibration.yaml`

### Terminal 3 — Publish bed TF (keep running)

```bash
ros2 run cnc_perception step03_publish_bed_tf
```

Verify TF:
```bash
ros2 run tf2_ros tf2_echo camera_link cnc_bed_frame
ros2 run tf2_ros tf2_echo camera_link camera_optical_frame
```

### Terminal 4 — RViz

```bash
ros2 run rviz2 rviz2 -d $(ros2 pkg prefix cnc_perception)/share/cnc_perception/config/perception.rviz
```

Check:
- **Fixed Frame** = `cnc_bed_frame`
- White bed outline: 284 mm × 180 mm
- Red sphere at origin (bottom-left)
- TF axes for `camera_link`, `camera_optical_frame`, `cnc_bed_frame`

### Terminal 5 — Workpiece pose

Place the 50×50×10 mm workpiece **flat on the bed** inside the white outline.

```bash
ros2 run cnc_perception step05_workpiece_pose_bed_frame
```

Expected when flat and centered:
- Terminal: `Z_top≈10.0 mm`, `tilt<10 deg`
- RViz: blue cube sits on bed plate, not floating
- Removing workpiece → cube **disappears** within ~3 frames

### Terminal 6 (optional) — Placement check

```bash
ros2 run cnc_perception step06_check_placement
```

---

## Tuning if pose is still rotated

Edit `config/cnc_bed.yaml`:

```yaml
reference_marker:
  marker_to_bed_yaw_deg: 0.0   # try 90.0 or -90.0 if X/Y are swapped
  flip_marker_y: true          # set false only if Y axis is inverted
```

Then **re-run step02 and step03**.

---

## Tuning detection (shadows)

Edit `config/workpiece_model.yaml`:

```yaml
detection:
  canny_low: 70
  canny_high: 150
  min_contour_area_px: 2000
  max_contour_area_px: 15000
```

Use step04 to debug:
```bash
ros2 run cnc_perception step04_debug_workpiece_detection
```

Improve lighting: diffuse light from above reduces shadows on the aluminum bed.

---

## Verify bed dimensions in RViz

With step03 + RViz running (no workpiece needed):
- Bottom-left corner = red sphere at (0,0,0)
- Bottom-right corner of white outline ≈ X = 0.284 m
- Top-left corner ≈ Y = 0.180 m

---

## Topic reference

| Topic | Description |
|-------|-------------|
| `/image_rect_color` | Undistorted color image (use this) |
| `/camera_info` | Rectified intrinsics (D=0) |
| `/workpiece/pose_in_bed_frame` | 6D pose in bed coordinates |
| `/workpiece/markers` | RViz cube in `cnc_bed_frame` |
| `/cnc_bed/markers` | Bed outline + axes |
| `/workpiece/debug_image` | Color overlay with detection |

---

## Quick troubleshooting

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| Cube tilted / floating | Old calibration | Re-run step02 + step03 |
| Wrong X/Y scale | Marker size wrong in yaml | Set `marker_size_m: 0.048` |
| Cube outside bed when on bed | Yaw misalignment | Tune `marker_to_bed_yaw_deg` |
| Cube offset from part | Shadow in contour | Improve lighting; tune canny thresholds |
| Ghost cube after removal | Old step05 | Use updated step05 (DELETE markers) |
| `Waiting for cnc_bed_frame TF` | step03 not running | Start step03 |
| `/image_rect_color` not published | usb_cam dead or wrong image_proc node | See camera troubleshooting below |

---

## Camera troubleshooting (`/image_rect_color` missing)

**Step A — Find the camera device**
```bash
v4l2-ctl --list-devices
ls -l /dev/video*
```
Use the device that lists your ELP camera (often `/dev/video0` or `/dev/video2`).

**Step B — Test permissions**
```bash
groups   # should include 'video'
# if not:
sudo usermod -aG video $USER
# then log out and back in
```

**Step C — While launch is running, check `/image_raw` first**
```bash
ros2 topic hz /image_raw
```
- **No `/image_raw`**: usb_cam failed. Read the launch terminal for errors (`Cannot open device`, wrong format, etc.).
- **`/image_raw` OK, no `/image_rect_color`**: rebuild after pull, ensure `ros-jazzy-image-proc` is installed, check `ros2 node list` for `rectify`.

**Step D — Launch with correct device**
```bash
ros2 launch cnc_perception image_proc_pipeline.launch.py video_device:=/dev/videoX
```

**Step E — Manual fallback (two terminals)**
```bash
# Terminal A
ros2 run usb_cam usb_cam_node_exe --ros-args \
  --params-file $(ros2 pkg prefix cnc_perception)/share/cnc_perception/config/camera_params.yaml

# Terminal B
ros2 run image_proc rectify_node --ros-args \
  -r image:=/image_raw -r camera_info:=/camera_info -r image_rect:=/image_rect_color
```

---

## Rebuild after any code change

```bash
cd ~/cnc_pose_ws_fix/dd
colcon build --packages-select cnc_perception
source install/setup.bash
```
