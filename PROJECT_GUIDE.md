# CNC Perception — Complete Project Guide

This guide explains **why** each step exists, the **mathematics** behind it, and **exact commands** to run scripts one at a time. Robot integration (Dobot MG400) comes later; this phase is camera + bed coordinates + workpiece pose only.

---

## 1. Problem statement

You need to know, in real time:

1. Where the **CNC bed** is relative to the camera (bed frame, origin at bottom-left, Z=0 on bed surface).
2. Where the **workpiece** is on that bed (position X,Y and orientation yaw).
3. Whether the workpiece is at the **target position** (e.g. 100 mm, 100 mm) within tolerance.

The Dobot robot will later place the part; the camera verifies placement. The camera does **not** watch the robot — it watches the bed and workpiece. The link to the robot is via calibrated transforms (Week 4).

---

## 2. Why detection failed before

Your workpiece is **50 mm × 50 mm × 10 mm**, but `workpiece_model.yaml` had **120 mm × 80 mm**. The contour filter rejected the real part because:

| Setting | Old value | Problem |
|---------|-----------|---------|
| `width_m` / `length_m` | 0.12 / 0.08 | Wrong aspect ratio and 3D model for solvePnP |
| `min_contour_area_px` | 2500 | Too large for a small part in the image |
| `min_area_ratio` | 0.02 | 50 mm part may occupy &lt; 2% of 1280×720 image |

These are now fixed in `config/workpiece_model.yaml`. If detection still fails, run **Step 04** (diagnostic) and tune `detection` thresholds.

---

## 3. Coordinate frames (critical)

```
cnc_bed_frame          Workpiece sitting on bed
    |                  Z_bed = 0     → bed surface
    |                  Z_bed = 0.01 m → top of 10 mm workpiece
    +-- workpiece_frame (center of top face)
    
camera_link            ELP camera optical frame (from usb_cam)
```

**Bed frame convention** (`cnc_bed.yaml`):

- Origin **(0, 0, 0)** = bottom-left corner of machining bed, on the top surface.
- **+X** = along bed length.
- **+Y** = along bed width.
- **+Z** = up from bed surface.

**Example:** Target center at (100 mm, 100 mm) on bed → `target_placement: x_m: 0.100, y_m: 0.100`.

---

## 4. Reference point on the bed — what to use?

### Recommended: ArUco marker at bed origin

| Option | Pros | Cons |
|--------|------|------|
| **ArUco at origin** (recommended) | Accurate 6D, automatic, small | One marker fixed to fixture |
| Manual 4-corner click | No marker on bed | Weaker 3D accuracy if camera is angled |
| Fiducial on workpiece | — | You explicitly have **no** markers on workpiece |

**Placement:** Print ArUco **DICT_4X4_50, id=0** (40 mm physical size). Glue at bed **bottom-left**, marker **center** at (0,0,0). Place **outside** the machining area or in a pocket that never gets cut.

Generate marker:

```bash
source install/setup.bash
ros2 run cnc_perception step00_generate_aruco_marker -- --output ~/aruco_bed_origin.png
```

Print so the black square measures **exactly** `marker_size_m` in `cnc_bed.yaml` (default 40 mm).

### Alternative: Step 02b (no ArUco)

Click four bed corners in the image (BL, BR, TR, TL). Good for **XY** on the bed plane; full 3D TF is approximate if the camera is tilted.

---

## 5. Mathematical logic

### 5.1 Camera intrinsics (you already calibrated)

Pinhole model:

\[
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}, \quad
\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
\]

Stored in `/camera_info` and `camera_info.yaml`.

### 5.2 Workpiece 6D pose (markerless)

1. Detect quadrilateral contour → 4 image points \((u_i, v_i)\).
2. Known 3D corners of top face in object frame (50 mm square, Z=0 on top face).
3. **solvePnP** finds rotation \(\mathbf{R}\) and translation \(\mathbf{t}\) such that:

\[
\mathbf{X}_c = \mathbf{R}\,\mathbf{X}_{obj} + \mathbf{t} = \mathbf{T}_{camera \leftarrow workpiece}\,\mathbf{X}_{obj}
\]

4. Published as TF `camera_link` → `workpiece_frame` and `/workpiece/actual_pose`.

Uses **IPPE** flag for planar objects (accurate for flat stock).

### 5.3 Bed frame from ArUco

ArUco at origin gives \(\mathbf{T}_{camera \leftarrow bed}\). Saved to `bed_calibration.yaml` and published as static TF:

\[
\mathbf{T}_{camera \leftarrow bed}
\]

### 5.4 Workpiece in bed coordinates

\[
\mathbf{T}_{bed \leftarrow workpiece} = \mathbf{T}_{bed \leftarrow camera}\,\mathbf{T}_{camera \leftarrow workpiece}
\]

where \(\mathbf{T}_{bed \leftarrow camera} = \left(\mathbf{T}_{camera \leftarrow bed}\right)^{-1}\) from TF.

Position in mm:

\[
X_{mm} = 1000 \cdot T_{bed \leftarrow workpiece}(0,3),\quad
Y_{mm} = 1000 \cdot T_{bed \leftarrow workpiece}(1,3)
\]

Top face height should be near \(Z = 0.01\,\text{m}\) when flat on bed.

### 5.5 Placement check

Target \(\mathbf{T}^*\); measured center \((X, Y, Z)\), yaw \(\psi\):

\[
\Delta X = X - X^*,\quad \Delta Y = Y - Y^*,\quad \Delta Z = Z - t_{workpiece},\quad \Delta\psi = \psi - \psi^*
\]

**CORRECT POSITION** if all \(|\Delta|\) within tolerances in `cnc_bed.yaml`.

### 5.6 Later: Robot base link (Week 4)

You will add \(\mathbf{T}_{base \leftarrow bed}\) (measured or calibrated). Then target in robot coordinates:

\[
\mathbf{T}_{base \leftarrow target} = \mathbf{T}_{base \leftarrow bed}\,\mathbf{T}_{bed \leftarrow target}
\]

The camera never sees the robot; it only measures \(\mathbf{T}_{bed \leftarrow workpiece}\). The robot uses its commanded pose plus this correction.

---

## 6. Configuration files (edit before running)

### `config/cnc_bed.yaml`

```yaml
cnc_bed:
  length_m: 0.250    # MEASURE your bed length (X)
  width_m: 0.180     # MEASURE your bed width (Y)

target_placement:
  x_m: 0.100         # Target X in meters (100 mm)
  y_m: 0.100         # Target Y in meters
```

### `config/workpiece_model.yaml`

Already set to 50×50×10 mm. Tune `detection` if Step 04 shows rejections.

### `config/bed_calibration.yaml`

Written by Step 02 or 02b. Do not edit by hand initially.

---

## 7. Step-by-step scripts (run in order)

Build once:

```bash
cd ~/cnc_pose_ws
source /opt/ros/jazzy/setup.bash   # or humble
colcon build
source install/setup.bash
```

Each step uses **separate terminals** unless noted.

---

### Step 0 — Generate ArUco marker (one-time)

```bash
ros2 run cnc_perception step00_generate_aruco_marker -- --output ~/aruco_bed_origin.png
```

Print at 40 mm (or your `marker_size_m`). Mount at bed origin.

---

### Step 1 — Verify camera

**Terminal A** — camera:

```bash
source install/setup.bash
ros2 run usb_cam usb_cam_node_exe --ros-args \
  --params-file install/cnc_perception/share/cnc_perception/config/camera_params.yaml
```

**Terminal B** — checker:

```bash
source install/setup.bash
ros2 run cnc_perception step01_verify_camera
```

**Expect:** Messages showing image size and fx, fy, cx, cy.

---

### Step 2 — Calibrate bed origin (ArUco)

Mount camera. Place ArUco at bed (0,0,0). Camera and Step 1 running.

```bash
source install/setup.bash
ros2 run cnc_perception step02_calibrate_bed_origin
```

**Expect:** `Bed calibration saved to .../bed_calibration.yaml`

**If no ArUco:** use Step 2b instead:

```bash
ros2 run cnc_perception step02b_calibrate_bed_corners
```

Click BL → BR → TR → TL, press ENTER.

---

### Step 3 — Publish bed TF

```bash
source install/setup.bash
ros2 run cnc_perception step03_publish_bed_tf
```

Keep running. Publishes `camera_link` → `cnc_bed_frame`.

---

### Step 4 — Debug workpiece detection

Camera running. Place **50×50 mm** workpiece on bed.

```bash
source install/setup.bash
ros2 run cnc_perception step04_debug_workpiece_detection
```

**Expect:** Images in `/tmp/cnc_perception_debug/`. Terminal logs top rejection reasons if not detected.

**Tuning tips:**

- Low contrast → lower `canny_low`, increase `adaptive_c`
- Part too small in image → lower `min_contour_area_px` and `min_area_ratio`
- Square part → `aspect_ratio_tolerance: 0.25` is OK

---

### Step 5 — Full pose in bed frame + RViz markers

**Terminals:** camera (A), step03 (B), step05 (C).

```bash
source install/setup.bash
ros2 run cnc_perception step05_workpiece_pose_bed_frame
```

**Publishes:**

| Topic | Meaning |
|-------|---------|
| `/workpiece/actual_pose` | Pose in camera frame |
| `/workpiece/pose_in_bed_frame` | Pose in bed frame |
| `/workpiece/bed_coordinates` | Human-readable X,Y,Z mm |
| `/cnc_bed/markers` | Bed outline, origin, target |
| `/workpiece/markers` | Workpiece box |
| `/workpiece/debug_image` | Corners overlay |

---

### Step 6 — CORRECT / NOT CORRECT

With step05 running:

```bash
source install/setup.bash
ros2 run cnc_perception step06_check_placement
```

**Publishes:** `/workpiece/placement_status` — e.g. `CORRECT POSITION` or `NOT CORRECT POSITION` with deltas.

---

## 8. RViz setup (first time — save your layout)

Open RViz manually as you requested:

```bash
source install/setup.bash
rviz2
```

### 8.1 Global settings

1. **Fixed Frame** → `cnc_bed_frame` (after Step 3 is running).
   - If TF missing, use `camera_link` until Step 3 runs.

### 8.2 Add displays (click Add → By display type)

| # | Display type | Topic / setting | Purpose |
|---|--------------|-----------------|---------|
| 1 | **TF** | Show Names, Show Axes | Bed origin, camera, workpiece frames |
| 2 | **Grid** | Plane=XY, Reference Frame=`cnc_bed_frame` | Bed plane |
| 3 | **MarkerArray** | `/cnc_bed/markers` | Bed outline, red origin sphere, target cylinder |
| 4 | **MarkerArray** | `/workpiece/markers` | Workpiece 50 mm cube |
| 5 | **MarkerArray** | `/workpiece/placement_markers` | CORRECT / NOT CORRECT text |
| 6 | **Pose** | `/workpiece/pose_in_bed_frame` | Workpiece pose on bed |
| 7 | **Image** | `/workpiece/debug_image` | Detection overlay for supervisor |
| 8 | **RobotModel** (optional) | Description file → `urdf/cnc_bed.urdf` | Static bed mesh |
| 9 | **RobotModel** (optional) | `urdf/workpiece.urdf` + TF `workpiece_frame` | Requires robot_state_publisher |

### 8.3 Save config

**File → Save Config As** → e.g. `~/cnc_pose_ws/my_supervisor_view.rviz`

Later: `rviz2 -d ~/cnc_pose_ws/my_supervisor_view.rviz`

### 8.4 What your supervisor should see

- Gray **bed plate** with white border.
- **Red sphere** at origin (0,0,0).
- **Green cylinder** at target (100,100) mm.
- **Blue semi-transparent cube** on the workpiece.
- Large text: **CORRECT POSITION** or **NOT CORRECT POSITION**.
- Side panel: debug camera image with green contour.

---

## 9. URDF and STL workpiece

- `urdf/cnc_bed.urdf` — simple box; edit sizes to match `cnc_bed.yaml`.
- `urdf/workpiece.urdf` — placeholder 50 mm cube; replace with STL when ready:

  1. Export STL from CAD to `meshes/workpiece.stl`.
  2. Update URDF to use `<mesh filename="package://cnc_perception/meshes/workpiece.stl"/>`.
  3. Run `robot_state_publisher` with `workpiece.urdf` and fixed frame `workpiece_frame`.

For now, **MarkerArray** (`/workpiece/markers`) updates in real time without URDF.

---

## 10. Troubleshooting

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| No workpiece detected | Wrong dimensions in YAML | Confirm 0.05×0.05×0.01; run Step 04 |
| No workpiece detected | Low contrast | Improve lighting; tune Canny/adaptive |
| No workpiece detected | Part too small in image | Lower camera or lower `min_area_ratio` |
| Waiting for cnc_bed_frame TF | Step 3 not running | Run `step03_publish_bed_tf` |
| ArUco not found | Wrong dictionary/id/size | Regenerate marker; check `cnc_bed.yaml` |
| Pose jumps | Motion blur | Fixed exposure in `camera_params.yaml` |
| Z not ≈ 10 mm | Bad calibration or part tilted | Recalibrate after mount; check part flat |

---

## 11. Recalibration after fixing camera on mount

You already calibrated intrinsics before mounting. After mounting:

1. **Intrinsics** — Recalibrate if focus/resolution changed (`camera_calibration`).
2. **Bed extrinsics** — **Always** rerun Step 02 + Step 03 after mount (camera pose relative to bed changed).

---

## 12. Script summary

| Script | Command | Purpose |
|--------|---------|---------|
| step00 | `ros2 run cnc_perception step00_generate_aruco_marker` | Print bed origin marker |
| step01 | `ros2 run cnc_perception step01_verify_camera` | Check stream + intrinsics |
| step02 | `ros2 run cnc_perception step02_calibrate_bed_origin` | ArUco bed calibration |
| step02b | `ros2 run cnc_perception step02b_calibrate_bed_corners` | Manual corner calibration |
| step03 | `ros2 run cnc_perception step03_publish_bed_tf` | Publish bed TF |
| step04 | `ros2 run cnc_perception step04_debug_workpiece_detection` | Detection diagnostics |
| step05 | `ros2 run cnc_perception step05_workpiece_pose_bed_frame` | 6D pose in bed frame |
| step06 | `ros2 run cnc_perception step06_check_placement` | CORRECT / NOT CORRECT |

**Launch files** (`perception.launch.py`) remain for later convenience. Use scripts until each step passes.

---

## 13. Next phase (robot — not implemented yet)

1. Measure \(\mathbf{T}_{base \leftarrow bed}\).
2. Convert target bed position to Dobot base coordinates.
3. After robot places part, Step 06 confirms placement.
4. If NOT CORRECT, compute small relative move in base frame (Dobot Python SDK).

---

## 14. Quick test today (after rebuild)

```bash
cd ~/cnc_pose_ws && colcon build && source install/setup.bash
```

Edit `config/cnc_bed.yaml` with real bed size and target.

Terminal 1: `ros2 run usb_cam usb_cam_node_exe --ros-args --params-file install/cnc_perception/share/cnc_perception/config/camera_params.yaml`

Terminal 2: `ros2 run cnc_perception step04_debug_workpiece_detection`

Place 50×50 mm part on bed — check `/tmp/cnc_perception_debug/` for green contour.

When Step 04 detects reliably, proceed Steps 2→3→5→6 with RViz.
