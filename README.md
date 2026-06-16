# Markerless 6D Pose Estimation for CNC Workpieces

ROS 2 Jazzy workspace for estimating the pose of a **plain rectangular workpiece** (no ArUco or fiducials) using an overhead ELP-USB GS1200P01 global shutter camera.

## Architecture

```text
ELP USB Camera (V4L2)
        |
   usb_cam_node  -->  /image_raw, /camera_info
        |
 pose_estimator_node
   |         |
   |         +--> TF: camera_link -> workpiece_frame
   |         +--> /workpiece/markers (RViz box + outline)
   +--> /workpiece/actual_pose (geometry_msgs/PoseStamped)
        |
     rviz2 (optional, perception.rviz)
```

Detection pipeline:

1. Grayscale conversion and adaptive thresholding (lighting robustness)
2. Canny edge detection and morphological closing
3. Contour extraction with aspect-ratio and area filtering
4. Quadrilateral fitting to recover the 4 top-face corners
5. `cv2.solvePnP` against known physical dimensions
6. Pose publication via TF2 and `/workpiece/actual_pose`

## Workspace layout

```text
cnc_pose_ws/
└── src/
    └── cnc_perception/
        ├── config/
        │   ├── camera_params.yaml      # usb_cam V4L2 parameters
        │   ├── camera_info.yaml        # placeholder intrinsics (replace after calibration)
        │   ├── workpiece_model.yaml    # physical dimensions + detection tuning
        │   └── perception.rviz         # RViz layout (TF, pose, markers, debug image)
        ├── launch/
        │   ├── perception.launch.py    # camera + pose estimator + RViz
        │   └── rviz.launch.py          # RViz only
        └── cnc_perception/
            ├── contour_detector.py
            ├── pose_estimator_node.py
            ├── pose_solver.py
            └── workpiece_config.py
```

## Prerequisites

- ROS 2 Jazzy
- `usb_cam` package
- OpenCV and cv_bridge

```bash
sudo apt update
sudo apt install ros-jazzy-usb-cam ros-jazzy-cv-bridge ros-jazzy-image-transport \
  ros-jazzy-tf2-ros ros-jazzy-camera-calibration ros-jazzy-rviz2 \
  ros-jazzy-rviz-default-plugins v4l-utils python3-yaml
```

## Build

```bash
cd cnc_pose_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Camera setup (ELP-USB GS1200P01)

1. Connect the camera and verify the device:

```bash
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

2. Confirm `1280x720 @ 60 FPS` is listed. If MJPEG is required for 60 Hz, keep `pixel_format: "mjpeg"` in `camera_params.yaml`.

3. Inspect V4L2 controls and tune exposure/white balance for stable bench lighting:

```bash
v4l2-ctl -d /dev/video0 -L
```

4. Update `video_device` if the camera is not `/dev/video0`.

## Camera calibration

Replace `config/camera_info.yaml` with a calibrated file:

```bash
ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.025 \
  --ros-args -r image:=/image_raw -r camera:=/camera
```

Accurate intrinsics are required for reliable 6D pose output.

## Workpiece configuration

Edit `config/workpiece_model.yaml`:

| Parameter | Description |
|-----------|-------------|
| `workpiece.width_m` | Physical width (meters) |
| `workpiece.length_m` | Physical length (meters) |
| `workpiece.thickness_m` | Stock thickness (meters) |
| `detection.*` | Thresholding, Canny, contour filters |
| `template.enabled` | Optional grayscale template matching fallback |

Object frame convention:

- Origin at the center of the top face
- +X along width, +Y along length, +Z outward from the top face

## Run

```bash
source install/setup.bash

# Camera + pose estimator + RViz (default)
ros2 launch cnc_perception perception.launch.py

# Headless (no RViz)
ros2 launch cnc_perception perception.launch.py use_rviz:=false

# RViz only (when perception is already running)
ros2 launch cnc_perception rviz.launch.py
```

Optional arguments:

```bash
ros2 launch cnc_perception perception.launch.py \
  video_device:=/dev/video2 \
  workpiece_config_path:=/path/to/custom_workpiece_model.yaml \
  use_rviz:=true
```

## RViz visualization

The bundled `perception.rviz` config shows:

| Display | Topic / source | Purpose |
|---------|----------------|---------|
| **TF** | `camera_link`, `workpiece_frame` | Camera and workpiece coordinate frames |
| **Workpiece Pose** | `/workpiece/actual_pose` | 6D pose arrow in `camera_link` |
| **Workpiece Model** | `/workpiece/markers` | Semi-transparent stock cube, top-face outline, local axes |
| **Detection Debug** | `/workpiece/debug_image` | Contour corners overlaid on the camera image |
| **Camera Raw** | `/image_raw` | Raw stream (disabled by default; enable in Displays panel) |

Fixed frame is `camera_link`. Orbit the 3D view to inspect pose relative to the overhead camera.

## Topics and TF

| Name | Type | Description |
|------|------|-------------|
| `/image_raw` | `sensor_msgs/Image` | Camera stream |
| `/camera_info` | `sensor_msgs/CameraInfo` | Intrinsics from usb_cam |
| `/workpiece/actual_pose` | `geometry_msgs/PoseStamped` | Estimated pose in camera frame |
| `/workpiece/markers` | `visualization_msgs/MarkerArray` | RViz workpiece box and outline |
| `/workpiece/debug_image` | `sensor_msgs/Image` | Annotated detection overlay |
| `camera_link` → `workpiece_frame` | TF | 6D transform from camera to workpiece |

Inspect output:

```bash
ros2 topic echo /workpiece/actual_pose
ros2 run tf2_ros tf2_echo camera_link workpiece_frame
```

## Tuning tips

- **Low contrast workpiece vs. bed**: Increase `adaptive_c` or adjust `canny_low` / `canny_high`.
- **False detections**: Tighten `aspect_ratio_tolerance`, `min_solidity`, and area ratio bounds.
- **Motion blur**: Disable auto-exposure in `camera_params.yaml` and reduce exposure time.
- **Jittery pose**: Increase `pose.pose_smoothing_alpha` (closer to 1.0 = more smoothing).

## Limitations

- Assumes a rectangular top face with known dimensions.
- Strong glare, shadows, or clutter can break contour matching.
- Roll/pitch/yaw accuracy depends on calibration quality and edge sharpness.
- Template matching is optional and requires a reference edge image.

## License

Apache-2.0
# dd
# dd
