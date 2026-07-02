# CNC Operator GUI

Simple desktop panel for non-technical operators. **Does not modify** any files under `src/cnc_perception/`.

## What it does

- **Build Workspace** — runs `colcon build --packages-select cnc_perception`
- **1. Run Camera** — camera + image rectification
- **2. Calibrate CNC Bed** — ArUco bed calibration
- **3. Publish Transforms** — bed TF
- **4. Detect & Estimate Pose** — step05
- **5. Verify Workpiece Placement** — step06 with green/red status light and dX, dY, dYaw
- **Open RViz Visualization**
- **Target X / Y / Yaw** — saved to `operator_gui/runtime/cnc_bed_gui.yaml` only

Runtime copies of configs live in `operator_gui/runtime/` so your package configs stay untouched.

## Start the GUI

```bash
cd /home/d/Downloads/dd
chmod +x operator_gui/launch_gui.sh
./operator_gui/launch_gui.sh
```

`launch_gui.sh` sources ROS and the workspace automatically. You do **not** need to edit `.bashrc`, but optional convenience:

```bash
alias cnc-gui='/home/d/Downloads/dd/operator_gui/launch_gui.sh'
```

## Desktop shortcut (Ubuntu 24.04)

```bash
cd /home/d/Downloads/dd
chmod +x operator_gui/install_desktop_shortcut.sh
./operator_gui/install_desktop_shortcut.sh
```

This places **CNC Operator GUI** on your desktop. If prompted, choose **Trust and Launch**.

Each button shows **Running...** or **Not running** on the right while you operate the panel.

## Suggested order

1. Build Workspace (first time or after code changes)
2. Run Camera
3. Calibrate CNC Bed (only when setting up / re-calibrating)
4. Publish Transforms
5. Detect & Estimate Pose
6. Verify Workpiece Placement
7. Open RViz Visualization (any time)

Press a running button again to stop that step.

## Target position

Enter values in **mm** and **degrees**, click **Apply Target Position**, then restart steps 4 and 5 if they are already running.

## Requirements

- ROS 2 (Jazzy/Humble/Iron)
- Python 3 with `tkinter` and `pyyaml`
- Built workspace (`install/setup.bash`)
