#!/usr/bin/env python3
"""Non-technical operator GUI for the CNC perception workspace."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

from config_manager import RuntimeConfigManager
from placement_listener import PlacementListener, PlacementStatus
from process_runner import ManagedProcess, find_ros_setup, run_build

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class CncOperatorGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title('CNC Perception Operator Panel')
        self.minsize(960, 620)
        self.geometry('1100x720')

        try:
            self.ros_setup = find_ros_setup()
        except FileNotFoundError as exc:
            messagebox.showerror('ROS not found', str(exc))
            raise SystemExit(1) from exc

        self.config_manager = RuntimeConfigManager(WORKSPACE_ROOT)
        self.processes: dict[str, ManagedProcess] = {
            key: ManagedProcess(key) for key in (
                'camera',
                'calibrate',
                'transforms',
                'pose',
                'verify',
                'rviz',
            )
        }
        self.placement_listener = PlacementListener(self._on_placement_status)
        self._build_running = False
        self._placement_active = False

        self._build_ui()
        self._load_target_fields()
        self.protocol('WM_DELETE_WINDOW', self._on_close)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=12)
        left.grid(row=0, column=0, sticky='nsew')
        left.columnconfigure(0, weight=1)

        self.action_buttons: dict[str, ttk.Button] = {}
        self.status_labels: dict[str, ttk.Label] = {}

        right = ttk.Frame(self, padding=12)
        right.grid(row=0, column=1, sticky='nsew')
        right.columnconfigure(0, weight=1)
        right.rowconfigure(4, weight=1)

        title = ttk.Label(left, text='CNC Perception Setup', font=('Segoe UI', 18, 'bold'))
        title.grid(row=0, column=0, sticky='w', pady=(0, 8))
        subtitle = ttk.Label(
            left,
            text='Press the buttons in order. Status on the right shows Running or Not running.',
            wraplength=460,
        )
        subtitle.grid(row=1, column=0, sticky='w', pady=(0, 12))

        build_row = ttk.Frame(left)
        build_row.grid(row=2, column=0, sticky='ew', pady=4)
        build_row.columnconfigure(0, weight=3)
        build_row.columnconfigure(1, weight=1)
        self.build_button = ttk.Button(build_row, text='Build Workspace', command=self._toggle_build)
        self.build_button.grid(row=0, column=0, sticky='ew', padx=(0, 8))
        self.build_status_label = ttk.Label(build_row, text='Not running', foreground='#6b7280')
        self.build_status_label.grid(row=0, column=1, sticky='e')

        self.button_specs = [
            ('camera', '1. Run Camera', self._toggle_camera),
            ('calibrate', '2. Calibrate CNC Bed', self._toggle_calibrate),
            ('transforms', '3. Publish Transforms', self._toggle_transforms),
            ('pose', '4. Detect && Estimate Pose', self._toggle_pose),
            ('verify', '5. Verify Workpiece Placement', self._toggle_verify),
            ('rviz', 'Open RViz Visualization', self._toggle_rviz),
        ]
        row = 3
        for key, label, handler in self.button_specs:
            row_frame = ttk.Frame(left)
            row_frame.grid(row=row, column=0, sticky='ew', pady=4)
            row_frame.columnconfigure(0, weight=3)
            row_frame.columnconfigure(1, weight=1)
            button = ttk.Button(row_frame, text=label, command=handler)
            button.grid(row=0, column=0, sticky='ew', padx=(0, 8))
            status_label = ttk.Label(row_frame, text='Not running', foreground='#6b7280')
            status_label.grid(row=0, column=1, sticky='e')
            self.action_buttons[key] = button
            self.status_labels[key] = status_label
            row += 1

        target_frame = ttk.LabelFrame(left, text='Target Position', padding=10)
        target_frame.grid(row=row, column=0, sticky='ew', pady=(16, 0))
        target_frame.columnconfigure(1, weight=1)

        ttk.Label(target_frame, text='Target X (mm)').grid(row=0, column=0, sticky='w', pady=3)
        ttk.Label(target_frame, text='Target Y (mm)').grid(row=1, column=0, sticky='w', pady=3)
        ttk.Label(target_frame, text='Target Yaw (deg)').grid(row=2, column=0, sticky='w', pady=3)

        self.target_x_var = tk.StringVar()
        self.target_y_var = tk.StringVar()
        self.target_yaw_var = tk.StringVar()
        ttk.Entry(target_frame, textvariable=self.target_x_var).grid(row=0, column=1, sticky='ew', padx=(8, 0))
        ttk.Entry(target_frame, textvariable=self.target_y_var).grid(row=1, column=1, sticky='ew', padx=(8, 0))
        ttk.Entry(target_frame, textvariable=self.target_yaw_var).grid(row=2, column=1, sticky='ew', padx=(8, 0))
        ttk.Button(target_frame, text='Apply Target Position', command=self._apply_target).grid(
            row=3, column=0, columnspan=2, sticky='ew', pady=(10, 0)
        )

        status_title = ttk.Label(right, text='Placement Status', font=('Segoe UI', 18, 'bold'))
        status_title.grid(row=0, column=0, sticky='w')

        self.status_hint = ttk.Label(
            right,
            text='Press button 5 (Verify Workpiece Placement) to activate the status light.',
            wraplength=420,
        )
        self.status_hint.grid(row=1, column=0, sticky='w', pady=(4, 12))

        indicator_frame = ttk.Frame(right, padding=8)
        indicator_frame.grid(row=2, column=0, pady=8)
        self.indicator_canvas = tk.Canvas(indicator_frame, width=140, height=140, highlightthickness=0)
        self.indicator_canvas.pack()
        self._indicator_id = self.indicator_canvas.create_oval(15, 15, 125, 125, fill='#9ca3af', outline='#6b7280', width=3)
        self.indicator_label = ttk.Label(indicator_frame, text='Waiting', font=('Segoe UI', 14, 'bold'))
        self.indicator_label.pack(pady=(8, 0))

        delta_frame = ttk.LabelFrame(right, text='Position Error (when not correct)', padding=10)
        delta_frame.grid(row=3, column=0, sticky='ew', pady=(8, 12))
        delta_frame.columnconfigure(1, weight=1)

        self.dx_var = tk.StringVar(value='—')
        self.dy_var = tk.StringVar(value='—')
        self.dyaw_var = tk.StringVar(value='—')
        ttk.Label(delta_frame, text='dX').grid(row=0, column=0, sticky='w', pady=2)
        ttk.Label(delta_frame, text='dY').grid(row=1, column=0, sticky='w', pady=2)
        ttk.Label(delta_frame, text='dYaw').grid(row=2, column=0, sticky='w', pady=2)
        ttk.Label(delta_frame, textvariable=self.dx_var, font=('Consolas', 12)).grid(row=0, column=1, sticky='w', padx=(10, 0))
        ttk.Label(delta_frame, textvariable=self.dy_var, font=('Consolas', 12)).grid(row=1, column=1, sticky='w', padx=(10, 0))
        ttk.Label(delta_frame, textvariable=self.dyaw_var, font=('Consolas', 12)).grid(row=2, column=1, sticky='w', padx=(10, 0))

        log_frame = ttk.LabelFrame(right, text='Activity Log', padding=8)
        log_frame.grid(row=4, column=0, sticky='nsew')
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_box = scrolledtext.ScrolledText(log_frame, height=16, wrap='word', state='disabled')
        self.log_box.grid(row=0, column=0, sticky='nsew')

    def _load_target_fields(self) -> None:
        x_mm, y_mm, yaw_deg = self.config_manager.read_target_mm_deg()
        self.target_x_var.set(f'{x_mm:.1f}')
        self.target_y_var.set(f'{y_mm:.1f}')
        self.target_yaw_var.set(f'{yaw_deg:.1f}')

    def _log(self, text: str) -> None:
        def _append() -> None:
            self.log_box.configure(state='normal')
            self.log_box.insert('end', text + '\n')
            self.log_box.see('end')
            self.log_box.configure(state='disabled')

        self.after(0, _append)

    def _set_step_status(self, key: str, running: bool) -> None:
        if key == 'build':
            label = self.build_status_label
        else:
            label = self.status_labels.get(key)
        if label is None:
            return
        if running:
            label.configure(text='Running...', foreground='#15803d')
        else:
            label.configure(text='Not running', foreground='#6b7280')

    def _set_button_running(self, key: str, running: bool) -> None:
        self._set_step_status(key, running)
        button = self.action_buttons.get(key)
        if button is None:
            return
        style = 'Accent.TButton' if running else 'TButton'
        try:
            button.configure(style=style)
        except tk.TclError:
            pass

    def _start_process(self, key: str, command: str) -> None:
        process = self.processes[key]

        def on_output(line: str) -> None:
            self._log(line)

        def on_exit(code: int) -> None:
            def _finish() -> None:
                self._set_button_running(key, False)
                if key == 'verify':
                    self._placement_active = False
                    self.placement_listener.stop()
                    self._set_indicator(None, 'Verification stopped')
                self._log(f'[{key}] exited with code {code}')

            self.after(0, _finish)

        process.start(
            command=command,
            workspace_root=WORKSPACE_ROOT,
            ros_setup=self.ros_setup,
            on_output=on_output,
            on_exit=on_exit,
        )
        self._set_button_running(key, True)

    def _stop_process(self, key: str) -> None:
        self.processes[key].stop()
        self._set_button_running(key, False)
        if key == 'verify':
            self._placement_active = False
            self.placement_listener.stop()
            self._set_indicator(None, 'Verification stopped')

    def _toggle_key(self, key: str, command: str) -> None:
        if self.processes[key].running:
            self._stop_process(key)
            return
        self._start_process(key, command)

    def _toggle_build(self) -> None:
        if self._build_running:
            return
        self._build_running = True
        self._set_step_status('build', True)
        self.build_button.configure(state='disabled')
        self._log('Building workspace...')

        def on_done(code: int) -> None:
            def _finish() -> None:
                self._build_running = False
                self._set_step_status('build', False)
                self.build_button.configure(state='normal')
                if code == 0:
                    self._log('Build finished successfully.')
                    messagebox.showinfo('Build complete', 'Workspace built successfully.')
                else:
                    self._log(f'Build failed with code {code}.')
                    messagebox.showerror('Build failed', f'Build failed with exit code {code}.')

            self.after(0, _finish)

        run_build(WORKSPACE_ROOT, self.ros_setup, self._log, on_done)

    def _toggle_camera(self) -> None:
        self._toggle_key('camera', 'ros2 launch cnc_perception image_proc_pipeline.launch.py')

    def _toggle_calibrate(self) -> None:
        bed = self.config_manager.runtime_bed_config
        out = self.config_manager.runtime_bed_calibration
        cmd = (
            'ros2 run cnc_perception step02_calibrate_bed_origin --ros-args '
            f'-p bed_config_path:={bed} '
            f'-p output_path:={out}'
        )
        self._toggle_key('calibrate', cmd)

    def _toggle_transforms(self) -> None:
        cal = self.config_manager.runtime_bed_calibration
        cmd = (
            'ros2 run cnc_perception step03_publish_bed_tf --ros-args '
            f'-p calibration_path:={cal}'
        )
        self._toggle_key('transforms', cmd)

    def _toggle_pose(self) -> None:
        bed = self.config_manager.runtime_bed_config
        cmd = (
            'ros2 run cnc_perception step05_workpiece_pose_bed_frame --ros-args '
            f'-p bed_config_path:={bed}'
        )
        self._toggle_key('pose', cmd)

    def _toggle_verify(self) -> None:
        if self.processes['verify'].running:
            self._stop_process('verify')
            return
        bed = self.config_manager.runtime_bed_config
        cmd = (
            'ros2 run cnc_perception step06_check_placement --ros-args '
            f'-p bed_config_path:={bed}'
        )
        self._start_process('verify', cmd)
        self._placement_active = True
        self.placement_listener.start()
        self._set_indicator(None, 'Waiting for placement status...')

    def _toggle_rviz(self) -> None:
        self._toggle_key('rviz', 'ros2 launch cnc_perception rviz.launch.py')

    def _apply_target(self) -> None:
        try:
            x_mm = float(self.target_x_var.get().strip())
            y_mm = float(self.target_y_var.get().strip())
            yaw_deg = float(self.target_yaw_var.get().strip())
        except ValueError:
            messagebox.showerror('Invalid target', 'Enter numeric values for target X, Y, and yaw.')
            return

        self.config_manager.write_target_mm_deg(x_mm, y_mm, yaw_deg)
        self._log(f'Target updated: X={x_mm:.1f} mm, Y={y_mm:.1f} mm, yaw={yaw_deg:.1f} deg')
        messagebox.showinfo(
            'Target saved',
            'Target position saved.\n\n'
            'If pose or verification is already running, stop and start those buttons again '
            'so they pick up the new target.',
        )

    def _set_indicator(self, ok: bool | None, label: str) -> None:
        if ok is True:
            color = '#22c55e'
            outline = '#15803d'
        elif ok is False:
            color = '#ef4444'
            outline = '#b91c1c'
        else:
            color = '#9ca3af'
            outline = '#6b7280'
        self.indicator_canvas.itemconfigure(self._indicator_id, fill=color, outline=outline)
        self.indicator_label.configure(text=label)

    def _on_placement_status(self, status: PlacementStatus) -> None:
        def _update() -> None:
            if not self._placement_active:
                return
            if status.ok is True:
                self._set_indicator(True, 'CORRECT POSITION')
                self.dx_var.set('—')
                self.dy_var.set('—')
                self.dyaw_var.set('—')
            elif status.ok is False:
                self._set_indicator(False, 'NOT CORRECT POSITION')
                self.dx_var.set(f'{status.dx_mm:+.2f} mm' if status.dx_mm is not None else '—')
                self.dy_var.set(f'{status.dy_mm:+.2f} mm' if status.dy_mm is not None else '—')
                self.dyaw_var.set(f'{status.dyaw_deg:+.2f} deg' if status.dyaw_deg is not None else '—')
            else:
                self._set_indicator(None, 'Listening...')

        self.after(0, _update)

    def _on_close(self) -> None:
        self.placement_listener.stop()
        for process in self.processes.values():
            process.stop()
        self.destroy()


def main() -> None:
    app = CncOperatorGui()
    app.mainloop()


if __name__ == '__main__':
    main()
