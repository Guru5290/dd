# Copyright 2026 CNC Perception Team
#
# Licensed under the Apache License, Version 2.0

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('cnc_perception')
    workpiece_model = os.path.join(pkg_share, 'config', 'workpiece_model.yaml')

    return LaunchDescription([
        Node(
            package='cnc_perception',
            executable='pose_estimator_node',
            name='pose_estimator_node',
            output='screen',
            parameters=[{'workpiece_config_path': workpiece_model}],
        ),
    ])
