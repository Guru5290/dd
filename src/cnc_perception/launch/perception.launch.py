from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('cnc_perception')
    camera_params = os.path.join(pkg_share, 'config', 'camera_params.yaml')
    workpiece_model = os.path.join(pkg_share, 'config', 'workpiece_model.yaml')
    rviz_config = os.path.join(pkg_share, 'config', 'perception.rviz')

    video_device_arg = DeclareLaunchArgument(
        'video_device',
        default_value='/dev/video0',
        description='V4L2 video device path for the ELP global shutter camera',
    )
    workpiece_config_arg = DeclareLaunchArgument(
        'workpiece_config_path',
        default_value=workpiece_model,
        description='Path to workpiece_model.yaml',
    )
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz with the perception visualization layout',
    )
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=rviz_config,
        description='RViz configuration file',
    )

    usb_cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam_node',
        output='screen',
        parameters=[
            camera_params,
            {'video_device': LaunchConfiguration('video_device')},
        ],
        remappings=[
            ('image_rect_color', '/image_rect_color'),
            ('camera_info', '/camera_info'),
        ],
    )

    pose_estimator_node = Node(
        package='cnc_perception',
        executable='pose_estimator_node',
        name='pose_estimator_node',
        output='screen',
        parameters=[
            {
                'workpiece_config_path': LaunchConfiguration('workpiece_config_path'),
                'image_topic': '/image_rect_color',
                'camera_info_topic': '/camera_info',
                'camera_frame': 'camera_link',
                'workpiece_frame': 'workpiece_frame',
                'max_reprojection_error_px': 8.0,
            }
        ],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    return LaunchDescription([
        video_device_arg,
        workpiece_config_arg,
        use_rviz_arg,
        rviz_config_arg,
        usb_cam_node,
        pose_estimator_node,
        rviz_node,
    ])
