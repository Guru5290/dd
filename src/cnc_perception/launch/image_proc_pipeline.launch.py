from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('cnc_perception')
    camera_params = os.path.join(pkg_share, 'config', 'camera_params.yaml')

    video_device_arg = DeclareLaunchArgument(
        'video_device',
        default_value='/dev/video0',
        description='V4L2 device path (try v4l2-ctl --list-devices if /dev/video0 fails)',
    )

    usb_cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='camera_driver',
        output='screen',
        parameters=[
            camera_params,
            {'video_device': LaunchConfiguration('video_device')},
        ],
        remappings=[
            ('image_raw', '/image_raw'),
            ('camera_info', '/camera_info_raw'),
        ],
    )

    # Built-in rectifier (OpenCV) — avoids image_proc QoS / composable load issues on Jazzy.
    rectifier_node = Node(
        package='cnc_perception',
        executable='rectify_image_node',
        name='image_rectifier',
        output='screen',
        parameters=[{
            'input_topic': '/image_raw',
            'output_topic': '/image_rect_color',
            'raw_camera_info_topic': '/camera_info_raw',
            'camera_info_topic': '/camera_info',
        }],
    )

    return LaunchDescription([
        video_device_arg,
        usb_cam_node,
        rectifier_node,
    ])
