from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('cnc_perception')
    camera_info_url = 'file://' + os.path.join(pkg_share, 'config', 'camera_info.yaml')

    image_proc_share = get_package_share_directory('image_proc')
    image_proc_launch = os.path.join(image_proc_share, 'launch', 'image_proc.launch.py')

    return LaunchDescription([
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='camera_driver',
            parameters=[{
                'camera_info_url': camera_info_url,
                'image_width': 640,
                'image_height': 480,
                'framerate': 30.0,
                'pixel_format': 'yuyv',
                'video_device': '/dev/video0',
                'frame_id': 'camera_link',
                'camera_name': 'narrow_stereo',
            }],
            remappings=[
                ('image_raw', '/image_raw'),
                ('camera_info', '/camera_info'),
            ],
            output='screen',
        ),
        ComposableNodeContainer(
            name='image_proc_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='image_proc',
                    plugin='image_proc::ImageProcNode',
                    name='image_proc',
                    remappings=[
                        ('image', '/image_raw'),
                        ('camera_info', '/camera_info'),
                        ('image_rect', '/image_rect'),
                        ('image_rect_color', '/image_rect_color'),
                    ],
                ),
            ],
            output='screen',
        ),
    ])
