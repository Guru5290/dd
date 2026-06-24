from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'cnc_perception'

setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='CNC Perception Team',
    maintainer_email='user@example.com',
    description='Markerless 6D pose estimation for plain CNC workpieces.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'pose_estimator_node = cnc_perception.pose_estimator_node:main',
            'step01_verify_camera = cnc_perception.step01_verify_camera:main',
            'step02_calibrate_bed_origin = cnc_perception.step02_calibrate_bed_origin:main',
            'step02b_calibrate_bed_corners = cnc_perception.step02b_calibrate_bed_corners:main',
            'step03_publish_bed_tf = cnc_perception.step03_publish_bed_tf:main',
            'step04_debug_workpiece_detection = cnc_perception.step04_debug_workpiece_detection:main',
            'step05_workpiece_pose_bed_frame = cnc_perception.step05_workpiece_pose_bed_frame:main',
            'step06_check_placement = cnc_perception.step06_check_placement:main',
            'step00_generate_aruco_marker = cnc_perception.step00_generate_aruco_marker:main',
            'rectify_image_node = cnc_perception.rectify_image_node:main',
        ],
    },
)
