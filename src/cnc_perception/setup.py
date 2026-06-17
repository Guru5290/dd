from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'cnc_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
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
        ],
    },
)
