from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'drone_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # Launch
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch','*launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='giucopp',
    maintainer_email='g-coppola134@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'inner_loop_contr = drone_nav.inner_loop_contr:main',
            'odom_to_posestamped = drone_nav.nav6d_pose_bridge:main',
            'path_tracker = drone_nav.path_tracker:main',
        ],
    },
)
