from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    params = {

        'use_sim_time': True,

        # frames
        'frame_id': 'base_link',
        'odom_frame_id': 'odom',
        'map_frame_id': 'map',

        # SENSORI
        'subscribe_scan_cloud': True,
        'subscribe_rgb': False,
        'subscribe_depth': False,

        # ODOM
        'odom_sensor_sync': True,

        # SLAM CORE
        'Reg/Strategy': '0',
        'Rtabmap/DetectionRate': '1',

        # KEYFRAMES (CRUCIALE)
        'RGBD/LinearUpdate': '0.05',
        'RGBD/AngularUpdate': '0.01',

        'Mem/IncrementalMemory': 'true',

        # CLOUD
        'Cloud/FilterNaN': 'true',
        'Cloud/Decimation': '2'
    }

    remap = [
        ('scan_cloud', '/scan_cloud'),
        ('odom', '/odom'),
        ('octomap_full', '/rtabmap/octomap_full'),
        ('octomap_binary', '/rtabmap/octomap_binary'),
        ('octomap_grid', '/rtabmap/octomap_grid'),
        ('octomap_occupancy_grid', '/rtabmap/octomap_occupancy_grid'),
        ('grid_map', '/rtabmap/grid_map'),
    ]

    # =========================
    # RTABMAP SLAM NODE 
    # =========================
    rtabmap_node = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        output='screen',
        parameters=[params],
        remappings=remap
    )

    # =========================
    # VIZ
    # =========================
    viz_node = Node(
        package='rtabmap_viz',
        executable='rtabmap_viz',
        output='screen',
        parameters=[params],
        remappings=remap
    )

    # =========================
    # OCTOMAP
    # =========================
    octomap_node = Node(
        package='octomap_server',
        executable='octomap_server_node',
        name='octomap_server',
        output='screen',
        parameters=[{
            'frame_id': 'map',
            'base_frame_id': 'base_link',

            'resolution': 0.10,

            'sensor_model/max_range': 15.0,
            'sensor_model/min_range': 0.4,

            'filter_ground': False,

            'latch': True,

            'point_cloud_min_z': -2.0,
            'point_cloud_max_z': 2.0,
        }],
        remappings=[
            ('cloud_in', '/scan_cloud')
        ]
    )

    return LaunchDescription([
        rtabmap_node,
        viz_node,
        octomap_node
    ])