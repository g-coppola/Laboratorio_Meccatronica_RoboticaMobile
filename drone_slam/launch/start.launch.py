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
        'Cloud/Decimation': '2',

        # OCCUPANCY GRID 3D / OCTOMAP (rtabmap genera lui l'octomap, niente octomap_server)
        'Grid/3D': 'true',                   # costruisce griglia 3D, non solo proiezione 2D
        'Grid/FromDepth': 'false',           # la sorgente e' la nuvola lidar, non una depth camera
        'Grid/Sensor': '0',                  # 0 = usa scan_cloud come sorgente della griglia
        'Grid/RangeMax': '15.0',             # deve combaciare col range max del lidar
        'Grid/CellSize': '0.25',             # risoluzione voxel (prima era 'resolution' di octomap_server)
        'Grid/RayTracing': 'true',           # marca anche le celle libere via ray tracing, non solo quelle occupate
        'Grid/NormalsSegmentation': 'false', # niente segmentazione superfici basata su normali (serve per depth camera)
        'RGBD/CreateOccupancyGrid': 'true'   # abilita la creazione/pubblicazione della grid 3D/octomap
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

    return LaunchDescription([
        rtabmap_node,
        viz_node
    ])