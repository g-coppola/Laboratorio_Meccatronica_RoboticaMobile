import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    # Sostituisci 'drone_slam' con il nome effettivo del tuo pacchetto se diverso
    pkg_name = 'drone_slam'
    
    # Percorso assoluto al file YAML
    config_file = os.path.join(
        get_package_share_directory(pkg_name),
        'config',
        'rtabmap_params.yaml'
    )

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
        parameters=[config_file],  # Passa direttamente il file YAML qui
        remappings=remap
    )

    # =========================
    # VIZ
    # =========================
    viz_node = Node(
        package='rtabmap_viz',
        executable='rtabmap_viz',
        output='screen',
        parameters=[config_file],  # Passa direttamente il file YAML qui
        remappings=remap
    )

    return LaunchDescription([
        rtabmap_node,
        viz_node
    ])