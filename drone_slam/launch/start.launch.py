import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Parametri fondamentali per RTAB-Map in modalità RGB-D
    parameters = [{
        'frame_id': 'base_link',
        'use_sim_time': use_sim_time,
        'subscribe_depth': True,
        'subscribe_rgb': True,
        'subscribe_scan': False,
        'approx_sync': True,          # Gazebo potrebbe avere leggeri ritardi tra RGB e Depth
        'Grid/3D': 'true',            # Genera la mappa a Voxel (come OctoMap)
        'Grid/RangeMax': '5.0',       # Raggio massimo della telecamera
        'Reg/Strategy': '0',          # Strategia di registrazione (0=Visual, 1=ICP, 2=Visual+ICP)
    }]

    # Dobbiamo dire a RTAB-Map dove leggere le immagini e l'odometria che il tuo nodo Python usa
    remappings = [
        ('rgb/image', '/camera/image_raw'),
        ('rgb/camera_info', '/camera/camera_info'),
        ('depth/image', '/camera/depth/image_raw'),
        ('odom', '/model/x500_drone/odometry')
    ]

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),

        # Il nodo principale dello SLAM (Cervello spaziale)
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            output='screen',
            parameters=parameters,
            remappings=remappings,
            arguments=['-d'] # Il flag '-d' cancella il database precedente ad ogni avvio (Clean start)
        ),

        # Il visualizzatore nativo di RTAB-Map (Molto utile per vedere i Loop Closure)
        Node(
            package='rtabmap_viz',
            executable='rtabmap_viz',
            name='rtabmap_viz',
            output='screen',
            parameters=parameters,
            remappings=remappings
        )
    ])