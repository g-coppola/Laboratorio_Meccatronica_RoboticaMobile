from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # Parametri di configurazione avanzata per SLAM e OctoMap 3Dc
    parameters = {
        'use_sim_time': True,           
        'frame_id': 'base_link',        
        'subscribe_depth': True,        
        'subscribe_rgb': True,          
        'subscribe_odom_info': False,   
        'approx_sync': True,            
        'queue_size': 30,               
        
        # --- CONFIGURAZIONE MAZZATURA VOLUMETRICA 3D (OCTOMAP) ---
        'Grid/Octomap': 'true',
        'Grid/Octomap/Header': 'true',      # pubblica anche il frame
        'Grid/Octomap/Publish': 'true',
        'Grid/Octomap/Color': 'true',
        'Grid/Sensor': '1',             
        'Grid/3D': 'true',              
        'Grid/RayTracing': 'true',      
        'Grid/CellSize': '0.25',        
        'Grid/MaxObstacleHeight': '5.0',
        'Grid/NormalsSegmentation': 'false',   # passthrough invece di segmentazione tramite normali
        'Grid/MaxGroundHeight': '0.2',         # disabilita la ricerca di un piano "terra" - tutto è ostacolo/punto valido
        'Grid/GroundIsObstacle': 'false',       # utile per UAV: considera tutto come ostacolo, niente distinzione ground/obstacle
        
        # ---> AGGIUNGI QUESTI 3 PARAMETRI ANTI-PIGRIZIA <---
        'RGBD/LinearUpdate': '0.0',    # Aggiorna la mappa anche se lo spostamento è di 0 cm
        'RGBD/AngularUpdate': '0.0',   # Aggiorna la mappa anche se la rotazione è di 0 gradi
        'map_always_update': True,     # Forza la pubblicazione sui topic ROS a prescindere dal movimento
    }

    # Mappatura dei canali di comunicazione (Remappings)
    remappings = [
        ('rgb/image', '/camera/image_raw'),
        ('depth/image', '/camera/depth/image_raw'),
        ('rgb/camera_info', '/camera/camera_info'),
        ('odom', '/model/x500_drone/odometry')
    ]

    # Nodo principale: Calcolatore SLAM
    rtabmap_node = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='rtabmap',
        output='screen',
        parameters=[parameters],
        remappings=remappings,
        arguments=['-d']                # Reset automatico del database ad ogni avvio per pulizia test
    )

    # Nodo interfaccia: RTAB-Map Visualizer GUI
    rtabmap_viz_node = Node(
        package='rtabmap_viz',
        executable='rtabmap_viz',
        name='rtabmap_viz',
        output='screen',
        parameters=[parameters],
        remappings=remappings
    )

    return LaunchDescription([
        rtabmap_node,
        rtabmap_viz_node
    ])