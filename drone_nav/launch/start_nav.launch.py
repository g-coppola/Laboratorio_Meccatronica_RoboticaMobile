from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():

    # --- Inner loop / mixer: stabilizza assetto e traduce in velocità motori ---
    inner_loop_node = Node(
        package='drone_nav',
        executable='inner_loop_contr',
        name='inner_loop_controller',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    odom_adapter_node = Node(
        package='drone_nav',
        executable='odom_to_posestamped',
        name='odom_to_posestamped',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'odom_topic': '/odom',
            'pose_topic': '/space_cobot/pose',
            'frame_id': 'map',
        }]
    )

    # --- nav6d: planner + controller PD a 6DoF ---
    nav6d_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('nav6d'),
                'launch',
                'n6d.launch.py'
            ])
        ),
        launch_arguments={
            'controller_type': 'velocity',
        }.items()
    )

    return LaunchDescription([
        inner_loop_node,
        odom_adapter_node,
        nav6d_launch,
    ])