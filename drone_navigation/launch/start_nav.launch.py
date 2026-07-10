from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='drone_navigation',
            namespace='',
            executable='planner',
            name='planner',
            arguments=[
                # To synchronize the node's clock with Gazebo. 
                # Without this parameter, RViz drops the trajectory messages due to a timestamp mismatch.
                {'use_sim_time': True}
            ]
        ),
        Node(
            package='drone_navigation',
            namespace='',
            executable='trajectory_generator',
            name='trajectory_generator',
        ),
        Node(
            package='drone_control',
            namespace='',
            executable='full_control',
            name='PID_controller',
        )
    ])