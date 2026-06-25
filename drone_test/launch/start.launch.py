import os
import xacro
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_name = 'drone_test'
    use_sim_time = LaunchConfiguration('use_sim_time')
    pkg_share = FindPackageShare(pkg_name)
    ros_gz_sim_pkg = FindPackageShare('ros_gz_sim')

    # Necessario per trovare mesh e risorse
    workspace_share_dir = PathJoinSubstitution([
        pkg_share,
        '..'
    ])

    world_file = PathJoinSubstitution([
        pkg_share,
        'worlds',
        'warehouse.sdf'
    ])

    gz_launch_file = PathJoinSubstitution([
        ros_gz_sim_pkg,
        'launch',
        'gz_sim.launch.py'
    ])

    # URDF/Xacro
    xacro_file = os.path.join(
        get_package_share_directory(pkg_name),
        'urdf',
        'x500.urdf'
    )
    robot_description = xacro.process_file(
        xacro_file
    ).toxml()

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock'
        ),

        SetEnvironmentVariable(
            'IGN_GAZEBO_RESOURCE_PATH',
            workspace_share_dir
        ),
        SetEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            workspace_share_dir
        ),

        # Gazebo Fortress
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                gz_launch_file
            ),
            launch_arguments={
                'gz_args': ['-r -v 4 ', world_file],
                'on_exit_shutdown': 'True'
            }.items(),
        ),

        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_description': robot_description
            }]
        ),

        # Spawn robot
        Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-topic', 'robot_description',
                '-name', 'x500_drone',
                '-x', '0.0',
                '-y', '0.0',
                '-z', '0.5'
            ],
            output='screen'
        ),

        # ROS <-> Gazebo Fortress Bridge
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=[
                # Clock
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                
                # RGB Camera
                '/camera@sensor_msgs/msg/Image[gz.msgs.Image',
                '/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
                
                # LiDAR Scan
                '/scan_cloud/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
                
                # Motor commands
                '/x500_drone/command/motor_speed@actuator_msgs/msg/Actuators]gz.msgs.Actuators',
                
                # Ground truth odometry (mantenuta per debug, ma non usata dai controllori)
                '/gazebo/ground_truth/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',
                
                # TF e Joints
                '/model/x500_drone/pose@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
                '/world/warehouse/model/x500_drone/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model',
            ],
            remappings=[
                ('/camera', '/camera/image_raw'),
                ('/camera_info', '/camera/camera_info'),
                ('/model/x500_drone/pose', '/tf'),
                ('/world/warehouse/model/x500_drone/joint_state', '/joint_states'),
                ('/scan_cloud/points', '/scan_cloud'),
                ('/gazebo/ground_truth/odometry', '/odom'),
            ]
        ),
    ])