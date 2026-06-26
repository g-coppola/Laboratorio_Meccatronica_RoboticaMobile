#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


class OdomToPoseStamped(Node):
    def __init__(self):
        super().__init__('odom_to_posestamped')

        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('pose_topic', '/nav6d/pose')
        self.declare_parameter('frame_id', 'map')

        odom_topic = self.get_parameter('odom_topic').value
        pose_topic = self.get_parameter('pose_topic').value
        self.frame_id = self.get_parameter('frame_id').value

        self.sub = self.create_subscription(Odometry, odom_topic, self.callback, 10)
        self.pub = self.create_publisher(PoseStamped, pose_topic, 10)

    def callback(self, msg: Odometry):
        out = PoseStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id
        out.pose = msg.pose.pose
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = OdomToPoseStamped()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()