import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraSubscriberNode(Node):

    def __init__(self):
        super().__init__('camera_subscriber')
        self.cv_bridge = CvBridge()
        # Inizializza il subscriber al topic della camera
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.camera_callback,
            10)

    def camera_callback(self, msg):
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        cv2.imshow('Camera', cv_image)
        cv2.waitKey(1) 

def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriberNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()