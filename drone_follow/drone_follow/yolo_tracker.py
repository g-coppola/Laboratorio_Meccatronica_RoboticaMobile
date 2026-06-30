import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped  # Importante per il planner
from cv_bridge import CvBridge

from ultralytics import YOLO

class YoloTrackerNode(Node):
    def __init__(self):
        super().__init__('yolo_tracker_node')
        
        self.cv_bridge = CvBridge()
        self.model = YOLO('yolov8n.pt') 
        
        # Sottoscrizioni
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # PUBLISHER: Invia le coordinate al planner A*
        self.goal_pub = self.create_publisher(PoseStamped, '/planner_goal', 10)
        
        # Stato del drone
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_yaw = 0.0
        
        # Parametri Ottici (da URDF)
        self.image_width = 640.0
        self.image_height = 480.0
        self.fov_h = 1.50098  
        self.fov_v = 1.12     
        
        # Regole di Inseguimento
        self.real_person_height = 1.7  
        self.follow_distance = 2.0     # Si ferma a 2 metri dalla persona
        self.target_altitude = 2.5     # Vola a 2.5 metri di quota fissa
        
        # Rate Limiting
        self.last_goal_time = self.get_clock().now()
        
        self.get_logger().info('🚀 YOLO Tracker (Headless) Avviato! Pronto all\'inseguimento.')

    def quaternion_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        self.curr_yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)

    def camera_callback(self, msg):
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        results = self.model(cv_image, conf=0.6, verbose=False)
        
        person_detected = False
        largest_bbox_area = 0
        best_bbox = None

        for box in results[0].boxes:
            if int(box.cls[0]) == 0:  # Classe 0 = Persona
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > largest_bbox_area:
                    largest_bbox_area = area
                    best_bbox = (x1, y1, x2, y2)
                    person_detected = True

        if person_detected:
            self.track_and_publish(best_bbox)

    def track_and_publish(self, bbox):
        now = self.get_clock().now()
        
        # RATE LIMITING: Calcola e pubblica il target massimo 1 volta al secondo
        if (now - self.last_goal_time).nanoseconds < 1e9:
            return

        x1, y1, x2, y2 = bbox
        bbox_center_x = (x1 + x2) / 2.0
        bbox_height = y2 - y1
        
        if bbox_height <= 0: 
            return

        # 1. Calcolo Angolo (Yaw)
        offset_x = bbox_center_x - (self.image_width / 2.0)
        yaw_error = - (offset_x / self.image_width) * self.fov_h
        
        # 2. Calcolo Distanza
        focal_length_y = (self.image_height / 2.0) / math.tan(self.fov_v / 2.0)
        distance = (self.real_person_height * focal_length_y) / bbox_height
        distance = max(1.0, min(15.0, distance))
        
        target_distance = max(0.0, distance - self.follow_distance)
        
        # 3. Trasformazione Globale
        global_target_yaw = self.curr_yaw + yaw_error
        target_x = self.curr_x + target_distance * math.cos(global_target_yaw)
        target_y = self.curr_y + target_distance * math.sin(global_target_yaw)
        
        # 4. PUBBLICAZIONE AL PLANNER A*
        goal_msg = PoseStamped()
        goal_msg.header.stamp = now.to_msg()
        goal_msg.header.frame_id = 'odom'
        
        goal_msg.pose.position.x = target_x
        goal_msg.pose.position.y = target_y
        goal_msg.pose.position.z = self.target_altitude 
        
        self.goal_pub.publish(goal_msg)
        self.last_goal_time = now
        
        self.get_logger().info(f"INSEGUIMENTO ATTIVO -> Target inviato al Planner A* (X:{target_x:.2f}, Y:{target_y:.2f}, Z:{self.target_altitude})")

def main(args=None):
    rclpy.init(args=args)
    node = YoloTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()