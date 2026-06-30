import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2

from ultralytics import YOLO

class DebugTrackerNode(Node):
    def __init__(self):
        super().__init__('debug_tracker_node')
        
        self.cv_bridge = CvBridge()
        self.model = YOLO('yolov8n.pt') 
        
        # Sottoscrizione alla telecamera e all'odometria del drone
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Stato del drone
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_yaw = 0.0
        
        # Parametri ottici della telecamera Gazebo
        self.image_width = 640.0
        self.image_height = 480.0
        self.fov_h = 1.50098  # FOV orizzontale letto da URDF
        self.fov_v = 1.12     # FOV verticale stimato (rapporto 4:3)
        
        self.real_person_height = 1.7  # Altezza media stimata della persona in metri
        self.follow_distance = 2.0     # Distanza di sicurezza
        
        self.get_logger().info('Nodo di DEBUG Tracker (RGB Camera) avviato!')

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
        
        # Esegui inferenza con YOLO
        results = self.model(cv_image, conf=0.6, verbose=False)
        
        person_detected = False
        largest_bbox_area = 0
        best_bbox = None

        # Cerca la classe 0 (Persona)
        for box in results[0].boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > largest_bbox_area:
                    largest_bbox_area = area
                    best_bbox = (x1, y1, x2, y2)
                    person_detected = True

        annotated_frame = results[0].plot()

        if person_detected:
            self.calculate_coordinates(best_bbox)
            cv2.rectangle(annotated_frame, (best_bbox[0], best_bbox[1]), (best_bbox[2], best_bbox[3]), (0, 255, 0), 3)

        cv2.imshow('YOLO Debug Tracker', annotated_frame)
        cv2.waitKey(1)

    def calculate_coordinates(self, bbox):
        x1, y1, x2, y2 = bbox
        bbox_center_x = (x1 + x2) / 2.0
        bbox_height = y2 - y1
        
        if bbox_height <= 0: 
            return

        # 1. Calcola l'angolo (Yaw) relativo rispetto al centro
        offset_x = bbox_center_x - (self.image_width / 2.0)
        yaw_error = - (offset_x / self.image_width) * self.fov_h
        
        # 2. Calcola la distanza stimata (Modello Pinhole)
        focal_length_y = (self.image_height / 2.0) / math.tan(self.fov_v / 2.0)
        distance = (self.real_person_height * focal_length_y) / bbox_height
        distance = max(1.0, min(15.0, distance))
        
        target_distance = max(0.0, distance - self.follow_distance)
        
        # 3. Proietta nelle coordinate globali
        global_target_yaw = self.curr_yaw + yaw_error
        
        target_x = self.curr_x + target_distance * math.cos(global_target_yaw)
        target_y = self.curr_y + target_distance * math.sin(global_target_yaw)
        
        # Stampa i dati di debug nel terminale
        self.get_logger().info(
            f"[DEBUG] Distanza stimata: {distance:.2f}m | Angolo relativo: {math.degrees(yaw_error):.1f}° | "
            f"Drone Pos (X:{self.curr_x:.1f}, Y:{self.curr_y:.1f}) -> Target Globale (X:{target_x:.1f}, Y:{target_y:.1f}, Z:2.5)"
        )

def main(args=None):
    rclpy.init(args=args)
    node = DebugTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()