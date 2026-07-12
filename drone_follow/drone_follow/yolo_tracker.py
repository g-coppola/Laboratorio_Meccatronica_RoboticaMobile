#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import math
import numpy as np

from ultralytics import YOLO

class YoloTrackerNode(Node):
    def __init__(self):
        super().__init__('yolo_tracker')
        
        self.cv_bridge = CvBridge()
        # Modello YOLO "nano"
        self.model = YOLO('yolov8n.pt') 
        
        # --- Sottoscrizioni e Pubblicazioni ---
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
        self.goal_pub = self.create_publisher(PoseStamped, '/planner_goal', 10)
        
        # --- Variabili di Stato (Odometria) ---
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_z = 0.0
        self.curr_yaw = 0.0
        self.have_odom = False
        
        # --- Macchina a Stati per l'Inseguimento ---
        self.state = "SEARCHING"
        self.active_goal = None
        self.goal_reach_threshold = 0.8  
        self.min_tracking_distance = 5.0 
        
        # NUOVO: Distanza di sicurezza da mantenere dalla persona (in metri)
        self.standoff_distance = 1.5
        
        # Variabile per memorizzare l'ultima misurazione fatta
        self.last_detection = None
        
        # --- Parametri Telecamera ---
        self.image_w = 640
        self.image_h = 480
        hfov = 1.50098
        self.fx = (self.image_w / 2.0) / math.tan(hfov / 2.0)

        self.get_logger().info('YOLO Tracker Avviato. In attesa della persona...')

    def odom_callback(self, msg: Odometry):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        self.curr_z = msg.pose.pose.position.z
        
        # Estrazione Yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.curr_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        self.have_odom = True

        if self.state == "NAVIGATING" and self.active_goal is not None:
            dist_to_goal = math.hypot(self.curr_x - self.active_goal[0], self.curr_y - self.active_goal[1])
            
            if dist_to_goal < self.goal_reach_threshold:
                self.get_logger().info('Goal raggiunto! Riprendo la scansione visiva.')
                self.state = "SEARCHING"
                self.active_goal = None
                
                # Se passiamo in SEARCHING, controlliamo se c'è un'ultima misurazione nota
                if self.last_detection is not None:
                    # Calcoliamo la distanza tra noi e l'ultima misurazione calcolata (che già include lo standoff)
                    dist_to_last = math.hypot(self.curr_x - self.last_detection[0], self.curr_y - self.last_detection[1])
                    
                    # Se l'ultima misurazione è oltre la soglia di arrivo, la persona si è mossa
                    if dist_to_last > self.goal_reach_threshold:
                        self.get_logger().info('Inoltro l\'ultima posizione nota (con offset) della persona al planner.')
                        self.active_goal = (self.last_detection[0], self.last_detection[1])
                        self.send_goal(*self.last_detection)
                        self.state = "NAVIGATING"
                    else:
                        # Se siamo già vicini all'ultima misurazione (persona ferma), puliamo la variabile
                        self.last_detection = None

    def image_callback(self, msg: Image):
        if not self.have_odom:
            return

        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(cv_image, classes=[0], conf=0.5, verbose=False)
        
        annotated_frame = results[0].plot()
        cv2.imshow('YOLO Tracker', annotated_frame)
        cv2.waitKey(1)

        # Estrazione diretta: essendoci una sola persona, prendiamo la prima rilevata
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            x_c, y_c, w, h = box.xywh[0].tolist()
            
            # Stima della distanza
            real_h = 1.7 
            dist = (real_h * self.fx) / h

            if dist > self.min_tracking_distance:
                # Calcolo della direzione
                angle_offset = -math.atan2(x_c - (self.image_w / 2.0), self.fx)
                target_yaw = self.curr_yaw + angle_offset
                
                # NUOVO: Sottraiamo la distanza di sicurezza per non finire addosso alla persona.
                # Se la persona per caso ci viene incontro a meno della standoff_distance, 
                # target_dist diventa negativo e il drone arretrerà naturalmente mantenendo lo yaw!
                target_dist = dist - self.standoff_distance
                
                # Proiezione nella mappa fermandosi 'standoff_distance' prima del target
                target_x = self.curr_x + target_dist * math.cos(target_yaw)
                target_y = self.curr_y + target_dist * math.sin(target_yaw)
                
                # Quota fissa di sicurezza a 3.2 metri
                target_z = 3.2 

                # Aggiorniamo SEMPRE l'ultima misurazione rilevata, a prescindere dallo stato
                self.last_detection = (target_x, target_y, target_z, target_yaw)

                # Se il drone è fermo e sta cercando, avvia la navigazione immediatamente
                if self.state == "SEARCHING":
                    self.get_logger().info(
                        f'Persona rilevata a {dist:.2f}m. '
                        f'Invio goal a X:{target_x:.2f}, Y:{target_y:.2f}, Z:{target_z:.2f} '
                        f'(Distanza mantenuta: {self.standoff_distance}m) con Yaw:{target_yaw:.2f} rad'
                    )
                    
                    self.active_goal = (target_x, target_y)
                    self.state = "NAVIGATING"
                    self.send_goal(target_x, target_y, target_z, target_yaw)

    def send_goal(self, x, y, z, target_yaw):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        
        # Conversione dell'angolo target_yaw in quaternione 
        # Questo forza il drone a guardare verso la persona quando arriva
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(target_yaw / 2.0)
        msg.pose.orientation.w = math.cos(target_yaw / 2.0)
        
        self.goal_pub.publish(msg)

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
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()