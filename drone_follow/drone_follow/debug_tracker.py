#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import math

import matplotlib.pyplot as plt
from ultralytics import YOLO

class StandaloneDebugTracker(Node):
    def __init__(self):
        super().__init__('debug_tracker')
        
        # --- Vision Setup ---
        self.cv_bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')
        
        # --- Parametri Telecamera ---
        self.image_w = 640
        self.image_h = 480
        hfov = 1.50098
        self.fx = (self.image_w / 2.0) / math.tan(hfov / 2.0)
        
        # --- Variabili di Stato (Odometria) ---
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_yaw = 0.0
        self.have_odom = False
        
        # --- Dati per il Plot ---
        self.person_x = []
        self.person_y = []
        
        # --- Sottoscrizioni ROS ---
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
        # --- Setup Matplotlib Interattivo con Assi Fissi ---
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 10)) # Proporzioni verticali ottimizzate per i tuoi limiti
        self.ax.set_title("Stima Posizione Persona (Standalone)")
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.grid(True)
        
        # Configurazione limiti fissi richiesti
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-40, 40)
        self.ax.set_aspect('equal') # Mantiene la proporzione 1:1 reale dei metri a schermo
        
        # Disegna una linea rossa con i marker
        self.person_plot, = self.ax.plot([], [], 'ro-', markersize=5, alpha=0.7, label='Persona')
        self.ax.legend()
        
        self.create_timer(0.1, self.update_plot)
        
        self.get_logger().info("Standalone Debug Tracker avviato. Plot 2D fisso (X: -20 a 20, Y: -40 a 40) pronto.")

    def odom_callback(self, msg: Odometry):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.curr_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        self.have_odom = True

    def image_callback(self, msg: Image):
        if not self.have_odom:
            return

        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(cv_image, classes=[0], conf=0.5, verbose=False)

        # Se rileva una persona, esegue la matematica
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            x_c, y_c, w, h = box.xywh[0].tolist()
            
            # Stima della distanza
            real_h = 1.7 
            dist = (real_h * self.fx) / h

            # Calcolo della direzione
            angle_offset = -math.atan2(x_c - (self.image_w / 2.0), self.fx)
            target_yaw = self.curr_yaw + angle_offset
            
            # Proiezione nella mappa
            target_x = self.curr_x + dist * math.cos(target_yaw)
            target_y = self.curr_y + dist * math.sin(target_yaw)
            
            # Salva i dati stimati nelle liste del plot
            self.person_x.append(target_x)
            self.person_y.append(target_y)
            
            # Mantiene una scia di 100 punti
            if len(self.person_x) > 100:
                self.person_x.pop(0)
                self.person_y.pop(0)

    def update_plot(self):
        if not self.person_x:
            return
            
        # Aggiorna semplicemente i dati sul grafico senza toccare i limiti degli assi
        self.person_plot.set_data(self.person_x, self.person_y)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def main(args=None):
    rclpy.init(args=args)
    node = StandaloneDebugTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close('all')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()