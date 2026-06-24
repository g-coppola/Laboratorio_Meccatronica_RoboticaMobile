import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from actuator_msgs.msg import Actuators

import numpy as np
import scipy.linalg
import math

class LQRDroneController(Node):
    def __init__(self):
        super().__init__('lqr_drone_controller')
        
        # --- Sottoscrizioni e Pubblicazioni ---
        self.odom_sub = self.create_subscription(Odometry, '/model/x500_drone/odometry', self.odom_callback, 10)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        
        self.motor_pub = self.create_publisher(Actuators, '/x500_drone/command/motor_speed', 10)
        
        # --- Parametri Fisici (estratti dal tuo x500.urdf) ---
        self.m = 2.064        # Massa totale (2.0 base_link + 4 * 0.01607 rotori)
        self.g = 9.81         # Gravità
        self.Ixx = 0.021666
        self.Iyy = 0.021666
        self.Izz = 0.040000
        
        # Parametri Motori
        self.k_f = 8.54858e-06  # Costante di forza
        self.k_m = 0.016        # Costante di momento
        self.l = 0.174 * math.sqrt(2) # Distanza dal centro di massa al rotore
        
        # Spinta di hovering base (da aggiungere all'output dell'LQR sull'asse Z)
        self.hover_thrust = self.m * self.g
        
        # Inizializzazione Vettori di Stato
        self.current_state = np.zeros((12, 1))
        self.target_state = np.zeros((12, 1))
        self.target_state[2, 0] = 2.0  # Altitudine target iniziale (Z)
        
        # Calcolo della matrice LQR K
        self.K = self.compute_lqr_gain()
        
        self.get_logger().info("LQR Flight Controller Avviato con successo!")
        self.debug_counter = 0

    def compute_lqr_gain(self):
        """Costruisce le matrici A e B linearizzate e risolve l'equazione di Riccati"""
        
        # Matrice di stato A (12x12)
        A = np.zeros((12, 12))
        # Derivate della posizione -> velocità (x_dot, y_dot, z_dot)
        A[0, 3] = 1.0; A[1, 4] = 1.0; A[2, 5] = 1.0
        # Derivate degli angoli -> velocità angolari (phi_dot, theta_dot, psi_dot)
        A[6, 9] = 1.0; A[7, 10] = 1.0; A[8, 11] = 1.0
        
        # Dinamica linearizzata per hovering (piccoli angoli)
        A[3, 7] = -self.g  # x_ddot = -g * theta
        A[4, 6] = self.g   # y_ddot = g * phi
        
        # Matrice di ingresso B (12x4)
        B = np.zeros((12, 4))
        B[5, 0] = 1.0 / self.m     # Z dipendente dalla spinta totale U1
        B[9, 1] = 1.0 / self.Ixx   # Rollio dipendente dalla coppia U2
        B[10, 2] = 1.0 / self.Iyy  # Beccheggio dipendente dalla coppia U3
        B[11, 3] = 1.0 / self.Izz  # Imbardata dipendente dalla coppia U4

        # Matrici di Costo Q (12x12) e R (4x4)
        # Il tuning di queste matrici definisce l'aggressività del drone
        Q = np.diag([
            150.0, 150.0, 150.0,  # x, y, z
            1.0, 1.0, 1.0,     # u, v, w (velocità lineari)
            50.0, 50.0, 50.0,     # roll, pitch, yaw
            0.5, 0.5, 0.5      # velocità angolari
        ])
        
        R = np.diag([1.0, 50.0, 50.0, 50.0]) # Penalità sugli input (U1, U2, U3, U4)

        # Risoluzione della Continuous Algebraic Riccati Equation (CARE)
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        
        # Calcolo del guadagno ottimo K = R^-1 B^T P
        K = np.linalg.inv(R).dot(B.T).dot(P)
        return K

    def quaternion_to_euler(self, w, x, y, z):
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def cmd_vel_callback(self, msg):
        self.target_state[0, 0] += msg.linear.x * 0.1  # x
        self.target_state[1, 0] += msg.linear.y * 0.1  # y
        self.target_state[2, 0] += msg.linear.z * 0.1  # z
        self.target_state[8, 0] += msg.angular.z * 0.1 # yaw

    def goal_callback(self, msg):
        # Update Waypoint
        self.target_state[0, 0] = msg.pose.position.x
        self.target_state[1, 0] = msg.pose.position.y
        
        q = msg.pose.orientation
        _, _, rviz_yaw = self.quaternion_to_euler(q.w, q.x, q.y, q.z)
        self.target_state[8, 0] = rviz_yaw
        
        self.get_logger().info(f"Nuovo waypoint LQR -> X:{self.target_state[0,0]:.2f}, Y:{self.target_state[1,0]:.2f}, Z:{self.target_state[2,0]:.2f}")

    def odom_callback(self, msg):
        # 1. ESTREZIONE STATO CORRENTE (12D)
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear
        ang_vel = msg.twist.twist.angular
        q = msg.pose.pose.orientation
        
        roll, pitch, yaw = self.quaternion_to_euler(q.w, q.x, q.y, q.z)

        # Costruiamo il vettore di stato attuale
        self.current_state = np.array([
            [pos.x], [pos.y], [pos.z],          # Posizione globale
            [vel.x], [vel.y], [vel.z],          # Velocità lineari
            [roll], [pitch], [yaw],             # Orientamento
            [ang_vel.x], [ang_vel.y], [ang_vel.z] # Velocità angolari
        ])

        # 2. CALCOLO DELL'ERRORE E AZIONE DI CONTROLLO
        state_error = self.current_state - self.target_state
        
        # Gestione del wrap-around dello yaw (l'errore deve stare tra -pi e pi)
        state_error[8, 0] = math.atan2(math.sin(state_error[8, 0]), math.cos(state_error[8, 0]))

        # Legge di controllo LQR: u = -K * e
        u = -self.K.dot(state_error)
        
        # Estrazione degli input virtuali ottimi calcolati da LQR
        u1 = float(u[0, 0]) # Delta Spinta Z
        u2 = float(u[1, 0]) # Coppia Rollio
        u3 = float(u[2, 0]) # Coppia Beccheggio
        u4 = float(u[3, 0]) # Coppia Imbardata

        # Aggiungiamo la spinta per bilanciare la gravità (Hovering thrust)
        total_thrust = self.hover_thrust + u1

        # 3. MATRICE DI MIXING DEI MOTORI
        # Calcoliamo i comandi per i singoli motori (w0..w3) dai comandi virtuali (U1..U4)
        # Scala empirica per mappare N in radianti/s per il plugin Gazebo
        scale = 100.0 
        
        # Disposizione a X derivata dal tuo URDF x500
        m1 = total_thrust - u2 + u3 - u4  # rot_0 (CCW) Anteriore Destro
        m2 = total_thrust + u2 - u3 - u4  # rot_1 (CCW) Posteriore Sinistro
        m3 = total_thrust - u2 - u3 + u4  # rot_2 (CW) Anteriore Sinistro
        m4 = total_thrust + u2 + u3 + u4  # rot_3 (CW) Posteriore Destro
        
        # 4. PUBBLICAZIONE ATTUAZIONE
        act_msg = Actuators()
        act_msg.velocity = [
            max(0.0, min(1000.0, m1 * scale)),
            max(0.0, min(1000.0, m2 * scale)),
            max(0.0, min(1000.0, m3 * scale)),
            max(0.0, min(1000.0, m4 * scale))
        ]
        self.motor_pub.publish(act_msg)
        
        # Debugging lento (1 volta ogni ~50 callback a 50Hz)
        self.debug_counter += 1
        if self.debug_counter >= 50:
            err_pos = np.linalg.norm(state_error[0:3, 0])
            self.get_logger().info(
                f"LQR Distanza dal Target: {err_pos:.2f}m | Spinta Tot: {total_thrust:.1f}N"
            )
            self.debug_counter = 0

def main(args=None):
    rclpy.init(args=args)
    node = LQRDroneController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()