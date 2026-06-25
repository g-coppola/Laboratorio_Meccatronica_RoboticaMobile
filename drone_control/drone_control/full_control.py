import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from actuator_msgs.msg import Actuators
import math

class PIDController:
    def __init__(self, kp, ki, kd, min_out, max_out):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_out = min_out
        self.max_out = max_out
        
        self.integral = 0.0
        self.last_error = 0.0

    def compute(self, error, dt):
        if dt <= 0.0:
            return 0.0
        self.integral += error * dt
        self.integral = max(-50.0, min(50.0, self.integral))
        
        derivative = (error - self.last_error) / dt
        self.last_error = error
        
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        return max(self.min_out, min(self.max_out, output))


class FullDroneController(Node):
    def __init__(self):
        super().__init__('full_drone_controller')
        
        # --- Sottoscrizioni e Pubblicazioni ---
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        
        # AGGIORNAMENTO: Riceviamo Odometry (Pose + Twist) invece di PoseStamped
        self.goal_sub = self.create_subscription(Odometry, '/goal_pose', self.goal_callback, 10)
        
        self.motor_pub = self.create_publisher(Actuators, '/x500_drone/command/motor_speed', 10)
        
        # --- Posizione Corrente ---
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_z = 0.0
        
        # --- Setpoint (Coordinate e Velocità Bersaglio) ---
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 2.0      
        self.target_yaw = 0.0
        
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_vz = 0.0
        
        self.target_roll = 0.0
        self.target_pitch = 0.0
        
        # Guadagno del Feedforward (Regolabile)
        self.k_ff_xy = 0.25 
        
        # --- PID LOOP ESTERNO (Posizione X, Y) ---
        self.pid_x = PIDController(kp=0.15, ki=0.0, kd=0.1, min_out=-0.45, max_out=0.45)
        self.pid_y = PIDController(kp=0.15, ki=0.0, kd=0.1, min_out=-0.45, max_out=0.45)

        # --- PID LOOP INTERNO (Altitudine e Assetto) ---
        self.pid_alt   = PIDController(kp=12.0, ki=0.0, kd=25.0, min_out=-150.0, max_out=150.0)
        self.pid_roll  = PIDController(kp=35.0, ki=0.0, kd=8.0,  min_out=-80.0,  max_out=80.0)
        self.pid_pitch = PIDController(kp=35.0, ki=0.0, kd=8.0,  min_out=-80.0,  max_out=80.0)
        self.pid_yaw   = PIDController(kp=10.0, ki=0.0, kd=2.0,  min_out=-50.0,  max_out=50.0)
        
        self.last_time = 0.0
        self.debug_counter = 0
        
        self.get_logger().info("Flight Controller Autonomo Completo (Feedforward Enable) Avviato!")

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
        self.target_x += msg.linear.x * 0.1
        self.target_y += msg.linear.y * 0.1
        self.target_z += msg.linear.z * 0.1
        self.target_yaw += msg.angular.z * 0.1

    def goal_callback(self, msg):
        # Estrai Posa
        self.target_x = msg.pose.pose.position.x
        self.target_y = msg.pose.pose.position.y
        self.target_z = msg.pose.pose.position.z
        
        # Estrai Twist (Velocità generate dal Planner/Traiettoria)
        self.target_vx = msg.twist.twist.linear.x
        self.target_vy = msg.twist.twist.linear.y
        self.target_vz = msg.twist.twist.linear.z
        
        q = msg.pose.pose.orientation
        _, _, rviz_yaw = self.quaternion_to_euler(q.w, q.x, q.y, q.z)
        self.target_yaw = rviz_yaw

    def odom_callback(self, msg):
        current_time = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
        if self.last_time == 0.0:
            self.last_time = current_time
            return
            
        dt = current_time - self.last_time
        if dt <= 0.0: return
        self.last_time = current_time

        # --- LETTURA SENSORI ---
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        self.curr_z = msg.pose.pose.position.z
        
        q = msg.pose.pose.orientation
        curr_roll, curr_pitch, curr_yaw = self.quaternion_to_euler(q.w, q.x, q.y, q.z)

        # --- LOOP ESTERNO: CONTROLLO POSIZIONE X, Y CON FEEDFORWARD ---
        err_x = self.target_x - self.curr_x
        err_y = self.target_y - self.curr_y
        
        cos_y = math.cos(curr_yaw)
        sin_y = math.sin(curr_yaw)
        
        # 1. Rotazione dell'errore globale nel Body Frame
        err_x_body = err_x * cos_y + err_y * sin_y
        err_y_body = -err_x * sin_y + err_y * cos_y
        
        # 2. Rotazione della velocità target nel Body Frame (Azione Anticipatoria)
        target_vx_body = self.target_vx * cos_y + self.target_vy * sin_y
        target_vy_body = -self.target_vx * sin_y + self.target_vy * cos_y
        
        # 3. Azione di Controllo Combinata: PID Reattivo + Feedforward Predittivo
        u_x = self.pid_x.compute(err_x_body, dt) + (self.k_ff_xy * target_vx_body)
        u_y = self.pid_y.compute(err_y_body, dt) + (self.k_ff_xy * target_vy_body)
        
        self.target_pitch = u_x
        self.target_roll = -u_y

        # --- LOOP INTERNO: CONTROLLO ASSETTO E ALTITUDINE ---
        err_alt   = self.target_z - self.curr_z
        err_roll  = self.target_roll - curr_roll
        err_pitch = self.target_pitch - curr_pitch
        
        err_yaw = self.target_yaw - curr_yaw
        err_yaw = math.atan2(math.sin(err_yaw), math.cos(err_yaw))

        base_thrust = 770.0 
        
        u_alt   = self.pid_alt.compute(err_alt, dt)
        u_roll  = self.pid_roll.compute(err_roll, dt)
        u_pitch = self.pid_pitch.compute(err_pitch, dt)
        u_yaw   = self.pid_yaw.compute(err_yaw, dt)
        
        thrust = base_thrust + u_alt

        # --- MATRICE DI MIXING ---
        w0 = thrust - u_roll - u_pitch - u_yaw   
        w1 = thrust + u_roll + u_pitch - u_yaw   
        w2 = thrust + u_roll - u_pitch + u_yaw   
        w3 = thrust - u_roll + u_pitch + u_yaw   

        # --- ATTUAZIONE ---
        act_msg = Actuators()
        act_msg.velocity = [
            max(0.0, min(1000.0, w0)),
            max(0.0, min(1000.0, w1)),
            max(0.0, min(1000.0, w2)),
            max(0.0, min(1000.0, w3))
        ]
        self.motor_pub.publish(act_msg)
        
        # --- DEBUG ---
        self.debug_counter += 1
        if self.debug_counter >= 50:
            distanza = math.sqrt(err_x**2 + err_y**2)
            # Rimuovo il log per evitare spam eccessivo, puoi riattivarlo se serve
            # self.get_logger().info(...)
            self.debug_counter = 0

def main(args=None):
    rclpy.init(args=args)
    node = FullDroneController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()