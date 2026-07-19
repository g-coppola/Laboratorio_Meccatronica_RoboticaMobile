import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from actuator_msgs.msg import Actuators
import math
import matplotlib.pyplot as plt

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
        
        # --- Subscriptions and Publications ---
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        
        # UPDATE: Receiving Odometry (Pose + Twist) instead of PoseStamped
        self.goal_sub = self.create_subscription(Odometry, '/goal_pose', self.goal_callback, 10)
        
        self.motor_pub = self.create_publisher(Actuators, '/x500_drone/command/motor_speed', 10)
        
        # --- Current Position ---
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_z = 0.0
        
        # --- Setpoint (Target Coordinates and Velocities) ---
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 3.0      
        self.target_yaw = 1.5708  # Approximately 90 degrees
        
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_vz = 0.0
        
        self.target_roll = 0.0
        self.target_pitch = 0.0

        # --- PHYSICAL PARAMETERS & BASE THRUST ---
        m_base = 2.0
        m_rotor = 0.01607
        mass = m_base + (4 * m_rotor)
        gravity = 9.81
        motor_constant = 8.54858e-06
        
        # Hovering speed calculation: w = sqrt(m*g / (4*C_T))
        self.base_thrust = math.sqrt((mass * gravity) / (4 * motor_constant))

        
        # Feedforward Gain
        self.k_ff_xy = 0.1
        self.k_ff_z = 0.5
        
        # --- OUTER PID LOOP (X, Y Position) ---
        self.pid_x = PIDController(kp=0.15, ki=0.0, kd=0.1, min_out=-0.25, max_out=0.25)
        self.pid_y = PIDController(kp=0.15, ki=0.0, kd=0.1, min_out=-0.25, max_out=0.25)

        # --- INNER PID LOOP (Altitude and Attitude) ---
        self.pid_alt   = PIDController(kp=15.0, ki=0.05, kd=45.0, min_out=-150.0, max_out=150.0)
        self.pid_roll  = PIDController(kp=35.0, ki=0.01, kd=8.0,  min_out=-80.0,  max_out=80.0)
        self.pid_pitch = PIDController(kp=35.0, ki=0.01, kd=8.0,  min_out=-80.0,  max_out=80.0)
        self.pid_yaw   = PIDController(kp=8.0, ki=0.01, kd=13.0,  min_out=-50.0,  max_out=50.0)
        
        self.last_time = 0.0
        self.start_time = None
        self.debug_counter = 0
        
        # --- Plotting Variables ---
        self.log_t = []
        self.log_x = []
        self.log_y = []
        self.log_z = []
        self.log_roll = []
        self.log_pitch = []
        self.log_yaw = []
        
        # Arrays to log motor inputs
        self.log_w0 = []
        self.log_w1 = []
        self.log_w2 = []
        self.log_w3 = []
        
        self.get_logger().info("Complete Autonomous Flight Controller Started! (Press Ctrl+C for plots)")

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
        self.target_x = msg.pose.pose.position.x
        self.target_y = msg.pose.pose.position.y
        self.target_z = msg.pose.pose.position.z
        
        self.target_vx = msg.twist.twist.linear.x
        self.target_vy = msg.twist.twist.linear.y
        self.target_vz = msg.twist.twist.linear.z
        
        q = msg.pose.pose.orientation
        _, _, rviz_yaw = self.quaternion_to_euler(q.w, q.x, q.y, q.z)
        self.target_yaw = rviz_yaw

    def odom_callback(self, msg):
        current_time_raw = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
        
        if self.start_time is None:
            self.start_time = current_time_raw
            self.last_time = current_time_raw
            return
            
        current_time = current_time_raw - self.start_time
        dt = current_time_raw - self.last_time
        if dt <= 0.0: return
        self.last_time = current_time_raw

        # --- SENSOR READINGS ---
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        self.curr_z = msg.pose.pose.position.z
        
        q = msg.pose.pose.orientation
        curr_roll, curr_pitch, curr_yaw = self.quaternion_to_euler(q.w, q.x, q.y, q.z)

        # --- DATA LOGGING FOR KINEMATICS ---
        self.log_t.append(current_time)
        self.log_x.append(self.curr_x)
        self.log_y.append(self.curr_y)
        self.log_z.append(self.curr_z)
        self.log_roll.append(math.degrees(curr_roll))
        self.log_pitch.append(math.degrees(curr_pitch))
        self.log_yaw.append(math.degrees(curr_yaw))

        # --- OUTER LOOP: X, Y POSITION CONTROL WITH FEEDFORWARD ---
        err_x = self.target_x - self.curr_x
        err_y = self.target_y - self.curr_y
        
        cos_y = math.cos(curr_yaw)
        sin_y = math.sin(curr_yaw)
        
        # 1. Rotation of global error into Body Frame
        err_x_body = err_x * cos_y + err_y * sin_y
        err_y_body = -err_x * sin_y + err_y * cos_y
        
        # 2. Rotation of target velocity into Body Frame (Anticipatory Action)
        target_vx_body = self.target_vx * cos_y + self.target_vy * sin_y
        target_vy_body = -self.target_vx * sin_y + self.target_vy * cos_y
        
        # 3. Combined Control Action: Reactive PID + Predictive Feedforward
        u_x = self.pid_x.compute(err_x_body, dt) + (self.k_ff_xy * target_vx_body)
        u_y = self.pid_y.compute(err_y_body, dt) + (self.k_ff_xy * target_vy_body)
        
        self.target_pitch = u_x
        self.target_roll = -u_y

        # --- INNER LOOP: ATTITUDE AND ALTITUDE CONTROL ---
        err_alt   = self.target_z - self.curr_z
        err_roll  = self.target_roll - curr_roll
        err_pitch = self.target_pitch - curr_pitch
        
        err_yaw = self.target_yaw - curr_yaw
        err_yaw = math.atan2(math.sin(err_yaw), math.cos(err_yaw))

        
        
        # NEW: + altitude feedforward
        u_alt   = self.pid_alt.compute(err_alt, dt) + (self.k_ff_z * self.target_vz)
        u_roll  = self.pid_roll.compute(err_roll, dt)
        u_pitch = self.pid_pitch.compute(err_pitch, dt)
        u_yaw   = self.pid_yaw.compute(err_yaw, dt)
        
        thrust = self.base_thrust + u_alt

        # --- MIXING MATRIX ---
        w0 = thrust - u_roll - u_pitch - u_yaw   
        w1 = thrust + u_roll + u_pitch - u_yaw   
        w2 = thrust + u_roll - u_pitch + u_yaw   
        w3 = thrust - u_roll + u_pitch + u_yaw   

        # --- ACTUATION CLAMPING AND LOGGING ---
        cmd_w0 = max(0.0, min(1000.0, w0))
        cmd_w1 = max(0.0, min(1000.0, w1))
        cmd_w2 = max(0.0, min(1000.0, w2))
        cmd_w3 = max(0.0, min(1000.0, w3))
        
        self.log_w0.append(cmd_w0)
        self.log_w1.append(cmd_w1)
        self.log_w2.append(cmd_w2)
        self.log_w3.append(cmd_w3)

        # --- PUBLISH MOTOR COMMANDS ---
        act_msg = Actuators()
        act_msg.velocity = [cmd_w0, cmd_w1, cmd_w2, cmd_w3]
        self.motor_pub.publish(act_msg)
        
        # --- DEBUG ---
        self.debug_counter += 1
        if self.debug_counter >= 50:
            distance = math.sqrt(err_x**2 + err_y**2)
            self.debug_counter = 0

    def generate_report_plots(self):
        if not self.log_t:
            print("\nNo recorded data to plot.")
            return

        print("\nGenerating plots...")
        plt.style.use('default')
        
        # ==========================================
        # FIGURE 1: DRONE KINEMATIC RESPONSE
        # ==========================================
        fig, axs = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('Drone Controller Kinematic Response', fontsize=16, fontweight='bold')

        t = self.log_t

        # 1. X Position
        axs[0, 0].plot(t, self.log_x, 'b-', label='X Position')
        axs[0, 0].axhline(y=0.0, color='r', linestyle='--', linewidth=2, label='Ref X=0')
        axs[0, 0].set_title('X Position')
        axs[0, 0].set_ylabel('[m]')
        axs[0, 0].grid(True)
        axs[0, 0].legend()

        # 2. Roll
        axs[0, 1].plot(t, self.log_roll, 'm-', label='Roll')
        axs[0, 1].axhline(y=0.0, color='r', linestyle='--', linewidth=2, label='Ref Roll=0°')
        axs[0, 1].set_title('Roll Angle')
        axs[0, 1].set_ylabel('[deg]')
        axs[0, 1].grid(True)
        axs[0, 1].legend()

        # 3. Y Position
        axs[1, 0].plot(t, self.log_y, 'b-', label='Y Position')
        axs[1, 0].axhline(y=0.0, color='r', linestyle='--', linewidth=2, label='Ref Y=0')
        axs[1, 0].set_title('Y Position')
        axs[1, 0].set_ylabel('[m]')
        axs[1, 0].grid(True)
        axs[1, 0].legend()

        # 4. Pitch
        axs[1, 1].plot(t, self.log_pitch, 'm-', label='Pitch')
        axs[1, 1].axhline(y=0.0, color='r', linestyle='--', linewidth=2, label='Ref Pitch=0°')
        axs[1, 1].set_title('Pitch Angle')
        axs[1, 1].set_ylabel('[deg]')
        axs[1, 1].grid(True)
        axs[1, 1].legend()

        # 5. Z Altitude
        axs[2, 0].plot(t, self.log_z, 'g-', label='Z Altitude')
        axs[2, 0].axhline(y=3.0, color='r', linestyle='--', linewidth=2, label='Ref Z=3')
        axs[2, 0].set_title('Z Altitude')
        axs[2, 0].set_xlabel('Time [s]')
        axs[2, 0].set_ylabel('[m]')
        axs[2, 0].grid(True)
        axs[2, 0].legend()

        # 6. Yaw
        axs[2, 1].plot(t, self.log_yaw, 'c-', label='Yaw')
        axs[2, 1].axhline(y=90.0, color='r', linestyle='--', linewidth=2, label='Ref Yaw=90°')
        axs[2, 1].set_title('Yaw Angle')
        axs[2, 1].set_xlabel('Time [s]')
        axs[2, 1].set_ylabel('[deg]')
        axs[2, 1].grid(True)
        axs[2, 1].legend()

        fig.tight_layout()
        fig.subplots_adjust(top=0.92)
        
        # ==========================================
        # FIGURE 2: CONTROL INPUTS (MOTOR SPEEDS)
        # ==========================================
        fig2, axs2 = plt.subplots(2, 2, figsize=(12, 8))
        fig2.suptitle('Control Inputs - Actuator Commands', fontsize=16, fontweight='bold')

        # Motor 0 (Front Right - CCW)
        axs2[0, 0].plot(t, self.log_w0, 'r-', label='Motor 0 Cmd')
        axs2[0, 0].set_title('Motor 0 (Front Right)')
        axs2[0, 0].set_ylabel('Speed [rad/s]')
        axs2[0, 0].grid(True)
        axs2[0, 0].legend()

        # Motor 1 (Rear Left - CCW)
        axs2[0, 1].plot(t, self.log_w1, 'b-', label='Motor 1 Cmd')
        axs2[0, 1].set_title('Motor 1 (Rear Left)')
        axs2[0, 1].set_ylabel('Speed [rad/s]')
        axs2[0, 1].grid(True)
        axs2[0, 1].legend()

        # Motor 2 (Front Left - CW)
        axs2[1, 0].plot(t, self.log_w2, 'g-', label='Motor 2 Cmd')
        axs2[1, 0].set_title('Motor 2 (Front Left)')
        axs2[1, 0].set_xlabel('Time [s]')
        axs2[1, 0].set_ylabel('Speed [rad/s]')
        axs2[1, 0].grid(True)
        axs2[1, 0].legend()

        # Motor 3 (Rear Right - CW)
        axs2[1, 1].plot(t, self.log_w3, 'm-', label='Motor 3 Cmd')
        axs2[1, 1].set_title('Motor 3 (Rear Right)')
        axs2[1, 1].set_xlabel('Time [s]')
        axs2[1, 1].set_ylabel('Speed [rad/s]')
        axs2[1, 1].grid(True)
        axs2[1, 1].legend()

        fig2.tight_layout()
        fig2.subplots_adjust(top=0.92)

        try:
            # Mostrerà entrambe le finestre contemporaneamente
            plt.show(block=True)
        except Exception as e:
            print(f"Unable to show the plot (missing GUI backend like tkinter?): {e}")


def main(args=None):
    rclpy.init(args=args)
    node = FullDroneController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nInterrupt command received (Ctrl+C).")
    finally:
        # Generate the plot right before shutting down ROS
        if len(node.log_t) > 0:
            node.generate_report_plots()
            
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()