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


class InnerLoopController(Node):
    def __init__(self):
        super().__init__('inner_loop_controller')

        # --- Sottoscrizioni e Pubblicazioni ---
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Comando di velocità body-frame da n6d_velocity_controller
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/space_cobot/cmd_vel', self.cmd_vel_callback, 10
        )

        self.motor_pub = self.create_publisher(Actuators, '/x500_drone/command/motor_speed', 10)

        # --- Posizione e assetto correnti ---
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_z = 0.0

        # --- Setpoint di attitude/quota, aggiornati dal comando di velocità ---
        self.target_z = 2.0
        self.target_yaw = 0.0
        self.target_roll = 0.0
        self.target_pitch = 0.0

        # Guadagno di conversione velocità body -> angolo di attitude target
        # (analogo al k_ff_xy precedente, ma qui è la mappatura primaria,
        # non un feedforward aggiuntivo: tarare in base alla risposta del drone)
        self.k_vel_to_angle = 0.5

        # Tempo dell'ultimo comando di velocità ricevuto, per timeout/hover di sicurezza
        self.last_cmd_vel_time = None
        self.cmd_vel_timeout = 0.5  # s

        self.last_cmd_vx = 0.0
        self.last_cmd_vy = 0.0
        self.last_cmd_vz = 0.0
        self.last_cmd_yaw_rate = 0.0

        # --- PID LOOP INTERNO (Altitudine e Assetto) — invariati dal controller originale ---
        self.pid_alt   = PIDController(kp=12.0, ki=0.0, kd=25.0, min_out=-150.0, max_out=150.0)
        self.pid_roll  = PIDController(kp=35.0, ki=0.0, kd=8.0,  min_out=-80.0,  max_out=80.0)
        self.pid_pitch = PIDController(kp=35.0, ki=0.0, kd=8.0,  min_out=-80.0,  max_out=80.0)
        self.pid_yaw   = PIDController(kp=10.0, ki=0.0, kd=2.0,  min_out=-50.0,  max_out=50.0)

        self.last_time = 0.0
        self.debug_counter = 0

        # Timer di sicurezza: se non arrivano comandi da nav6d, azzera il moto orizzontale
        self.safety_timer = self.create_timer(0.1, self.safety_check)

        self.get_logger().info("Inner Loop Controller (nav6d-compatible) Avviato!")

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

    def cmd_vel_callback(self, msg: Twist):
        # Comando di velocità body-frame da nav6d: vx avanti, vy laterale, vz verticale, yaw_rate
        self.last_cmd_vx = msg.linear.x
        self.last_cmd_vy = msg.linear.y
        self.last_cmd_vz = msg.linear.z
        self.last_cmd_yaw_rate = msg.angular.z
        self.last_cmd_vel_time = self.get_clock().now()

    def safety_check(self):
        if self.last_cmd_vel_time is None:
            return
        elapsed = (self.get_clock().now() - self.last_cmd_vel_time).nanoseconds * 1e-9
        if elapsed > self.cmd_vel_timeout:
            # Nessun comando recente da nav6d: azzera moto orizzontale per sicurezza,
            # mantieni la quota corrente (hover)
            self.last_cmd_vx = 0.0
            self.last_cmd_vy = 0.0
            self.last_cmd_vz = 0.0
            self.last_cmd_yaw_rate = 0.0

    def odom_callback(self, msg):
        current_time = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
        if self.last_time == 0.0:
            self.last_time = current_time
            return

        dt = current_time - self.last_time
        if dt <= 0.0:
            return
        self.last_time = current_time

        # --- LETTURA SENSORI ---
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        self.curr_z = msg.pose.pose.position.z

        q = msg.pose.pose.orientation
        curr_roll, curr_pitch, curr_yaw = self.quaternion_to_euler(q.w, q.x, q.y, q.z)

        # --- INTEGRAZIONE DEL COMANDO DI VELOCITA' IN SETPOINT DI ASSETTO/QUOTA ---
        # vx/vy sono body-frame: li convertiamo direttamente in target di
        # pitch/roll (stessa convenzione segno del controller originale:
        # pitch avanti positivo -> target_pitch, roll laterale -> -target_roll)
        self.target_pitch = self.k_vel_to_angle * self.last_cmd_vx
        self.target_roll = -self.k_vel_to_angle * self.last_cmd_vy

        # vz e yaw_rate li integriamo nel tempo per ottenere setpoint assoluti
        self.target_z += self.last_cmd_vz * dt
        self.target_yaw += self.last_cmd_yaw_rate * dt

        # --- LOOP INTERNO: CONTROLLO ASSETTO E ALTITUDINE (invariato) ---
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

        # --- MATRICE DI MIXING (invariata) ---
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

        self.debug_counter += 1
        if self.debug_counter >= 50:
            self.debug_counter = 0


def main(args=None):
    rclpy.init(args=args)
    node = InnerLoopController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()