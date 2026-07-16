import math
from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO


class SimpleKalmanFilter:
    """2D Kalman Filter (Constant Velocity Model) with adaptive R and outlier gating."""
    def __init__(self, base_r=0.5):
        self.x = np.zeros((4, 1))  # State: [x, y, vx, vy]
        self.P = np.eye(4) * 1000.0
        self.F = np.eye(4)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.base_r = base_r
        self.R = np.eye(2) * base_r
        self.Q = np.eye(4) * 0.1
        self.is_initialized = False

    def predict(self, dt):
        if not self.is_initialized or dt <= 0.0:
            return
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def set_measurement_noise(self, drone_ang_rate, drone_speed):
        scale = 1.0 + 4.0 * min(drone_ang_rate, 2.0) + 1.5 * min(drone_speed, 3.0)
        self.R = np.eye(2) * (self.base_r * scale)

    def mahalanobis_gate(self, z, gate_threshold=9.21):
        if not self.is_initialized:
            return True
        Z = np.array([[z[0]], [z[1]]])
        y = Z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        try:
            d2 = float(np.dot(np.dot(y.T, np.linalg.inv(S)), y))
        except np.linalg.LinAlgError:
            return True
        return d2 <= gate_threshold

    def update(self, z):
        Z = np.array([[z[0]], [z[1]]])
        if not self.is_initialized:
            self.x[0, 0] = z[0]
            self.x[1, 0] = z[1]
            self.is_initialized = True
            return

        y = Z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)


class OdomBuffer:
    """Interpolates position/attitude to the exact image timestamp."""
    def __init__(self, max_len=200):
        self.buf = deque(maxlen=max_len)

    def push(self, t, x, y, z, roll, pitch, yaw):
        self.buf.append((t, x, y, z, roll, pitch, yaw))

    @staticmethod
    def _lerp_angle(a0, a1, alpha):
        diff = math.atan2(math.sin(a1 - a0), math.cos(a1 - a0))
        return a0 + diff * alpha

    def query(self, t):
        if not self.buf: return None
        if t <= self.buf[0][0]: return self.buf[0][1:]
        if t >= self.buf[-1][0]: return self.buf[-1][1:]

        for i in range(1, len(self.buf)):
            t0, x0, y0, z0, r0, p0, yw0 = self.buf[i - 1]
            t1, x1, y1, z1, r1, p1, yw1 = self.buf[i]
            if t0 <= t <= t1:
                alpha = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                x = x0 + (x1 - x0) * alpha
                y = y0 + (y1 - y0) * alpha
                z = z0 + (z1 - z0) * alpha
                roll = self._lerp_angle(r0, r1, alpha)
                pitch = self._lerp_angle(p0, p1, alpha)
                yaw = self._lerp_angle(yw0, yw1, alpha)
                return (x, y, z, roll, pitch, yaw)
        return self.buf[-1][1:]

    def angular_rate(self):
        if len(self.buf) < 2: return 0.0
        t0, _, _, _, r0, p0, _ = self.buf[-2]
        t1, _, _, _, r1, p1, _ = self.buf[-1]
        dt = t1 - t0
        if dt <= 1e-3: return 0.0
        droll = self._lerp_angle(r0, r1, 1.0) - r0
        dpitch = self._lerp_angle(p0, p1, 1.0) - p0
        return math.hypot(droll, dpitch) / dt

    def linear_speed(self):
        if len(self.buf) < 2: return 0.0
        t0, x0, y0, z0, *_ = self.buf[-2]
        t1, x1, y1, z1, *_ = self.buf[-1]
        dt = t1 - t0
        if dt <= 1e-3: return 0.0
        return math.dist((x0, y0, z0), (x1, y1, z1)) / dt


class YoloTrackerNode(Node):
    def __init__(self):
        super().__init__('yolo_tracker_node')

        self.cv_bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')

        self.cam_width = 640
        self.cam_height = 480
        self.fov_h = 1.50098
        self.cam_pitch = 0.349
        self.focal_length = (self.cam_width / 2.0) / math.tan(self.fov_h / 2.0)
        self.cam_offset_body = np.array([0.17, 0.0, -0.01])

        self.odom_buffer = OdomBuffer(max_len=200)
        self.have_odom = False

        self.person_x = None
        self.person_y = None
        
        # Path extracted from RAW measurements
        self.path_history = []

        # DRONE TRAJECTORY LOGGING
        self.drone_history_x = []
        self.drone_history_y = []

        # ACTOR'S REAL TRAJECTORY (Extracted from XML)
        # Removing duplicates caused by in-place rotations and keeping the order
        self.actor_real_x = [9.0, 9.0, 0.0, 0.0, -1.5, -1.5, -9.0, -9.0, -1.5, -1.5, 1.0, 1.0, 9.0]
        self.actor_real_y = [-21.5, 1.0, 1.0, 13.0, 13.0, 23.5, 23.5, -21.5, -21.5, -14.0, -14.0, -21.5, -21.5]

        self.is_moving = False
        self.current_goal_x = None
        self.current_goal_y = None

        self.kf = SimpleKalmanFilter(base_r=0.5)
        self.last_meas_time = None
        self.lost_frames = 0

        self.bbox_ema_alpha = 0.5
        self._u_smooth = None
        self._v_smooth = None

        self.activation_distance = 5.5
        self.stop_distance = 0.5
        self.goal_reach_tolerance = 0.1

        self.goal_delay = 2.0             # Seconds to wait before moving
        self.activation_start_time = None # Timestamp when the threshold was crossed

        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/planner_goal', 10)

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 8))
        self.setup_plot()
        self.plot_timer = self.create_timer(0.5, self.update_plot)

        self.get_logger().info('YOLO Tracker started (Raw/Farthest if visible, Kalman if lost)!')

    def setup_plot(self):
        self.ax.clear()
        self.ax.set_xlim(-15, 15)
        self.ax.set_ylim(-25, 25)
        self.ax.set_title("Mixed Tracking: Ray-Casting & Kalman")
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")
        self.ax.grid(True)

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

    def odom_callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        q = msg.pose.pose.orientation
        roll, pitch, yaw = self.quaternion_to_euler(q.w, q.x, q.y, q.z)

        # Record drone trajectory
        self.drone_history_x.append(msg.pose.pose.position.x)
        self.drone_history_y.append(msg.pose.pose.position.y)

        self.odom_buffer.push(
            t,
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            roll, pitch, yaw,
        )
        self.have_odom = True

        if self.is_moving and self.current_goal_x is not None and self.current_goal_y is not None:
            dist_to_goal = math.hypot(
                msg.pose.pose.position.x - self.current_goal_x,
                msg.pose.pose.position.y - self.current_goal_y,
            )
            if dist_to_goal <= self.goal_reach_tolerance:
                self.get_logger().info('Goal reached! Waiting for new instructions...')
                self.is_moving = False
                self.current_goal_x = None
                self.current_goal_y = None
                self.lost_frames = 0
                self.activation_start_time = None 

    def camera_callback(self, msg):
        if not self.have_odom:
            return

        img_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if img_time <= 0.0:
            img_time = self.get_clock().now().nanoseconds * 1e-9

        if self.last_meas_time is None:
            self.last_meas_time = img_time
            return
        dt = img_time - self.last_meas_time
        if dt <= 0.0:
            return
        self.last_meas_time = img_time

        pose = self.odom_buffer.query(img_time)
        if pose is None:
            return
        drone_x, drone_y, drone_z, drone_roll, drone_pitch, drone_yaw = pose

        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(cv_image, classes=[0], conf=0.5, verbose=False)

        person_detected = False
        meas_x, meas_y = None, None

        if len(results) > 0 and len(results[0].boxes) > 0:
            best_box = results[0].boxes[0]
            x1, y1, x2, y2 = best_box.xyxy[0].tolist()

            u_raw = (x1 + x2) / 2.0
            v_raw = y2

            if self._u_smooth is None:
                self._u_smooth, self._v_smooth = u_raw, v_raw
            else:
                a = self.bbox_ema_alpha
                self._u_smooth = a * u_raw + (1 - a) * self._u_smooth
                self._v_smooth = a * v_raw + (1 - a) * self._v_smooth
            u, v = self._u_smooth, self._v_smooth

            u_c = u - (self.cam_width / 2.0)
            v_c = v - (self.cam_height / 2.0)

            ray_opt = np.array([u_c / self.focal_length, v_c / self.focal_length, 1.0])
            ray_cam = np.array([ray_opt[2], -ray_opt[0], -ray_opt[1]])

            cp_m = math.cos(self.cam_pitch)
            sp_m = math.sin(self.cam_pitch)
            R_mount = np.array([
                [cp_m, 0, sp_m],
                [0, 1, 0],
                [-sp_m, 0, cp_m]
            ])
            ray_body = np.dot(R_mount, ray_cam)

            cr = math.cos(drone_roll)
            sr = math.sin(drone_roll)
            cp_d = math.cos(drone_pitch)
            sp_d = math.sin(drone_pitch)
            cy = math.cos(drone_yaw)
            sy = math.sin(drone_yaw)

            R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            R_y = np.array([[cp_d, 0, sp_d], [0, 1, 0], [-sp_d, 0, cp_d]])
            R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

            R_drone = np.dot(R_z, np.dot(R_y, R_x))
            ray_world = np.dot(R_drone, ray_body)

            cam_offset_world = np.dot(R_drone, self.cam_offset_body)
            cam_x = drone_x + cam_offset_world[0]
            cam_y = drone_y + cam_offset_world[1]
            cam_z = drone_z + cam_offset_world[2]

            D_x, D_y, D_z = ray_world[0], ray_world[1], ray_world[2]

            if D_z < -0.15:
                t = -cam_z / D_z
                meas_x = cam_x + t * D_x
                meas_y = cam_y + t * D_y

                ang_rate = self.odom_buffer.angular_rate()
                lin_speed = self.odom_buffer.linear_speed()
                self.kf.set_measurement_noise(ang_rate, lin_speed)

                if self.kf.mahalanobis_gate([meas_x, meas_y]):
                    self.kf.predict(dt)
                    self.kf.update([meas_x, meas_y])
                    
                    self.person_x = meas_x
                    self.person_y = meas_y
                    person_detected = True
                    self.lost_frames = 0
                else:
                    self.get_logger().debug('Measurement discarded: outlier compared to Kalman.')

        if not person_detected:
            self.person_x = None
            self.person_y = None
            if self.kf.is_initialized:
                self.kf.predict(dt)
                self.lost_frames += 1

        # --- PHASE 1: PATH POPULATION ---
        if person_detected and meas_x is not None and meas_y is not None:
            if not self.path_history:
                self.path_history.append((meas_x, meas_y))
            else:
                last_x, last_y = self.path_history[-1]
                # Add only if moved a minimum distance to prevent memory bloat
                if math.hypot(last_x - meas_x, last_y - meas_y) > 0.2:
                    self.path_history.append((meas_x, meas_y))

        # --- PHASE 2: RAW MEMORY CLEANUP ---
        if self.path_history:
            min_d = float('inf')
            min_idx = 0
            for i, (px, py) in enumerate(self.path_history):
                d = math.hypot(px - drone_x, py - drone_y)
                if d < min_d:
                    min_d = d
                    min_idx = i
            self.path_history = self.path_history[min_idx:]

        # --- PHASE 3: TARGET SELECTION AND GOAL DISPATCH ---
        if not self.is_moving:
            target_x = None
            target_y = None
            tracking_mode = ""

            if person_detected and self.path_history:
                max_dist = -1.0
                for px, py in self.path_history:
                    d = math.hypot(px - drone_x, py - drone_y)
                    if d > max_dist:
                        max_dist = d
                        target_x, target_y = px, py
                tracking_mode = "RAW MEASUREMENT (Farthest)"

            elif not person_detected and self.kf.is_initialized and self.lost_frames < 30:
                target_x = float(self.kf.x[0, 0])
                target_y = float(self.kf.x[1, 0])
                tracking_mode = "KALMAN (Prediction)"

            if target_x is not None and target_y is not None:
                dist_to_target = math.hypot(target_x - drone_x, target_y - drone_y)

                if dist_to_target >= self.activation_distance:
                    if self.activation_start_time is None:
                        self.activation_start_time = img_time
                        self.get_logger().info(f'Distance > 5.5m [{tracking_mode}]. Flying in {self.goal_delay}s...')
                    
                    elif (img_time - self.activation_start_time) >= self.goal_delay:
                        target_yaw = math.atan2(target_y - drone_y, target_x - drone_x)
                        goal_x = target_x - self.stop_distance * math.cos(target_yaw)
                        goal_y = target_y - self.stop_distance * math.sin(target_yaw)
                        
                        self.send_goal(goal_x, goal_y, target_yaw)
                        self.current_goal_x = goal_x
                        self.current_goal_y = goal_y
                        self.is_moving = True
                        
                        self.activation_start_time = None
                        self.get_logger().info(f'Departing! Tracking target based on {tracking_mode}.')
                
                else:
                    if self.activation_start_time is not None:
                        self.activation_start_time = None
                        self.get_logger().info('Target returned below 5.5m, departure canceled.')

    def send_goal(self, gx, gy, target_yaw):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'

        msg.pose.position.x = float(gx)
        msg.pose.position.y = float(gy)
        msg.pose.position.z = 3.0

        msg.pose.orientation.z = math.sin(target_yaw / 2.0)
        msg.pose.orientation.w = math.cos(target_yaw / 2.0)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0

        self.goal_pub.publish(msg)
        self.get_logger().info(f"Goal Sent -> X:{gx:.2f}, Y:{gy:.2f}")

    def update_plot(self):
        self.setup_plot()

        if len(self.path_history) > 0:
            hx = [p[0] for p in self.path_history]
            hy = [p[1] for p in self.path_history]
            self.ax.plot(hx, hy, 'r.', markersize=4, alpha=0.5, label='Raw Path')

        if self.person_x is not None and self.person_y is not None:
            self.ax.plot(self.person_x, self.person_y, 'ro', markersize=8, label='YOLO Position')

        if self.kf.is_initialized:
            kf_x = self.kf.x[0, 0]
            kf_y = self.kf.x[1, 0]
            self.ax.plot(kf_x, kf_y, 'gx', markersize=6, label='Kalman Filter')

        if self.have_odom and self.odom_buffer.buf:
            _, dx, dy, dz, dr, dp, dyaw = self.odom_buffer.buf[-1]
            self.ax.plot(dx, dy, 'bo', markersize=8, label='Drone')
            self.ax.arrow(dx, dy,
                          math.cos(dyaw) * 1.5, math.sin(dyaw) * 1.5,
                          head_width=0.5, head_length=0.5, fc='b', ec='b')

            circle = plt.Circle((dx, dy), self.activation_distance, color='gray', fill=False, linestyle='--')
            self.ax.add_patch(circle)

        if self.is_moving and self.current_goal_x is not None:
            self.ax.plot(self.current_goal_x, self.current_goal_y, 'gX', markersize=10, label='Current Goal')
            self.ax.text(-14, -23, "State: TRAVELING", color='green', weight='bold')
        
        elif self.activation_start_time is not None:
            current_time = self.get_clock().now().nanoseconds * 1e-9
            time_elapsed = current_time - self.activation_start_time
            countdown = max(0.0, self.goal_delay - time_elapsed)
            self.ax.text(-14, -23, f"State: DEPARTING ({countdown:.1f}s)", color='red', weight='bold')
        
        else:
            self.ax.text(-14, -23, "State: WAITING", color='orange', weight='bold')

        self.ax.legend(loc='upper right')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def generate_final_report(self):
        """Generates a summary plot with global trajectories upon node shutdown."""
        self.get_logger().info("Generating trajectory report...")
        
        plt.ioff() # Disable interactive mode
        plt.close('all') # Safely close all matplotlib windows
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title("Final Path: Drone vs Person", fontsize=14, fontweight='bold')
        ax.set_xlabel("X [meters]")
        ax.set_ylabel("Y [meters]")
        ax.grid(True)

        # Plot Real Person Trajectory (Waypoints)
        ax.plot(self.actor_real_x, self.actor_real_y, 'r--', linewidth=2, label="Person Trajectory (Real)")
        ax.scatter(self.actor_real_x, self.actor_real_y, color='red', marker='x', s=50)

        # Plot Real Drone Trajectory (Odometry)
        if self.drone_history_x and self.drone_history_y:
            ax.plot(self.drone_history_x, self.drone_history_y, 'b-', linewidth=2, label="Drone Trajectory (Odom)")

        ax.legend(loc='best')
        ax.axis('equal') # Keep geometric proportions to visualize the map correctly
        
        # Show the window blockingly until it is closed
        plt.show(block=True)


def main(args=None):
    rclpy.init(args=args)
    node = YoloTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupt command received (Ctrl+C).')
    finally:
        # When the node stops via Ctrl+C, generate the closing plot
        node.generate_final_report()
        
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()