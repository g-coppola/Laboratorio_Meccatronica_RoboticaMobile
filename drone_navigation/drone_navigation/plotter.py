#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import threading
import numpy as np
from scipy.interpolate import CubicSpline

class WaypointPlotterNode(Node):
    def __init__(self):
        super().__init__('waypoint_plotter_node')
        
        # Subscription to the path planned by A*
        self.path_sub = self.create_subscription(
            Path,
            '/planner/path',
            self.path_callback,
            10
        )
        
        # Subscription to odometry to monitor real-time position
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        self.current_path = []
        self.real_trajectory_history = []  # Stores the real trajectory flown by the drone (x, y, z, yaw)
        self.new_path_received = False
        self.plot_triggered = False        # Flag to avoid continuously reopening the plot after arrival
        self.lock = threading.Lock()
        
        # Current drone coordinates and yaw
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_z = 0.0
        self.curr_yaw = 0.0
        
        self.get_logger().info("3D Trajectory Plotter started! The plot will only be shown upon reaching the goal.")

    def odom_callback(self, msg):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        self.curr_z = msg.pose.pose.position.z
        
        # Estrazione dello Yaw reale dai quaternioni
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.curr_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        with self.lock:
            # Save a point of the real trajectory every 5 cm to avoid overloading memory
            if not self.real_trajectory_history:
                # Aggiunto curr_yaw alla tupla
                self.real_trajectory_history.append((self.curr_x, self.curr_y, self.curr_z, self.curr_yaw))
            else:
                last_pos = self.real_trajectory_history[-1]
                dist = math.sqrt((self.curr_x - last_pos[0])**2 + (self.curr_y - last_pos[1])**2 + (self.curr_z - last_pos[2])**2)
                if dist > 0.05:
                    # Aggiunto curr_yaw alla tupla
                    self.real_trajectory_history.append((self.curr_x, self.curr_y, self.curr_z, self.curr_yaw))
            
            # If a valid path exists and we haven't plotted this goal yet
            if self.current_path and not self.plot_triggered:
                goal_wp = self.current_path[-1]
                
                # Calculate remaining distance to the last waypoint of the path (the final goal)
                dist_to_goal = math.sqrt(
                    (self.curr_x - goal_wp[0])**2 +
                    (self.curr_y - goal_wp[1])**2 +
                    (self.curr_z - goal_wp[2])**2
                )
                
                # Goal arrival threshold (e.g., 0.08 meters, matching the planner's goal_reach_threshold)
                if dist_to_goal < 0.08:
                    self.plot_triggered = True
                    self.new_path_received = True  # Wake up the main loop in the main thread to trigger the plot
                    self.get_logger().info("Goal successfully reached! Generating the final plot...")

    def path_callback(self, msg):
        if not msg.poses:
            return

        # Extract XYZ coordinates from the Path message
        extracted_points = []
        for pose_stamped in msg.poses:
            extracted_points.append((
                pose_stamped.pose.position.x,
                pose_stamped.pose.position.y,
                pose_stamped.pose.position.z
            ))

        with self.lock:
            if self.current_path:
                last_wp_new = extracted_points[-1]
                last_wp_curr = self.current_path[-1]
                
                dist_goal = math.sqrt(
                    (last_wp_new[0] - last_wp_curr[0])**2 +
                    (last_wp_new[1] - last_wp_curr[1])**2 +
                    (last_wp_new[2] - last_wp_curr[2])**2
                )
                
                # If the new goal is identical to the previous one (anti-flickering), ignore republication
                if dist_goal < 0.1:
                    return
            
            self.current_path = extracted_points
            self.plot_triggered = False  # Reset the trigger state to allow plotting for the new goal
            # Reset the real track for the new flight, includendo lo yaw
            self.real_trajectory_history = [(self.curr_x, self.curr_y, self.curr_z, self.curr_yaw)]  
            self.get_logger().info(f"Received new path with {len(extracted_points)} waypoints. Plot in stand-by until arrival.")

def main(args=None):
    rclpy.init(args=args)
    node = WaypointPlotterNode()

    # ROS spin runs in the background to gather data from topics without blocking the GUI
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    plt.ion()
    fig = None

    try:
        while rclpy.ok():
            # The plotting block triggers only when odom_callback sets new_path_received to True (upon arrival)
            if node.new_path_received:
                with node.lock:
                    node.new_path_received = False
                    waypoints = list(node.current_path)
                    real_history = list(node.real_trajectory_history)

                if len(waypoints) >= 2:
                    pts = np.asarray(waypoints, dtype=float)
                    
                    # Clean up overlapping points
                    cleaned = [pts[0]]
                    for p in pts[1:]:
                        if np.linalg.norm(p - cleaned[-1]) > 1e-3:
                            cleaned.append(p)
                    pts = np.array(cleaned)
                    
                    if len(pts) >= 2:
                        # Time parameters to reconstruct the theoretical spline calculated by the generator
                        cruise_speed = 0.3
                        min_segment_time = 0.4
                        
                        seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
                        seg_times = np.maximum(seg_lengths / cruise_speed, min_segment_time)
                        t = np.concatenate(([0.0], np.cumsum(seg_times)))
                        
                        # Generate Clamped Cubic Spline (Theoretical)
                        spline_x = CubicSpline(t, pts[:, 0], bc_type='clamped')
                        spline_y = CubicSpline(t, pts[:, 1], bc_type='clamped')
                        spline_z = CubicSpline(t, pts[:, 2], bc_type='clamped')
                        
                        t_dense = np.linspace(0, t[-1], 150)
                        xs_dense = spline_x(t_dense)
                        ys_dense = spline_y(t_dense)
                        zs_dense = spline_z(t_dense)

                        # Matplotlib figure management
                        if fig is not None and plt.fignum_exists(fig.number):
                            plt.close(fig)
                        
                        fig = plt.figure(figsize=(10, 7))
                        ax = fig.add_subplot(111, projection='3d')
                        
                        # PLOT 1: Theoretical reference trajectory (C2 Spline)
                        ax.plot(xs_dense, ys_dense, zs_dense, label='Reference Trajectory (Spline)', color='#ff7f0e', linewidth=2.5, linestyle='--')
                        
                        # PLOT 2 & 6: Real trajectory actually covered by the drone (Odom) + REAL YAW
                        if len(real_history) > 1:
                            real_pts = np.array(real_history)
                            ax.plot(real_pts[:, 0], real_pts[:, 1], real_pts[:, 2], label='Real Trajectory (Odometry)', color='blue', linewidth=2.0)
                            
                            # Calcola i vettori per lo yaw reale
                            real_step = max(1, len(real_pts) // 15)  # Campiona i punti per non sovraffollare il grafico
                            real_us = np.cos(real_pts[::real_step, 3]) # Indice 3 è lo yaw
                            real_vs = np.sin(real_pts[::real_step, 3])
                            real_ws = np.zeros_like(real_us)
                            
                            # Quiver per lo Yaw Reale (Viola)
                            ax.quiver(real_pts[::real_step, 0], real_pts[::real_step, 1], real_pts[::real_step, 2], 
                                      real_us, real_vs, real_ws, length=0.4, color='purple', normalize=True, label='Real Yaw (Odom)')
                        
                        # PLOT 3: Discrete waypoints extracted from the A* voxel grid
                        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color='gray', s=20, alpha=0.4, label='A* Voxel Waypoints')
                        
                        # PLOT 4: Start and End (Goal Target) indicators
                        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color='green', s=100, label='Start')
                        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color='gold', s=150, marker='*', label='Goal Reached')
                        
                        # Calculate orientation vectors (Tangent Yaw) on the theoretical spline
                        step = max(1, len(t_dense) // 15)
                        us, vs, ws = [], [], []
                        for i in range(0, len(t_dense), step):
                            vx = spline_x(t_dense[i], 1)
                            vy = spline_y(t_dense[i], 1)
                            yaw = math.atan2(vy, vx)
                            us.append(math.cos(yaw))
                            vs.append(math.sin(yaw))
                            ws.append(0.0)
                            
                        # PLOT 5: Yaw Vectors (Theorici)
                        ax.quiver(xs_dense[::step], ys_dense[::step], zs_dense[::step], 
                                  us, vs, ws, length=0.4, color='red', normalize=True, label='Theoretical Yaw Tangent')
                        
                        # Aesthetics, grid, and legend
                        ax.set_xlabel('X Axis [m]')
                        ax.set_ylabel('Y Axis [m]')
                        ax.set_zlabel('Z Altitude [m]')
                        ax.set_title('Final Trajectory Analysis: Reference vs Real Odometry')
                        ax.legend(loc='upper left')
                        ax.grid(True)
                        
                        # Show the window and block execution until the user manually closes it
                        plt.show(block=True)
            
            plt.pause(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        plt.close('all')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()