#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import threading
import numpy as np
from scipy.interpolate import CubicSpline

class WaypointPlotterNode(Node):
    def __init__(self):
        super().__init__('waypoint_plotter_node')
        
        self.subscription = self.create_subscription(
            Path,
            '/planner/path',
            self.path_callback,
            10
        )
        
        self.current_path = []
        self.new_path_received = False
        self.lock = threading.Lock()
        
        self.get_logger().info("3D Trajectory Plotter Node Started! Waiting for paths...")

    def path_callback(self, msg):
        if not msg.poses:
            return

        # 1. Extract XYZ coordinates
        extracted_points = []
        for pose_stamped in msg.poses:
            extracted_points.append((
                pose_stamped.pose.position.x,
                pose_stamped.pose.position.y,
                pose_stamped.pose.position.z
            ))

        with self.lock:
            # Anti-flickering check based on final target
            if self.current_path:
                last_wp_new = extracted_points[-1]
                last_wp_curr = self.current_path[-1]
                
                dist_goal = math.sqrt(
                    (last_wp_new[0] - last_wp_curr[0])**2 +
                    (last_wp_new[1] - last_wp_curr[1])**2 +
                    (last_wp_new[2] - last_wp_curr[2])**2
                )
                
                if dist_goal < 0.1:
                    return
            
            self.current_path = extracted_points
            self.new_path_received = True
            self.get_logger().info(f"New path received ({len(extracted_points)} wps). Calculating spline...")

def main(args=None):
    rclpy.init(args=args)
    node = WaypointPlotterNode()

    # Spin in background
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    plt.ion()
    fig = None

    try:
        while rclpy.ok():
            if node.new_path_received:
                with node.lock:
                    node.new_path_received = False
                    waypoints = list(node.current_path)

                if len(waypoints) >= 2:
                    # --- CONTINUOUS TRAJECTORY GENERATION ---
                    pts = np.asarray(waypoints, dtype=float)
                    
                    # Clean coincident points to avoid spline singularities
                    cleaned = [pts[0]]
                    for p in pts[1:]:
                        if np.linalg.norm(p - cleaned[-1]) > 1e-3:
                            cleaned.append(p)
                    pts = np.array(cleaned)
                    
                    if len(pts) >= 2:
                        # Time parameters matching the trajectory generator
                        cruise_speed = 0.8
                        min_segment_time = 0.4
                        
                        seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
                        seg_times = np.maximum(seg_lengths / cruise_speed, min_segment_time)
                        t = np.concatenate(([0.0], np.cumsum(seg_times)))
                        
                        # Cubic Spline Generation
                        spline_x = CubicSpline(t, pts[:, 0], bc_type='clamped')
                        spline_y = CubicSpline(t, pts[:, 1], bc_type='clamped')
                        spline_z = CubicSpline(t, pts[:, 2], bc_type='clamped')
                        
                        # High-density sampling for plotting (150 points)
                        t_dense = np.linspace(0, t[-1], 150)
                        xs_dense = spline_x(t_dense)
                        ys_dense = spline_y(t_dense)
                        zs_dense = spline_z(t_dense)

                        # Matplotlib window management
                        if fig is not None and plt.fignum_exists(fig.number):
                            plt.close(fig)
                        
                        fig = plt.figure(figsize=(10, 7))
                        ax = fig.add_subplot(111, projection='3d')
                        
                        # PLOT 1: Generated Trajectory
                        ax.plot(xs_dense, ys_dense, zs_dense, label='Trajectory (C2 Spline)', color='#ff7f0e', linewidth=2.5)
                        
                        # PLOT 2: Original A* Waypoints in background
                        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color='gray', s=20, alpha=0.4, label='Voxel Waypoints')
                        
                        # PLOT 3: Start and Goal markers
                        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color='green', s=100, label='Start')
                        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color='gold', s=150, marker='*', label='Goal Target')
                        
                        # --- YAW VECTOR CALCULATION VIA FIRST DERIVATIVE ---
                        step = max(1, len(t_dense) // 15)  # About 15 arrows total
                        us, vs, ws = [], [], []
                        
                        for i in range(0, len(t_dense), step):
                            vx = spline_x(t_dense[i], 1)
                            vy = spline_y(t_dense[i], 1)
                            
                            yaw = math.atan2(vy, vx)
                            us.append(math.cos(yaw))
                            vs.append(math.sin(yaw))
                            ws.append(0.0)
                            
                        # PLOT 4: Orientation Vectors
                        ax.quiver(xs_dense[::step], ys_dense[::step], zs_dense[::step], 
                                  us, vs, ws, length=0.4, color='red', normalize=True, label='Yaw Tangent')
                        
                        # Aesthetics and labels
                        ax.set_xlabel('X Axis [m]')
                        ax.set_ylabel('Y Axis [m]')
                        ax.set_zlabel('Altitude Z [m]')
                        ax.set_title('3D Continuous Trajectory Planning')
                        ax.legend(loc='upper left')
                        ax.grid(True)
                        
                        plt.show()
            
            plt.pause(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        plt.close('all')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()