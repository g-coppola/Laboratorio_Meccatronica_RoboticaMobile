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
        super().__init__('plotter_node')
        
        self.path_sub = self.create_subscription(Path, '/planner/path', self.path_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        self.current_path = []             # Conterrà tuple: (x, y, z, roll, pitch, yaw)
        self.real_trajectory_history = []  # Conterrà tuple: (t, x, y, z, roll, pitch, yaw)
        
        self.new_path_received = False
        self.plot_triggered = False        
        self.lock = threading.Lock()
        
        # Variabili di stato correnti
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_z = 0.0
        self.curr_yaw = 0.0
        
        # Gestione tempo per i grafici
        self.start_time = None
        self.last_log_time = 0.0
        
        self.get_logger().info("3D & 2D Trajectory Plotter started! I grafici appariranno all'arrivo.")

    def quaternion_to_euler(self, q):
        """Converte un quaternione in Roll, Pitch, Yaw (in radianti)"""
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (q.w * q.y - q.z * q.x)
        pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)
        
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw

    def odom_callback(self, msg):
        # Calcolo del tempo trascorso dal primo messaggio ricevuto
        current_time_raw = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
        if self.start_time is None:
            self.start_time = current_time_raw
        t = current_time_raw - self.start_time

        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        self.curr_z = msg.pose.pose.position.z
        
        roll, pitch, self.curr_yaw = self.quaternion_to_euler(msg.pose.pose.orientation)
        
        with self.lock:
            # Salviamo un punto della traiettoria reale ogni 0.1 secondi (10 Hz) per il grafico temporale
            if not self.real_trajectory_history or (t - self.last_log_time >= 0.1):
                self.real_trajectory_history.append((t, self.curr_x, self.curr_y, self.curr_z, roll, pitch, self.curr_yaw))
                self.last_log_time = t
            
            # Controllo arrivo a destinazione
            if self.current_path and not self.plot_triggered:
                goal_wp = self.current_path[-1] # Target X, Y, Z
                dist_to_goal = math.sqrt(
                    (self.curr_x - goal_wp[0])**2 +
                    (self.curr_y - goal_wp[1])**2 +
                    (self.curr_z - goal_wp[2])**2
                )
                
                if dist_to_goal < 0.08:
                    self.plot_triggered = True
                    self.new_path_received = True 
                    self.get_logger().info("Goal successfully reached! Generating the final plots...")

    def path_callback(self, msg):
        if not msg.poses: return

        extracted_points = []
        for pose_stamped in msg.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            z = pose_stamped.pose.position.z
            r, p, yw = self.quaternion_to_euler(pose_stamped.pose.orientation)
            extracted_points.append((x, y, z, r, p, yw))

        with self.lock:
            if self.current_path:
                last_wp_new = extracted_points[-1]
                last_wp_curr = self.current_path[-1]
                
                dist_goal = math.sqrt(
                    (last_wp_new[0] - last_wp_curr[0])**2 +
                    (last_wp_new[1] - last_wp_curr[1])**2 +
                    (last_wp_new[2] - last_wp_curr[2])**2
                )
                
                if dist_goal < 0.1: return
            
            self.current_path = extracted_points
            self.plot_triggered = False
            
            # Al reset del path, riavviamo anche la storia e il timer
            self.start_time = None
            self.real_trajectory_history.clear()
            
            self.get_logger().info(f"Received new path with {len(extracted_points)} waypoints. Plot in stand-by until arrival.")

def main(args=None):
    rclpy.init(args=args)
    node = WaypointPlotterNode()

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    plt.ion()
    fig3d = None
    fig2d = None

    try:
        while rclpy.ok():
            if node.new_path_received:
                with node.lock:
                    node.new_path_received = False
                    waypoints = list(node.current_path)
                    real_history = list(node.real_trajectory_history)

                if len(waypoints) >= 2 and len(real_history) > 1:
                    pts = np.asarray(waypoints, dtype=float)
                    real_pts = np.array(real_history)
                    
                    # pts[:,0]=X, pts[:,1]=Y, pts[:,2]=Z, pts[:,3]=Roll, pts[:,4]=Pitch, pts[:,5]=Yaw
                    
                    # Clean up overlapping points per la Spline Teorica
                    cleaned = [pts[0]]
                    for p in pts[1:]:
                        if np.linalg.norm(p[0:3] - cleaned[-1][0:3]) > 1e-3:
                            cleaned.append(p)
                    pts = np.array(cleaned)
                    
                    if len(pts) >= 2:
                        # --- CALCOLI SPLINE TEORICA ---
                        cruise_speed = 0.3
                        min_segment_time = 0.4
                        
                        seg_lengths = np.linalg.norm(np.diff(pts[:, 0:3], axis=0), axis=1)
                        seg_times = np.maximum(seg_lengths / cruise_speed, min_segment_time)
                        t_wp = np.concatenate(([0.0], np.cumsum(seg_times))) # Tempi teorici ai waypoint
                        
                        spline_x = CubicSpline(t_wp, pts[:, 0], bc_type='clamped')
                        spline_y = CubicSpline(t_wp, pts[:, 1], bc_type='clamped')
                        spline_z = CubicSpline(t_wp, pts[:, 2], bc_type='clamped')
                        
                        t_dense = np.linspace(0, t_wp[-1], 150)
                        xs_dense = spline_x(t_dense)
                        ys_dense = spline_y(t_dense)
                        zs_dense = spline_z(t_dense)
                        
                        # Yaw Teorico Tangente dalla derivata della Spline
                        vx_dense = spline_x(t_dense, 1)
                        vy_dense = spline_y(t_dense, 1)
                        yaw_theor_dense = np.arctan2(vy_dense, vx_dense)
                        
                        # --- GESTIONE FIGURE ---
                        if fig3d is not None and plt.fignum_exists(fig3d.number): plt.close(fig3d)
                        if fig2d is not None and plt.fignum_exists(fig2d.number): plt.close(fig2d)
                        
                        # ==========================================
                        # FIGURA 1: PLOT 3D SPAZIALE (Come prima)
                        # ==========================================
                        fig3d = plt.figure(figsize=(10, 7))
                        ax3d = fig3d.add_subplot(111, projection='3d')
                        
                        ax3d.plot(xs_dense, ys_dense, zs_dense, label='Reference (Spline)', color='#ff7f0e', linewidth=2.5, linestyle='--')
                        ax3d.plot(real_pts[:, 1], real_pts[:, 2], real_pts[:, 3], label='Real (Odometry)', color='blue', linewidth=2.0)
                        
                        # Vettori Yaw reali
                        real_step = max(1, len(real_pts) // 15)
                        ax3d.quiver(real_pts[::real_step, 1], real_pts[::real_step, 2], real_pts[::real_step, 3], 
                                    np.cos(real_pts[::real_step, 6]), np.sin(real_pts[::real_step, 6]), np.zeros_like(real_pts[::real_step, 6]), 
                                    length=0.4, color='purple', normalize=True, label='Real Yaw')
                        
                        ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color='gray', s=20, alpha=0.4, label='Waypoints')
                        ax3d.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color='green', s=100, label='Start')
                        ax3d.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color='gold', s=150, marker='*', label='Goal')
                        
                        # Fix per aspect ratio 1:1:1
                        x_limits, y_limits, z_limits = ax3d.get_xlim3d(), ax3d.get_ylim3d(), ax3d.get_zlim3d()
                        max_range = max([abs(x_limits[1]-x_limits[0]), abs(y_limits[1]-y_limits[0]), abs(z_limits[1]-z_limits[0])])
                        x_mid, y_mid, z_mid = np.mean(x_limits), np.mean(y_limits), np.mean(z_limits)
                        ax3d.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
                        ax3d.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
                        ax3d.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])
                        
                        ax3d.set_xlabel('X [m]'); ax3d.set_ylabel('Y [m]'); ax3d.set_zlabel('Z [m]')
                        ax3d.set_title('3D Trajectory Analysis')
                        ax3d.legend(loc='upper left'); ax3d.grid(True)

                        # ==========================================
                        # FIGURA 2: PLOT TIME-SERIES (6 Subplots)
                        # ==========================================
                        fig2d, axs = plt.subplots(3, 2, figsize=(14, 10))
                        fig2d.suptitle('Dinamica di Volo nel Tempo: Riferimento vs Reale', fontsize=16, fontweight='bold')
                        
                        t_real = real_pts[:, 0]
                        
                        # -- PLOT X, Y, Z --
                        titles_pos = ['Posizione X', 'Posizione Y', 'Altitudine Z']
                        y_labels_pos = ['X [m]', 'Y [m]', 'Z [m]']
                        for i in range(3):
                            axs[i, 0].plot(t_real, real_pts[:, i+1], 'b-', linewidth=2, label='Reale (Odom)')
                            axs[i, 0].plot(t_dense, [xs_dense, ys_dense, zs_dense][i], 'r--', linewidth=2, label='Teorico (Spline)')
                            axs[i, 0].set_title(titles_pos[i])
                            axs[i, 0].set_ylabel(y_labels_pos[i])
                            axs[i, 0].grid(True, linestyle=':', alpha=0.7)
                            axs[i, 0].legend()

                        # -- PLOT ROLL, PITCH, YAW --
                        # Estrazione e conversione in gradi (utilizzando unwrap per evitare salti -180/180)
                        real_roll_deg  = np.degrees(np.unwrap(real_pts[:, 4]))
                        real_pitch_deg = np.degrees(np.unwrap(real_pts[:, 5]))
                        real_yaw_deg   = np.degrees(np.unwrap(real_pts[:, 6]))
                        
                        theor_yaw_deg  = np.degrees(np.unwrap(yaw_theor_dense))
                        
                        # Plot Roll
                        axs[0, 1].plot(t_real, real_roll_deg, 'm-', linewidth=2, label='Reale')
                        # I planner geometrici A* spesso non calcolano Roll/Pitch, plottiamo i WP teorici come reference
                        axs[0, 1].step(t_wp, np.degrees(np.unwrap(pts[:, 3])), 'k--', where='post', alpha=0.5, label='Reference WP')
                        axs[0, 1].set_title('Angolo di Rollio')
                        axs[0, 1].set_ylabel('Roll [deg]')
                        axs[0, 1].grid(True, linestyle=':', alpha=0.7)
                        axs[0, 1].legend()

                        # Plot Pitch
                        axs[1, 1].plot(t_real, real_pitch_deg, 'm-', linewidth=2, label='Reale')
                        axs[1, 1].step(t_wp, np.degrees(np.unwrap(pts[:, 4])), 'k--', where='post', alpha=0.5, label='Reference WP')
                        axs[1, 1].set_title('Angolo di Beccheggio')
                        axs[1, 1].set_ylabel('Pitch [deg]')
                        axs[1, 1].grid(True, linestyle=':', alpha=0.7)
                        axs[1, 1].legend()

                        # Plot Yaw
                        axs[2, 1].plot(t_real, real_yaw_deg, 'c-', linewidth=2, label='Reale')
                        axs[2, 1].plot(t_dense, theor_yaw_deg, 'r--', linewidth=2, label='Teorico (Tangente)')
                        axs[2, 1].set_title('Angolo di Imbardata')
                        axs[2, 1].set_ylabel('Yaw [deg]')
                        axs[2, 1].grid(True, linestyle=':', alpha=0.7)
                        axs[2, 1].legend()

                        # Setup comune Asse X per l'ultima riga
                        axs[2, 0].set_xlabel('Tempo [s]')
                        axs[2, 1].set_xlabel('Tempo [s]')

                        plt.tight_layout()
                        plt.subplots_adjust(top=0.92)
                        
                        # Mostra ENTRAMBE le finestre bloccando il thread finchè non vengono chiuse manualmente
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