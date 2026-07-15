import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path


def build_trajectory(points_xyz, quats, cruise_speed, min_segment_time):
    pts = np.asarray(points_xyz, dtype=float)
    qts = np.asarray(quats, dtype=float)

    # Rimuove punti consecutivi troppo vicini per evitare tempi di segmento ~0
    cleaned_pts = [pts[0]]
    cleaned_qts = [qts[0]]
    for p, q in zip(pts[1:], qts[1:]):
        if np.linalg.norm(p - cleaned_pts[-1]) > 1e-3:
            cleaned_pts.append(p)
            cleaned_qts.append(q)
            
    pts = np.array(cleaned_pts)
    qts = np.array(cleaned_qts)

    if len(pts) < 2:
        return None

    seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    seg_times = np.maximum(seg_lengths / cruise_speed, min_segment_time)
    t = np.concatenate(([0.0], np.cumsum(seg_times)))

    spline_x = CubicSpline(t, pts[:, 0], bc_type='clamped')
    spline_y = CubicSpline(t, pts[:, 1], bc_type='clamped')
    spline_z = CubicSpline(t, pts[:, 2], bc_type='clamped')

    # Normalizza i quaternioni per evitare errori con lo Slerp
    norms = np.linalg.norm(qts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Evita divisioni per zero
    qts_norm = qts / norms

    # Crea l'interpolatore sferico (Slerp) per le rotazioni
    rotations = Rotation.from_quat(qts_norm)
    slerp = Slerp(t, rotations)

    return spline_x, spline_y, spline_z, slerp, float(t[-1])


class TrajectoryGenerator(Node):
    def __init__(self):
        super().__init__('trajectory_generator')

        self.declare_parameter('cruise_speed', 0.3)
        self.declare_parameter('update_rate', 30.0)
        self.declare_parameter('min_segment_time', 0.4)

        self.cruise_speed = self.get_parameter('cruise_speed').value
        self.update_rate = self.get_parameter('update_rate').value
        self.min_segment_time = self.get_parameter('min_segment_time').value

        # --- Stato odometria e Tempo di Simulazione ---
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_z = 0.0
        self.curr_orientation = None
        self.have_odom = False
        
        # Variabili chiave per usare il tempo di simulazione (Gazebo) invece del Wall Time
        self.current_sim_time = 0.0

        # --- Stato traiettoria corrente ---
        self.spline_x = None
        self.spline_y = None
        self.spline_z = None
        self.slerp = None
        self.total_time = 0.0
        self.traj_start_time = 0.0
        self.has_trajectory = False

        self._last_raw_waypoints = None

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Path, '/planner/path', self.path_callback, 10)
        self.goal_pose_pub = self.create_publisher(Odometry, '/goal_pose', 10)
        
        self.create_timer(1.0 / self.update_rate, self.timer_callback)

        self.get_logger().info(
            f'Trajectory generator avviato (cruise_speed={self.cruise_speed:.2f} m/s, '
            f'{self.update_rate:.0f} Hz).'
        )

    def odom_callback(self, msg: Odometry):
        # Estraiamo il tempo di simulazione direttamente dal messaggio!
        self.current_sim_time = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
        
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        self.curr_z = msg.pose.pose.position.z
        self.curr_orientation = msg.pose.pose.orientation
        self.have_odom = True

    def path_callback(self, msg: Path):
        if not self.have_odom or len(msg.poses) < 1:
            return

        new_waypoints = [
            (p.pose.position.x, p.pose.position.y, p.pose.position.z)
            for p in msg.poses
        ]
        
        # Estraiamo anche i quaternioni assegnati dal planner
        new_quats = [
            (p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w)
            for p in msg.poses
        ]

        if self._same_path(new_waypoints):
            return

        # Recupera il quaternione di partenza attuale del drone in modo sicuro
        curr_quat = (
            self.curr_orientation.x,
            self.curr_orientation.y,
            self.curr_orientation.z,
            self.curr_orientation.w
        ) if self.curr_orientation else (0.0, 0.0, 0.0, 1.0)

        points = [(self.curr_x, self.curr_y, self.curr_z)] + new_waypoints
        quats = [curr_quat] + new_quats
        
        result = build_trajectory(points, quats, self.cruise_speed, self.min_segment_time)

        if result is None:
            self.get_logger().warn('Path troppo corto o degenere, traiettoria ignorata.')
            return

        self.spline_x, self.spline_y, self.spline_z, self.slerp, self.total_time = result
        
        # Qui registriamo l'inizio della traiettoria basandoci sull'orologio di simulazione
        self.traj_start_time = self.current_sim_time
        
        self.has_trajectory = True
        self._last_raw_waypoints = new_waypoints

        self.get_logger().info(
            f'Nuova traiettoria 6-DOF: {len(points)} punti, durata {self.total_time:.1f} s (Sim Time).'
        )

    def _same_path(self, new_waypoints, tol=1e-3):
        old = self._last_raw_waypoints
        if old is None or len(old) != len(new_waypoints):
            return False
        return all(math.dist(a, b) <= tol for a, b in zip(old, new_waypoints))

    def timer_callback(self):
        if not self.has_trajectory or self.current_sim_time == 0.0:
            return

        # Il tempo elapsed è calcolato basandosi ESCLUSIVAMENTE sull'orologio di Gazebo
        elapsed = self.current_sim_time - self.traj_start_time
        t = min(max(elapsed, 0.0), self.total_time)

        # Posizione e velocità
        x = float(self.spline_x(t))
        y = float(self.spline_y(t))
        z = float(self.spline_z(t))
        vx = float(self.spline_x(t, 1))
        vy = float(self.spline_y(t, 1))
        vz = float(self.spline_z(t, 1))

        # Rotazione interpolata morbidamente tramite Slerp!
        curr_rot = self.slerp(t)
        qx, qy, qz, qw = curr_rot.as_quat()

        msg = Odometry()
        # Per mantenere totale consistenza, usiamo il tempo dell'odometria anche nell'header
        msg.header.stamp.sec = int(self.current_sim_time)
        msg.header.stamp.nanosec = int((self.current_sim_time - int(self.current_sim_time)) * 1e9)
        msg.header.frame_id = 'odom'

        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = z

        msg.pose.pose.orientation.x = float(qx)
        msg.pose.pose.orientation.y = float(qy)
        msg.pose.pose.orientation.z = float(qz)
        msg.pose.pose.orientation.w = float(qw)

        msg.twist.twist.linear.x = vx
        msg.twist.twist.linear.y = vy
        msg.twist.twist.linear.z = vz

        self.goal_pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()