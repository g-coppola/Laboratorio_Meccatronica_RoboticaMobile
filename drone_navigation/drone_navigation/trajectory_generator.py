#!/usr/bin/env python3
"""
Trajectory Generator Node
==========================
Si inserisce TRA il planner A* (planner.py) e il controllore di volo
(full_control.py), senza che quest'ultimo debba cambiare.

Perche' serve
-------------
planner.py oggi pubblica un SOLO waypoint alla volta su /goal_pose (lo
"pop-a" dalla lista quando il drone gli si avvicina) e non riempie mai il
campo twist del messaggio. Risultato: full_control.py insegue punti fermi
con il solo PID reattivo (target_vx/vy restano sempre a zero) -> il drone
frena ad ogni waypoint e riparte, cioe' la navigazione "punto-punto" che
si vuole eliminare.

Questo nodo prende la LISTA di waypoint pubblicata da planner.py su
/planner/path (gia' esistente, nessuna modifica al planner A* necessaria)
e ci fa passare tre spline cubiche x(t), y(t), z(t):
  - continue in posizione, velocita' E accelerazione (C2)
  - velocita' nulla al primo e all'ultimo punto ("clamped": volo
    hover-to-hover)
Poi, ad ogni tick di un timer, campiona posizione E velocita' all'istante
corrente e le pubblica su /goal_pose nello stesso formato (nav_msgs/Odometry)
che full_control.py gia' consuma: il campo twist, che finora arrivava
sempre a zero, ora porta la velocita' vera della traiettoria e va ad
alimentare direttamente il feedforward (k_ff_xy) gia' presente nel tuo
controllore.

Dipendenze aggiuntive: numpy, scipy
    pip install scipy --break-system-packages     # o: sudo apt install python3-scipy
"""

import math

import numpy as np
from scipy.interpolate import CubicSpline

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path


def build_trajectory(points_xyz, cruise_speed, min_segment_time):
    """
    Costruisce tre spline cubiche "clamped" (velocita' nulla ai due estremi)
    attraverso i punti dati, con parametrizzazione temporale basata sulla
    lunghezza d'arco (tempo di ogni segmento = distanza / cruise_speed).

    points_xyz: lista di (x, y, z). Il primo elemento deve essere il punto
                di partenza "vero" (tipicamente la posizione ODOM attuale
                del drone, non il primo waypoint discretizzato del planner:
                cosi' non c'e' nessun salto iniziale).
    cruise_speed: velocita' media desiderata lungo i segmenti [m/s].
    min_segment_time: tempo minimo per segmento [s], evita segmenti
                       "sparati" quando due punti sono molto ravvicinati.

    Ritorna (spline_x, spline_y, spline_z, durata_totale) oppure None se i
    punti non bastano a definire una traiettoria (es. < 2 punti distinti).
    """
    pts = np.asarray(points_xyz, dtype=float)

    # Rimuove punti consecutivi troppo vicini: altrimenti creano segmenti a
    # lunghezza ~0 che non aggiungono nulla alla forma della traiettoria.
    cleaned = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - cleaned[-1]) > 1e-3:
            cleaned.append(p)
    pts = np.array(cleaned)

    if len(pts) < 2:
        return None

    seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    seg_times = np.maximum(seg_lengths / cruise_speed, min_segment_time)
    t = np.concatenate(([0.0], np.cumsum(seg_times)))

    spline_x = CubicSpline(t, pts[:, 0], bc_type='clamped')
    spline_y = CubicSpline(t, pts[:, 1], bc_type='clamped')
    spline_z = CubicSpline(t, pts[:, 2], bc_type='clamped')

    return spline_x, spline_y, spline_z, float(t[-1])


class TrajectoryGenerator(Node):
    def __init__(self):
        super().__init__('trajectory_generator')

        self.declare_parameter('cruise_speed', 0.8)       # m/s, velocita' media lungo i segmenti
        self.declare_parameter('update_rate', 30.0)       # Hz, frequenza di pubblicazione su /goal_pose
        self.declare_parameter('min_segment_time', 0.4)   # s, tempo minimo per segmento

        self.cruise_speed = self.get_parameter('cruise_speed').value
        self.update_rate = self.get_parameter('update_rate').value
        self.min_segment_time = self.get_parameter('min_segment_time').value

        # --- Stato odometria ---
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_z = 0.0
        self.have_odom = False

        # --- Stato traiettoria corrente ---
        self.spline_x = None
        self.spline_y = None
        self.spline_z = None
        self.total_time = 0.0
        self.traj_start_time = None
        self.final_orientation = None
        self.has_trajectory = False

        # Ultimo set di waypoint "grezzi" ricevuti da /planner/path: serve
        # per ignorare le ripubblicazioni del path quando NON e' cambiato
        # (planner.py lo ripubblica ogni 0.2s per RViz anche se e' identico:
        # senza questo controllo la traiettoria verrebbe rigenerata/azzerata
        # in continuazione e il drone non avanzerebbe mai davvero).
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
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        self.curr_z = msg.pose.pose.position.z
        self.have_odom = True

    def path_callback(self, msg: Path):
        if not self.have_odom or len(msg.poses) < 1:
            return

        new_waypoints = [
            (p.pose.position.x, p.pose.position.y, p.pose.position.z)
            for p in msg.poses
        ]

        if self._same_path(new_waypoints):
            return

        # Il primo punto della spline e' la posizione ODOM continua attuale,
        # non il primo waypoint (discretizzato sulla griglia voxel) del
        # planner: evita un piccolo salto/offset iniziale.
        points = [(self.curr_x, self.curr_y, self.curr_z)] + new_waypoints
        result = build_trajectory(points, self.cruise_speed, self.min_segment_time)

        if result is None:
            self.get_logger().warn('Path troppo corto o degenere, traiettoria ignorata.')
            return

        self.spline_x, self.spline_y, self.spline_z, self.total_time = result
        self.traj_start_time = self.get_clock().now()
        self.final_orientation = msg.poses[-1].pose.orientation
        self.has_trajectory = True
        self._last_raw_waypoints = new_waypoints

        self.get_logger().info(
            f'Nuova traiettoria: {len(points)} punti, durata {self.total_time:.1f} s.'
        )

    def _same_path(self, new_waypoints, tol=1e-3):
        old = self._last_raw_waypoints
        if old is None or len(old) != len(new_waypoints):
            return False
        return all(math.dist(a, b) <= tol for a, b in zip(old, new_waypoints))

    def timer_callback(self):
        if not self.has_trajectory:
            return

        elapsed = (self.get_clock().now() - self.traj_start_time).nanoseconds * 1e-9
        t = min(max(elapsed, 0.0), self.total_time)

        x = float(self.spline_x(t))
        y = float(self.spline_y(t))
        z = float(self.spline_z(t))
        vx = float(self.spline_x(t, 1))
        vy = float(self.spline_y(t, 1))
        vz = float(self.spline_z(t, 1))

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'

        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = z

        # Qui la differenza chiave rispetto a prima: il twist NON e' piu'
        # sempre zero, e va ad attivare il feedforward gia' scritto in
        # full_control.py (k_ff_xy, ora anche k_ff_z se applichi la
        # modifica opzionale a full_control.py).
        msg.twist.twist.linear.x = vx
        msg.twist.twist.linear.y = vy
        msg.twist.twist.linear.z = vz

        speed_xy = math.hypot(vx, vy)
        at_goal = elapsed >= self.total_time

        if at_goal and self.final_orientation is not None:
            msg.pose.pose.orientation = self.final_orientation
        elif speed_xy > 0.05:
            yaw = math.atan2(vy, vx)
            msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
            msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        elif self.final_orientation is not None:
            msg.pose.pose.orientation = self.final_orientation
        else:
            msg.pose.pose.orientation.w = 1.0

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