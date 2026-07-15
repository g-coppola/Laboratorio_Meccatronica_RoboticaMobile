import math
import heapq
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class VoxelGrid:
    def __init__(self, cell_size: float, inflate_cells: int):
        self.cell_size = cell_size
        self.inflate_cells = inflate_cells
        self.occupied = set()

    def world_to_grid(self, x: float, y: float, z: float):
        return (
            int(math.floor(x / self.cell_size)),
            int(math.floor(y / self.cell_size)),
            int(math.floor(z / self.cell_size)),
        )

    def grid_to_world(self, i: int, j: int, k: int):
        return (
            (i + 0.5) * self.cell_size,
            (j + 0.5) * self.cell_size,
            (k + 0.5) * self.cell_size,
        )

    def update_from_points(self, points_xyz: np.ndarray):
        raw_occupied = set()
        for x, y, z in points_xyz:
            raw_occupied.add(self.world_to_grid(x, y, z))

        if not raw_occupied:
            self.occupied = set()
            return

        inflated = set()
        r = self.inflate_cells
        offsets = [
            (di, dj, dk)
            for di in range(-r, r + 1)
            for dj in range(-r, r + 1)
            for dk in range(-r, r + 1)
        ]
        for (i, j, k) in raw_occupied:
            for (di, dj, dk) in offsets:
                inflated.add((i + di, j + dj, k + dk))

        self.occupied = inflated

    def is_occupied(self, i: int, j: int, k: int) -> bool:
        return (i, j, k) in self.occupied


NEIGHBORS_26 = [
    (di, dj, dk)
    for di in (-1, 0, 1)
    for dj in (-1, 0, 1)
    for dk in (-1, 0, 1)
    if not (di == 0 and dj == 0 and dk == 0)
]


def astar_3d(grid: VoxelGrid, start_cell, goal_cell, max_expansions: int = 200000):
    def heuristic(a, b):
        return math.sqrt(sum((a[d] - b[d]) ** 2 for d in range(3)))

    open_heap = []
    heapq.heappush(open_heap, (0.0, start_cell))

    g_score = {start_cell: 0.0}
    came_from = {}
    visited = set()

    expansions = 0
    weight = 1.5 

    while open_heap:
        expansions += 1
        if expansions > max_expansions:
            return None

        _, current = heapq.heappop(open_heap)

        if current in visited:
            continue
        visited.add(current)

        if current == goal_cell:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for (di, dj, dk) in NEIGHBORS_26:
            neighbor = (current[0] + di, current[1] + dj, current[2] + dk)

            if grid.is_occupied(*neighbor):
                continue

            step_cost = math.sqrt(di * di + dj * dj + dk * dk)
            tentative_g = g_score[current] + step_cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                
                f_score = tentative_g + (weight * heuristic(neighbor, goal_cell))
                heapq.heappush(open_heap, (f_score, neighbor))

    return None


def has_line_of_sight(grid, pA_world, pB_world):
    x1, y1, z1 = grid.world_to_grid(*pA_world)
    x2, y2, z2 = grid.world_to_grid(*pB_world)

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)

    xs = 1 if x2 > x1 else -1
    ys = 1 if y2 > y1 else -1
    zs = 1 if z2 > z1 else -1

    cx, cy, cz = x1, y1, z1

    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while cx != x2:
            if grid.is_occupied(cx, cy, cz): return False
            cx += xs
            if p1 >= 0:
                cy += ys
                p1 -= 2 * dx
            if p2 >= 0:
                cz += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while cy != y2:
            if grid.is_occupied(cx, cy, cz): return False
            cy += ys
            if p1 >= 0:
                cx += xs
                p1 -= 2 * dy
            if p2 >= 0:
                cz += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while cz != z2:
            if grid.is_occupied(cx, cy, cz): return False
            cz += zs
            if p1 >= 0:
                cy += ys
                p1 -= 2 * dz
            if p2 >= 0:
                cx += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx

    if grid.is_occupied(x2, y2, z2):
        return False

    return True


def simplify_path(grid, path_world):
    if len(path_world) <= 2:
        return path_world

    simplified = [path_world[0]]
    last_added = path_world[0]
    
    for i in range(1, len(path_world) - 1):
        current_point = path_world[i]
        next_point = path_world[i + 1]
        
        if not has_line_of_sight(grid, last_added, next_point):
            simplified.append(current_point)
            last_added = current_point
            
    simplified.append(path_world[-1])
    
    return simplified


def enforce_min_spacing(path_world, min_dist):
    if len(path_world) <= 2:
        return path_world

    filtered = [path_world[0]]
    for p in path_world[1:-1]:
        if math.dist(p, filtered[-1]) >= min_dist:
            filtered.append(p)
    filtered.append(path_world[-1])

    return filtered


def merge_collinear(path_world, angle_threshold_deg=6.0):
    if len(path_world) <= 2:
        return path_world

    cos_threshold = math.cos(math.radians(angle_threshold_deg))

    merged = [path_world[0]]
    for i in range(1, len(path_world) - 1):
        prev_p = merged[-1]
        curr_p = path_world[i]
        next_p = path_world[i + 1]

        v1 = tuple(a - b for a, b in zip(curr_p, prev_p))
        v2 = tuple(a - b for a, b in zip(next_p, curr_p))

        n1 = math.sqrt(sum(c * c for c in v1))
        n2 = math.sqrt(sum(c * c for c in v2))

        if n1 < 1e-6 or n2 < 1e-6:
            continue

        cos_angle = sum(a * b for a, b in zip(v1, v2)) / (n1 * n2)

        if cos_angle < cos_threshold:
            merged.append(curr_p)

    merged.append(path_world[-1])
    return merged


class AStarPlannerNode(Node):
    def __init__(self):
        super().__init__('astar_planner_node')

        self.declare_parameter('cell_size', 0.25)          
        self.declare_parameter('inflate_cells', 3)          
        self.declare_parameter('waypoint_min_dist', 0.5)    
        self.declare_parameter('goal_reach_threshold', 0.3) 

        cell_size = self.get_parameter('cell_size').value
        inflate_cells = self.get_parameter('inflate_cells').value
        self.waypoint_min_dist = self.get_parameter('waypoint_min_dist').value
        self.goal_reach_threshold = self.get_parameter('goal_reach_threshold').value

        self.grid = VoxelGrid(cell_size=cell_size, inflate_cells=inflate_cells)

        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_z = 0.0
        self.curr_orientation = None  # Salva il quaternione reale
        self.have_odom = False

        self.path_waypoints = []   
        self.active_goal = None    
        self.final_goal_orientation = None  
        
        self.is_planning = False

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            PointCloud2,
            '/octomap_occupied_space',
            self.octomap_callback,
            qos_sensor,
        )

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(PoseStamped, '/planner_goal', self.goal_callback, 10)

        self.path_pub = self.create_publisher(Path, '/planner/path', 10)
        self.create_timer(0.2, self.control_loop)

        self.get_logger().info('A* 3D planner avviato e in attesa di bersagli...')

    def octomap_callback(self, msg: PointCloud2):
        points = point_cloud2.read_points_numpy(
            msg, field_names=('x', 'y', 'z'), skip_nans=True
        )
        if points.size == 0: return
        self.grid.update_from_points(points)

    def odom_callback(self, msg: Odometry):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        self.curr_z = msg.pose.pose.position.z
        self.curr_orientation = msg.pose.pose.orientation  # Aggiorna quaternione
        self.have_odom = True

    def goal_callback(self, msg: PoseStamped):
        try:
            self._handle_goal(msg)
        except Exception as ex:
            self.is_planning = False
            self.get_logger().error(f'Errore nel gestire /planner_goal: {ex}', throttle_duration_sec=2.0)

    def _handle_goal(self, msg: PoseStamped):
        if not self.have_odom:
            return

        if self.is_planning:
            self.get_logger().warn('A* sta ancora calcolando! Goal ignorato.')
            return

        gx = msg.pose.position.x
        gy = msg.pose.position.y
        gz = msg.pose.position.z

        if not all(math.isfinite(v) for v in (gx, gy, gz)):
            self.get_logger().warn('Goal con coordinate non finite, scartato.', throttle_duration_sec=2.0)
            return

        self.is_planning = True
        threading.Thread(target=self._compute_path_thread, args=(msg,)).start()

    def _compute_path_thread(self, msg: PoseStamped):
        try:
            goal_x = msg.pose.position.x
            goal_y = msg.pose.position.y
            goal_z = msg.pose.position.z

            start_world = (self.curr_x, self.curr_y, self.curr_z)
            goal_world = (goal_x, goal_y, goal_z)

            self.active_goal = goal_world
            self.final_goal_orientation = msg.pose.orientation 

            is_same_altitude = abs(goal_z - self.curr_z) < 0.15
            
            if is_same_altitude and has_line_of_sight(self.grid, start_world, goal_world):
                path_world = [start_world, goal_world]
            else:
                start_cell = self.grid.world_to_grid(*start_world)
                goal_cell = self.grid.world_to_grid(*goal_world)

                path_cells = astar_3d(self.grid, start_cell, goal_cell)

                if path_cells is None:
                    self.get_logger().error('Nessun percorso trovato!')
                    self.path_waypoints = []
                    return

                path_world = [self.grid.grid_to_world(*c) for c in path_cells]
                path_world = simplify_path(self.grid, path_world)
                path_world = enforce_min_spacing(path_world, self.waypoint_min_dist)
                path_world = merge_collinear(path_world)

            self.path_waypoints = path_world
            self.publish_path()

        except Exception as ex:
            self.get_logger().error(f'Errore durante il calcolo del percorso: {ex}')
            self.path_waypoints = []
        finally:
            self.is_planning = False

    def control_loop(self):
        if not self.path_waypoints:
            return
        self.publish_path()

    def publish_path(self):
        if not self.path_waypoints:
            return

        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'

        for i, wp in enumerate(self.path_waypoints):
            pose = PoseStamped()
            pose.header.stamp = msg.header.stamp
            pose.header.frame_id = 'odom'
            pose.pose.position.x = float(wp[0])
            pose.pose.position.y = float(wp[1])
            pose.pose.position.z = float(wp[2])
            
            # --- ASSEGNAZIONE ORIENTAMENTO ---
            if i == 0 and self.curr_orientation is not None:
                # Il punto di partenza ha ESATTAMENTE l'orientamento reale attuale del drone
                pose.pose.orientation = self.curr_orientation
            elif i == len(self.path_waypoints) - 1 and self.final_goal_orientation is not None:
                # Il punto finale ha l'orientamento richiesto dal goal
                pose.pose.orientation = self.final_goal_orientation
            else:
                # I punti intermedi usano la tangente della traiettoria (Yaw)
                if i < len(self.path_waypoints) - 1:
                    next_wp = self.path_waypoints[i + 1]
                    dx = next_wp[0] - wp[0]
                    dy = next_wp[1] - wp[1]
                    yaw = math.atan2(dy, dx)
                else:
                    yaw = 0.0 # Fallback di sicurezza 
                    
                # Costruzione del Quaternione (rotazione solo su asse Z)
                pose.pose.orientation.z = math.sin(yaw / 2.0)
                pose.pose.orientation.w = math.cos(yaw / 2.0)
                
            msg.poses.append(pose)

        self.path_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AStarPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()