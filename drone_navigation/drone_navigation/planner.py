import math
import heapq
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
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


def simplify_path(path_world, min_dist: float):
    if not path_world:
        return []

    simplified = [path_world[0]]
    for p in path_world[1:]:
        last = simplified[-1]
        dist = math.sqrt(sum((p[d] - last[d]) ** 2 for d in range(3)))
        if dist >= min_dist:
            simplified.append(p)

    if simplified[-1] != path_world[-1]:
        simplified.append(path_world[-1])

    return simplified


class AStarPlannerNode(Node):
    def __init__(self):
        super().__init__('astar_planner_node')

        self.declare_parameter('cell_size', 0.25)          
        self.declare_parameter('inflate_cells', 3)          
        self.declare_parameter('waypoint_min_dist', 0.5)    
        self.declare_parameter('goal_reach_threshold', 0.35) 

        cell_size = self.get_parameter('cell_size').value
        inflate_cells = self.get_parameter('inflate_cells').value
        self.waypoint_min_dist = self.get_parameter('waypoint_min_dist').value
        self.goal_reach_threshold = self.get_parameter('goal_reach_threshold').value

        self.grid = VoxelGrid(cell_size=cell_size, inflate_cells=inflate_cells)

        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_z = 0.0
        self.have_odom = False

        self.path_waypoints = []   
        self.active_goal = None    
        
        # --- NUOVO: LOCK PER PROTEGGERE IL COMPUTER ---
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

        self.goal_pose_pub = self.create_publisher(Odometry, '/goal_pose', 10)
        self.create_timer(0.2, self.control_loop)

        self.get_logger().info('A* 3D planner avviato (Tolleranza Danni Attiva!). In attesa di bersagli...')

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
        self.have_odom = True

    def goal_callback(self, msg: PoseStamped):
        if not self.have_odom:
            return

        # --- PROTEZIONE THREAD (RACE CONDITION PREVENTION) ---
        if self.is_planning:
            self.get_logger().warn('A* sta ancora calcolando! Goal ignorato per salvare la CPU.')
            return

        self.is_planning = True
        threading.Thread(target=self._compute_path_thread, args=(msg,)).start()

    def _compute_path_thread(self, msg: PoseStamped):
        try:
            goal_x = msg.pose.position.x
            goal_y = msg.pose.position.y
            goal_z = msg.pose.position.z

            self.active_goal = (goal_x, goal_y, goal_z)

            start_cell = self.grid.world_to_grid(self.curr_x, self.curr_y, self.curr_z)
            goal_cell = self.grid.world_to_grid(goal_x, goal_y, goal_z)

            self.get_logger().info(f'Pianificazione verso ({goal_x:.2f},{goal_y:.2f},{goal_z:.2f})...')

            path_cells = astar_3d(self.grid, start_cell, goal_cell)

            if path_cells is None:
                self.get_logger().error('Nessun percorso trovato!')
                self.path_waypoints = []
                return

            path_world = [self.grid.grid_to_world(*c) for c in path_cells]
            self.path_waypoints = simplify_path(path_world, self.waypoint_min_dist)

            self.get_logger().info(f'Percorso aggiornato: {len(self.path_waypoints)} waypoint.')
            
        finally:
            # Rilascia il lock per consentire il calcolo di nuovi percorsi futuri
            self.is_planning = False

    def control_loop(self):
        if not self.path_waypoints:
            return

        target = self.path_waypoints[0]
        dist = math.sqrt(
            (target[0] - self.curr_x) ** 2
            + (target[1] - self.curr_y) ** 2
            + (target[2] - self.curr_z) ** 2
        )

        if dist <= self.goal_reach_threshold:
            self.path_waypoints.pop(0)
            if not self.path_waypoints:
                self.get_logger().info('Goal finale raggiunto.')
                return
            target = self.path_waypoints[0]

        self.publish_goal_pose(target)

    def publish_goal_pose(self, target):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'

        msg.pose.pose.position.x = target[0]
        msg.pose.pose.position.y = target[1]
        msg.pose.pose.position.z = target[2]

        dx = target[0] - self.curr_x
        dy = target[1] - self.curr_y
        yaw_desiderato = math.atan2(dy, dx)

        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = math.sin(yaw_desiderato / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw_desiderato / 2.0)

        self.goal_pose_pub.publish(msg)


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