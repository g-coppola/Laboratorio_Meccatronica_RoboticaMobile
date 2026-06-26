import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
import math

class PathTracker(Node):
    def __init__(self):
        super().__init__('path_tracker')
        
        self.path_sub = self.create_subscription(Path, '/nav6d/planner/path', self.path_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.goal_pub = self.create_publisher(Odometry, '/goal_pose', 10)
        
        self.path = []
        self.current_wp_idx = 0
        
        # PARAMETRI AGGIORNATI PER VOLO PIU' MORBIDO
        self.lookahead_dist = 0.4  # Leggermente più stretto per seguire meglio le curve
        self.max_speed = 0.4       # Abbassato da 1.0 a 0.4 m/s per sicurezza in magazzino
        self.deceleration_dist = 1.5 # Inizia a frenare quando manca 1.5m al traguardo

        self.get_logger().info("Path Tracker 2.0 (Smooth) avviato!")

    def path_cb(self, msg):
        self.path = msg.poses
        self.current_wp_idx = 0
        if len(self.path) > 0:
            self.get_logger().info(f"Nuova rotta! Waypoint: {len(self.path)}. Volo in corso...")

    def odom_cb(self, msg):
        if not self.path or self.current_wp_idx >= len(self.path):
            return 

        curr_x = msg.pose.pose.position.x
        curr_y = msg.pose.pose.position.y
        curr_z = msg.pose.pose.position.z

        # Destinazione FINALE (per calcolare la frenata)
        final_pose = self.path[-1].pose.position
        dist_to_goal = math.sqrt((final_pose.x - curr_x)**2 + 
                                 (final_pose.y - curr_y)**2 + 
                                 (final_pose.z - curr_z)**2)

        # Waypoint LOCALE (la carota)
        target_pose = self.path[self.current_wp_idx].pose
        dist_to_wp = math.sqrt((target_pose.position.x - curr_x)**2 + 
                               (target_pose.position.y - curr_y)**2 + 
                               (target_pose.position.z - curr_z)**2)

        # Avanzamento waypoint
        if dist_to_wp < self.lookahead_dist:
            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.path):
                self.get_logger().info("Destinazione raggiunta! Hovering.")
                self.publish_setpoint(final_pose.x, final_pose.y, final_pose.z, target_pose.orientation, 0.0, 0.0, 0.0)
                self.path = [] # Svuota la traiettoria
                return

        # Calcolo velocità dinamica (Frenata intelligente)
        current_speed = self.max_speed
        if dist_to_goal < self.deceleration_dist:
            # Scala linearmente la velocità man mano che si avvicina
            current_speed = max(0.1, self.max_speed * (dist_to_goal / self.deceleration_dist))

        # Calcolo del vettore verso il prossimo WP
        next_pose = self.path[self.current_wp_idx].pose
        dir_x = next_pose.position.x - curr_x
        dir_y = next_pose.position.y - curr_y
        dir_z = next_pose.position.z - curr_z
        norm = math.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
        
        vx, vy, vz = 0.0, 0.0, 0.0
        if norm > 0.01:
            vx = (dir_x / norm) * current_speed
            vy = (dir_y / norm) * current_speed
            vz = (dir_z / norm) * current_speed

        self.publish_setpoint(next_pose.position.x, next_pose.position.y, next_pose.position.z, next_pose.orientation, vx, vy, vz)

    def publish_setpoint(self, x, y, z, q, vx, vy, vz):
        goal_msg = Odometry()
        goal_msg.header.frame_id = "map"
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = float(z)
        goal_msg.pose.pose.orientation = q
        goal_msg.twist.twist.linear.x = float(vx)
        goal_msg.twist.twist.linear.y = float(vy)
        goal_msg.twist.twist.linear.z = float(vz)
        self.goal_pub.publish(goal_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PathTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()