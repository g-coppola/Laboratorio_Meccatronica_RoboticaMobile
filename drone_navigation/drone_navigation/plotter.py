#!/usr/bin/env encoding=utf-8
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessario per alcune versioni di matplotlib
import math
import threading

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
        
        self.get_logger().info("Nodo Waypoint Plotter 3D Avviato! In attesa di percorsi...")

    def path_callback(self, msg):
        if not msg.poses:
            return

        # 1. Estrai le coordinate XYZ da tutti i pose del percorso
        extracted_points = []
        for pose_stamped in msg.poses:
            extracted_points.append((
                pose_stamped.pose.position.x,
                pose_stamped.pose.position.y,
                pose_stamped.pose.position.z
            ))

        with self.lock:
            # 2. Controllo per evitare lo sfarfallio (Flickering)
            # Se abbiamo già un percorso, verifichiamo se l'obiettivo finale (l'ultimo punto) è cambiato
            if self.current_path:
                last_wp_new = extracted_points[-1]
                last_wp_curr = self.current_path[-1]
                
                # Calcola la distanza tra il vecchio goal finale e il nuovo goal finale
                dist_goal = math.sqrt(
                    (last_wp_new[0] - last_wp_curr[0])**2 +
                    (last_wp_new[1] - last_wp_curr[1])**2 +
                    (last_wp_new[2] - last_wp_curr[2])**2
                )
                
                # Se il traguardo è cambiato di meno di 10 centimetri, consideriamo il percorso 
                # come un semplice aggiornamento di avanzamento del drone e non rifacciamo il plot.
                if dist_goal < 0.1:
                    return
            
            # Se superiamo il controllo o se è il primo percorso in assoluto, aggiorna
            self.current_path = extracted_points
            self.new_path_received = True
            self.get_logger().info(f"Rilevato nuovo piano di volo globale. Waypoint totali: {len(extracted_points)}")


def main(args=None):
    rclpy.init(args=args)
    node = WaypointPlotterNode()

    # Avviamo lo spin di ROS 2 in un thread separato in background
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # Attiviamo la modalità interattiva di Matplotlib sul thread principale
    plt.ion()
    fig = None

    try:
        while rclpy.ok():
            # Controlla in modo non bloccante se il callback ha registrato una nuova traiettoria
            if node.new_path_received:
                with node.lock:
                    node.new_path_received = False
                    waypoints = list(node.current_path)

                if len(waypoints) >= 2:
                    # Se c'è una finestra aperta dal precedente goal, la chiudiamo
                    if fig is not None and plt.fignum_exists(fig.number):
                        plt.close(fig)
                    
                    # Generazione del nuovo spazio grafico 3D
                    fig = plt.figure(figsize=(10, 7))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    xs = [wp[0] for wp in waypoints]
                    ys = [wp[1] for wp in waypoints]
                    zs = [wp[2] for wp in waypoints]
                    
                    # 3. Calcolo geometrico dello Yaw per ogni segmento
                    us, vs, ws = [], [], []
                    for i in range(len(waypoints)):
                        if i < len(waypoints) - 1:
                            dx = xs[i+1] - xs[i]
                            dy = ys[i+1] - ys[i]
                            yaw = math.atan2(dy, dx)
                        else:
                            # Per l'ultimo punto, mantiene lo yaw del segmento precedente
                            yaw = math.atan2(ys[-1] - ys[-2], xs[-1] - xs[-2]) if len(waypoints) > 1 else 0.0
                        
                        us.append(math.cos(yaw))
                        vs.append(math.sin(yaw))
                        ws.append(0.0) # Vettore sul piano orizzontale XY
                    
                    # 4. Tracciamento della spezzata dei waypoint
                    ax.plot(xs, ys, zs, label='Traiettoria Calcolata', color='#1f77b4', marker='o', linewidth=2)
                    
                    # Evidenzia in modo specifico la Partenza e il Goal Finale
                    ax.scatter(xs[0], ys[0], zs[0], color='green', s=100, label='Partenza Drone')
                    ax.scatter(xs[-1], ys[-1], zs[-1], color='gold', s=150, marker='*', label='Goal Target')
                    
                    # 5. Rappresentazione dell'orientamento tramite Quiver (Frecce)
                    ax.quiver(xs, ys, zs, us, vs, ws, length=0.3, color='red', normalize=True, label='Angolo di Yaw')
                    
                    # Estetica del grafico
                    ax.set_xlabel('Asse X (metri)')
                    ax.set_ylabel('Asse Y (metri)')
                    ax.set_zlabel('Altitudine Z (metri)')
                    ax.set_title('Mappa dei Waypoint 3D con Vettore di Orientamento')
                    ax.legend(loc='upper left')
                    ax.grid(True)
                    
                    # Forza il rendering della nuova finestra senza bloccare l'esecuzione
                    plt.show()
            
            # Gestisce gli eventi della GUI di Matplotlib (rotazione della telecamera con il mouse)
            plt.pause(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        plt.close('all')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()