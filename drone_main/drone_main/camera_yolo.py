import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from ultralytics import YOLO

class YoloDetectorNode(Node):

    def __init__(self):
        super().__init__('yolo_detector_node')
        
        # Inizializza il bridge per convertire immagini ROS in OpenCV
        self.cv_bridge = CvBridge()

        # Carica il modello YOLO (v8n è il più veloce, "nano")
        # Al primo avvio scaricherà il file yolov8n.pt automaticamente
        self.model = YOLO('yolov8n.pt') 

        # NOTA: Assicurati che il topic sia lo stesso del tuo URDF/Gazebo
        # Se nel plugin avevi <cameraName>cube/camera</cameraName> e 
        # <imageTopicName>image_raw</imageTopicName>, il topic sarà 'cube/camera/image_raw'
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw', 
            self.camera_callback,
            10)
        
        self.get_logger().info('YOLO Detection Node avviato!')

    def camera_callback(self, msg):
        # 1. Converti il messaggio ROS in immagine OpenCV
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 2. Esegui l'inferenza con YOLO
        # conf=0.5 significa che mostra solo oggetti con sicurezza superiore al 50%
        results = self.model(cv_image, conf=0.5, verbose=False)

        # 3. Visualizza i risultati
        # results[0].plot() restituisce l'immagine con i box e le label già disegnati
        annotated_frame = results[0].plot()

        # 4. Mostra l'immagine in una finestra
        cv2.imshow('YOLOv8 Detection - Robot Camera', annotated_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()