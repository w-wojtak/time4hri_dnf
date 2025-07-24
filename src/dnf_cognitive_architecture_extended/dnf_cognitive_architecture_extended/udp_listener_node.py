import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import socket
import threading

UDP_IP = "0.0.0.0"     # Listen on all interfaces
UDP_PORT = 5005        # Must match sender


class UDPListenerNode(Node):
    def __init__(self):
        super().__init__('udp_listener_node')
        self.publisher_ = self.create_publisher(String, 'voice_command', 10)
        self.get_logger().info(
            f'Listening for UDP messages on port {UDP_PORT}')
        self.thread = threading.Thread(target=self.listen_loop, daemon=True)
        self.thread.start()

    def listen_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))
        while rclpy.ok():
            data, _ = sock.recvfrom(1024)
            message = data.decode().strip()
            self.get_logger().info(f"Received: '{message}'")
            self.publisher_.publish(String(data=message))


def main(args=None):
    rclpy.init(args=args)
    node = UDPListenerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
