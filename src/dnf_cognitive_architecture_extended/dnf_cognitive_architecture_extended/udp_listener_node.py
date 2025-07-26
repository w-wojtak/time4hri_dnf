import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import socket
import threading

UDP_IP = "0.0.0.0"
UDP_PORT = 5005


class UDPListenerNode(Node):
    def __init__(self):
        super().__init__('udp_listener_node')

        # Publisher for filtered messages
        self.filtered_publisher = self.create_publisher(
            String, 'voice_command_filtered', 10)

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

            if self.filter_message(message):
                self.filtered_publisher.publish(String(data=message))
                self.get_logger().info(
                    f"Published filtered message: '{message}'")
            else:
                self.get_logger().info("Message discarded.")

    def filter_message(self, message: str) -> bool:
        """
        Define your filtering logic here.
        Example: Only forward messages that contain the word 'start'
        """
        return 'start' in message.lower()


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
