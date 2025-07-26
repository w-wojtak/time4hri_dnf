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
        self.filtered_publisher = self.create_publisher(
            String, 'voice_command_filtered', 10)

        # Track "start" and "finished" to publish only once
        self.sent_flags = {
            "start": False,
            "finished": False
        }

        self.get_logger().info(
            f'Listening for UDP messages on port {UDP_PORT}')
        self.thread = threading.Thread(target=self.listen_loop, daemon=True)
        self.thread.start()

    def filter_message(self, message: str) -> str:
        """
        Returns message if it should be forwarded, otherwise None.
        """
        message = message.lower().strip()
        if message in self.sent_flags:
            if not self.sent_flags[message]:
                self.sent_flags[message] = True
                return message
            else:
                return None  # Already sent once, skip
        return message  # Other commands always forwarded

    def listen_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))
        while rclpy.ok():
            data, _ = sock.recvfrom(1024)
            message = data.decode().strip()
            self.get_logger().info(f"Received: '{message}'")

            filtered = self.filter_message(message)
            if filtered:
                self.filtered_publisher.publish(String(data=filtered))
                self.get_logger().info(
                    f"Published filtered message: '{filtered}'")
            else:
                self.get_logger().info("Filtered out (duplicate or unwanted).")


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
