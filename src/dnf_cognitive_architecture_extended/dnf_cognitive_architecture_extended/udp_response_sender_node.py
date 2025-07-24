# udp_response_sender_node.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import socket

# Change this to the IP of your Windows machine
UDP_IP = "10.205.240.222"   # example Windows IP (adjust!)
UDP_PORT = 5006            # different port than the listener, to avoid conflict


class UDPResponseSenderNode(Node):
    def __init__(self):
        super().__init__('udp_response_sender_node')
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.subscription = self.create_subscription(
            String,
            'response_command',
            self.listener_callback,
            10
        )
        self.get_logger().info(
            f"UDP Response Sender ready. Sending to {UDP_IP}:{UDP_PORT}")

    def listener_callback(self, msg):
        message = msg.data.strip()
        self.sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
        self.get_logger().info(f"Sent response: '{message}'")


def main(args=None):
    rclpy.init(args=args)
    node = UDPResponseSenderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
