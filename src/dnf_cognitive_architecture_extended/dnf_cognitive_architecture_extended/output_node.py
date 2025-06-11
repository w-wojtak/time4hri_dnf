import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class OutputNode(Node):
    def __init__(self):
        super().__init__('output_node')

        # Subscription to the topic publishing threshold-crossing x values
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'threshold_crossings',  # Make sure this matches the topic name in the publisher node
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Timer that triggers periodically to print based on received values
        self.timer = self.create_timer(
            1.0, self.timer_callback)  # Runs every 1 second

        # Initialize received value to None
        self.received_value = None

    def listener_callback(self, msg):
        if msg.data:
            self.received_value = msg.data[0]
            self.get_logger().info(
                f"Received threshold crossing value: {self.received_value:.2f}")

        if self.received_value is not None:
            if -45 <= self.received_value <= -35:
                print("Message: Threshold crossed near the left input position.")
            elif -5 <= self.received_value <= 5:
                print("Message: Threshold crossed near the center input position.")
            elif 35 <= self.received_value <= 45:
                print("Message: Threshold crossed near the right input position.")
            else:
                print(
                    "Message: Threshold crossing detected outside expected input positions.")

        else:
            self.get_logger().info("Received message with empty data.")

    def timer_callback(self):
        pass


def main(args=None):
    rclpy.init(args=args)

    output_node = OutputNode()

    try:
        rclpy.spin(output_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up and shutdown the node
        output_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
