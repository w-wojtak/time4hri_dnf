import rclpy
from rclpy.node import Node
import pika
import json
import threading
from enum import Enum, auto


class GraspState(Enum):
    SCANNING = auto()
    APPROACHING = auto()
    GRASPING = auto()
    HOLDING = auto()
    RELEASING = auto()


class RabbitSubscriberNode(Node):
    def __init__(self):
        super().__init__('rabbit_subscriber_node')

        # RabbitMQ setup
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters('localhost'))
        self.channel = self.connection.channel()
        # Changed queue name to match publisher
        self.channel.queue_declare(queue='hand_data')
        self.thread = threading.Thread(target=self.consume)

        # State tracking
        self.current_state = None
        self.selected_object = None
        self.detected_objects = []
        self.hand_position = {'x': 0.0, 'y': 0.0}
        self.grasp_progress = 0.0

        # Create publishers for different types of data
        # (You can use these to republish data to other ROS 2 nodes)
        # self.state_publisher = self.create_publisher(String, 'grasp_state', 10)
        # self.object_publisher = self.create_publisher(ObjectMsg, 'detected_objects', 10)

        # Start consuming messages
        self.thread.start()

    def handle_message(self, message):
        """Process the received message and update internal state"""
        try:
            # Update state
            self.current_state = message.get('state', 'UNKNOWN')
            self.detected_objects = message.get('detected_objects', [])
            self.hand_position = message.get(
                'hand_position', {'x': 0.0, 'y': 0.0})

            # Log different information based on the state
            if self.current_state == 'SCANNING':
                self.log_scanning_state(message)
            elif self.current_state == 'APPROACHING':
                self.log_approaching_state(message)
            elif self.current_state == 'GRASPING':
                self.log_grasping_state(message)
            elif self.current_state == 'HOLDING':
                self.log_holding_state(message)
            elif self.current_state == 'RELEASING':
                self.log_releasing_state(message)

        except Exception as e:
            self.get_logger().error(f"Error processing message: {str(e)}")

    def log_scanning_state(self, message):
        """Log information during SCANNING state"""
        num_objects = len(self.detected_objects)
        objects_str = ", ".join(obj['name'] for obj in self.detected_objects)
        self.get_logger().info(
            f"SCANNING: Detected {num_objects} objects: {objects_str}"
        )

    def log_approaching_state(self, message):
        """Log information during APPROACHING state"""
        if 'selected_object' in message:
            obj = message['selected_object']
            hand_pos = message['hand_position']
            obj_pos = obj['position']
            self.get_logger().info(
                f"APPROACHING: Moving to {obj['name']} at ({obj_pos['x']:.2f}, {obj_pos['y']:.2f}). "
                f"Hand at ({hand_pos['x']:.2f}, {hand_pos['y']:.2f})"
            )

    def log_grasping_state(self, message):
        """Log information during GRASPING state"""
        if 'selected_object' in message and 'grasp_progress' in message:
            obj = message['selected_object']
            progress = message['grasp_progress']
            self.get_logger().info(
                f"GRASPING: {obj['name']} - Progress: {progress:.2%}"
            )

    def log_holding_state(self, message):
        """Log information during HOLDING state"""
        if 'selected_object' in message:
            obj = message['selected_object']
            self.get_logger().info(
                f"HOLDING: Successfully grasped {obj['name']}"
            )

    def log_releasing_state(self, message):
        """Log information during RELEASING state"""
        if 'selected_object' in message and 'grasp_progress' in message:
            obj = message['selected_object']
            # Convert to release progress
            progress = 1.0 - message['grasp_progress']
            self.get_logger().info(
                f"RELEASING: {obj['name']} - Progress: {progress:.2%}"
            )

    def consume(self):
        """Consume messages from RabbitMQ"""
        for method_frame, properties, body in self.channel.consume('hand_data', inactivity_timeout=1):
            if body:
                try:
                    message = json.loads(body)
                    self.handle_message(message)
                    self.channel.basic_ack(method_frame.delivery_tag)
                except json.JSONDecodeError as e:
                    self.get_logger().error(
                        f"Failed to decode message: {str(e)}")
                except Exception as e:
                    self.get_logger().error(
                        f"Error processing message: {str(e)}")

            if not rclpy.ok():
                break

    def destroy_node(self):
        """Cleanup when node is destroyed"""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RabbitSubscriberNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
