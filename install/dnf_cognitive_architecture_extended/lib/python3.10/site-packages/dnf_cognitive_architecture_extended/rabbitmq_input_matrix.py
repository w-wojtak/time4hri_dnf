#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float32MultiArray
import pika
import json
import threading


class RabbitMQInputMatrix(Node):
    def __init__(self):
        super().__init__("rabbitmq_input_matrix")

        # Publisher for the combined matrices
        self.input_pub = self.create_publisher(
            Float32MultiArray, "input_matrices_combined", 10
        )

        # Subscriber for threshold crossings
        self.threshold_sub = self.create_subscription(
            Float32MultiArray,
            "threshold_crossings",
            self.threshold_callback,
            10
        )

        # Initialize spatial parameters
        self.x_lim = 80
        self.t_lim = 15
        self.dx = 0.2
        self.dt = 0.1
        self.x = np.arange(-self.x_lim, self.x_lim + self.dx, self.dx)
        self.t = np.arange(0, self.t_lim + self.dt, self.dt)

        # Fixed object positions and amplitude
        self.object_positions = [-40, 0, 40]  # Left, Center, Right
        self.gaussian_amplitude = 5.0
        self.gaussian_width = 2.0

        # Initialize matrices
        self.input_matrix_1 = np.zeros((len(self.t), len(self.x)))  # Agent 1
        self.input_matrix_2 = np.zeros((len(self.t), len(self.x)))  # Agent 2
        self.input_matrix_3 = np.zeros((len(self.t), len(self.x)))  # Threshold

        # Timer for publishing matrices
        self.current_time_index = 0
        self.timer = self.create_timer(1, self.publish_slices)

        # RabbitMQ setup
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters('localhost')
        )
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='hand_data')

        # Start RabbitMQ consumer thread
        self.thread = threading.Thread(target=self.consume)
        self.thread.start()

        self.get_logger().info("RabbitMQ Input Matrix node initialized")

    def gaussian(self, x, center=0):
        """Generate Gaussian function with fixed amplitude and width"""
        return self.gaussian_amplitude * np.exp(
            -((x - center) ** 2) / (2 * (self.gaussian_width ** 2))
        )

    def update_matrices_from_vision(self, message):
        """Update matrices 1 and 2 based on RabbitMQ vision message"""
        try:
            # Initialize zero matrices for this time step
            self.input_matrix_1[self.current_time_index] = np.zeros(
                len(self.x))
            self.input_matrix_2[self.current_time_index] = np.zeros(
                len(self.x))

            # Update Agent 1 matrix (matrix 1)
            agent1_obj = message.get('agent1_grasped_object')
            if agent1_obj and message['agent1_state'] == 'GRASPING':
                position = agent1_obj['position']
                self.input_matrix_1[self.current_time_index] = self.gaussian(
                    self.x, center=position)

            # Update Agent 2 matrix (matrix 2)
            agent2_obj = message.get('agent2_grasped_object')
            if agent2_obj and message['agent2_state'] == 'GRASPING':
                position = agent2_obj['position']
                self.input_matrix_2[self.current_time_index] = self.gaussian(
                    self.x, center=position)

            self.get_logger().debug(
                f"Updated matrices for time {self.t[self.current_time_index]:.2f} - "
                f"Agent1: {message['agent1_state']}, Agent2: {message['agent2_state']}"
            )

        except Exception as e:
            self.get_logger().error(f"Error updating matrices: {str(e)}")

    def threshold_callback(self, msg):
        """Handle threshold crossing messages"""
        self.received_threshold_value = msg.data[0]
        self.get_logger().info(
            f"Received threshold crossing value: {self.received_threshold_value:.2f}"
        )

        # Determine the input position based on the received threshold crossing value
        if -45 <= self.received_threshold_value <= -35:
            input_position = -40  # Left input position
        elif -5 <= self.received_threshold_value <= 5:
            input_position = 0    # Center input position
        elif 35 <= self.received_threshold_value <= 45:
            input_position = 40   # Right input position
        else:
            self.get_logger().warning(
                "Threshold crossing value is outside expected ranges.")
            return

        # Define the delay in time steps
        delay = 5

        # Calculate t_start and t_stop for the updated Gaussian input
        t_start = self.t[self.current_time_index] + delay * self.dt
        t_stop = t_start + (self.dt * 10)  # Keep the duration of input fixed

        # Update matrix 3 based on threshold crossings
        for i in range(len(self.t)):
            if t_start <= self.t[i] <= t_stop:
                self.input_matrix_3[i] = self.gaussian(
                    self.x,
                    center=input_position
                )
            else:
                self.input_matrix_3[i] = np.zeros(len(self.x))

    def publish_slices(self):
        """Publish the current time slice of all matrices"""
        if self.current_time_index < len(self.t):
            # Combine the matrices into a single array
            combined_input = [
                self.input_matrix_1[self.current_time_index].tolist(),
                self.input_matrix_2[self.current_time_index].tolist(),
                self.input_matrix_3[self.current_time_index].tolist()
            ]

            # Create and publish the message
            msg = Float32MultiArray()
            msg.data = [item for sublist in combined_input for item in sublist]
            self.input_pub.publish(msg)

            # Log publication
            self.get_logger().info(
                f"Published t={self.t[self.current_time_index]:.2f}, "
                f"Max values - Matrix 1 (Agent 1): {self.input_matrix_1[self.current_time_index].max():.2f}, "
                f"Matrix 2 (Agent 2): {self.input_matrix_2[self.current_time_index].max():.2f}, "
                f"Matrix 3 (Threshold): {self.input_matrix_3[self.current_time_index].max():.2f}"
            )

            self.current_time_index += 1
        else:
            # Reset time index instead of stopping
            self.current_time_index = 0
            self.get_logger().info("Resetting time index to 0")

    def consume(self):
        """Consume messages from RabbitMQ"""
        try:
            for method_frame, properties, body in self.channel.consume('hand_data'):
                if not rclpy.ok():
                    break

                if body:
                    try:
                        message = json.loads(body)
                        self.update_matrices_from_vision(message)
                        self.channel.basic_ack(method_frame.delivery_tag)
                    except json.JSONDecodeError as e:
                        self.get_logger().error(
                            f"Failed to decode message: {str(e)}")
                    except Exception as e:
                        self.get_logger().error(
                            f"Error processing message: {str(e)}")
        except Exception as e:
            self.get_logger().error(f"Error in consume loop: {str(e)}")
        finally:
            if self.connection and not self.connection.is_closed:
                self.connection.close()

    def destroy_node(self):
        """Cleanup when node is destroyed"""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RabbitMQInputMatrix()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
