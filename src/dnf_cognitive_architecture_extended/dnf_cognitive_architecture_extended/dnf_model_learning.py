#!/usr/bin/env python3

import threading
from datetime import datetime
import os
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
from std_msgs.msg import Float32MultiArray
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
import rclpy


class DNFModel(Node):
    def __init__(self):
        super().__init__("dnf_model_learning")

        # Spatial and temporal parameters
        self.x_lim = 80
        self.t_lim = 15
        self.dx = 0.2
        self.dt = 0.1

        # Spatial grid
        self.x = np.arange(-self.x_lim, self.x_lim + self.dx, self.dx)
        self.t = np.arange(0, self.t_lim + self.dt, self.dt)

        self.u_sm_history = []  # List to store values at each time step
        self.u_sm_2_history = []  # List to store values at each time step
        self.u_d_history = []  # List to store values at each time step

        # Lock for threading
        self._lock = threading.Lock()

        # Create a subscriber to the 'input_matrices_combined' topic
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'input_matrices_combined',
            self.input_callback,
            10
        )

        # Initialize figure and axis for plotting
        # self.fig, self.ax = plt.subplots()

        self.fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        self.ax1, self.ax2 = axes.flatten()

        # Plot for u_sm_1 on ax1
        self.line_u_sm_1, = self.ax1.plot(
            self.x, np.zeros_like(self.x), label="u_sm_1")
        self.ax1.set_xlim(-self.x_lim, self.x_lim)
        self.ax1.set_ylim(-2, 10)  # Adjust based on expected amplitude
        self.ax1.set_xlabel("x")
        self.ax1.set_ylabel("u_sm_1(x)")
        self.ax1.set_title("Sequence Memory Field 1 (Robot)")
        self.ax1.legend()

        # Plot for u_sm_2 on ax2
        self.line_u_sm_2, = self.ax2.plot(
            self.x, np.zeros_like(self.x), label="u_sm_2")
        self.ax2.set_xlim(-self.x_lim, self.x_lim)
        self.ax2.set_ylim(-2, 10)  # Adjust based on expected amplitude
        self.ax2.set_xlabel("x")
        self.ax2.set_ylabel("u_sm_2(x)")
        self.ax2.set_title("Sequence Memory Field 2 (Human)")
        self.ax2.legend()

        # self.line, = self.ax.plot(
        #     self.x, np.zeros_like(self.x), label="u_field")
        # self.line_input, = self.ax.plot(self.x, np.zeros_like(
        #     self.x), label="Input", linestyle="--", color="orange")

        # self.line_u_d, = self.ax.plot(
        #     self.x, np.zeros_like(self.x), label="duration")

        # self.ax.set_xlim(-self.x_lim, self.x_lim)
        # self.ax.set_ylim(-1, 5)  # Adjust based on expected input amplitude
        # self.ax.set_xlabel("x")
        # self.ax.set_ylabel("u_sm(x), I(x)")
        # self.ax.set_title("u_sm(x), I(x)")
        # self.ax.legend()

        # Variable to store the latest input slice
        self.latest_input_slice = np.zeros_like(self.x)
        self.latest_input_slice_2 = np.zeros_like(self.x)
        # Initialize time counter and step counter
        self.time_counter = 0.0
        self.current_step = 1  # Tracks the next full step to print

        # Parameters for sequence memory field u_sm

        self.h_0_sm = 0
        self.tau_h_sm = 20
        self.theta_sm = 1.5

        self.h_0_sm_2 = 0
        self.theta_sm_2 = 1.5

        self.kernel_pars_sm = (1, 0.7, 0.9)
        # Fourier transform of the kernel function
        self.w_hat_sm = np.fft.fft(self.kernel_osc(*self.kernel_pars_sm))

        # Default initialization
        self.u_sm = self.h_0_sm * np.ones(np.shape(self.x))
        self.h_u_sm = self.h_0_sm * np.ones(np.shape(self.x))

        self.u_sm_2 = self.h_0_sm_2 * np.ones(np.shape(self.x))
        self.h_u_sm_2 = self.h_0_sm_2 * np.ones(np.shape(self.x))

        # Parameters for task duration field u_d
        self.h_0_d = 0
        self.tau_h_d = 20
        self.theta_d = 1.5

        self.kernel_pars_d = (1, 0.7, 0.9)
        # Fourier transform of the kernel function
        self.w_hat_d = np.fft.fft(self.kernel_osc(*self.kernel_pars_d))
        self.u_d = self.h_0_d * np.ones(np.shape(self.x))
        self.h_u_d = self.h_0_d * np.ones(np.shape(self.x))

    def input_callback(self, msg):

        # received_data = np.array(msg.data)
        # # Extract the two matrices from the flattened data
        # # Assuming each matrix has the same length as the x-dimension of the grid
        # n = len(received_data) // 2  # Since both matrices are of equal size

        # # Split the received data back into two matrices
        # matrix_1 = received_data[:n]
        # matrix_2 = received_data[n:]

        received_data = np.array(msg.data)
        self.latest_input_slice = received_data
        n = len(self.latest_input_slice) // 3

        input_agent1 = self.latest_input_slice[:n]
        input_agent2 = self.latest_input_slice[n:2*n]

        self.get_logger().info(f"1ST INPUT MAX {max(input_agent1)}")
        self.get_logger().info(f"2ND INPUT MAX {max(input_agent2)}")

        # self.get_logger().info(f"2ND INPUT MAX {max(matrix_2)}")

        # with self._lock:
        #     self.latest_input_slice = matrix_1
        #     self.latest_input_slice_2 = matrix_2

        # 2nd input
        # 2nd plot

        # Update time counter
        self.time_counter += self.dt
        max_value = input_agent1.max()

        # Input at time t=0 for the task duration field
        if 0.0 <= self.time_counter < 1.0:
            input_d = self.gaussian(0, 5.0, 2.0)
        else:
            input_d = 0.0

        f_d = np.heaviside(self.u_d - self.theta_d, 1)
        f_hat_d = np.fft.fft(f_d)
        conv_d = self.dx * \
            np.fft.ifftshift(np.real(np.fft.ifft(f_hat_d * self.w_hat_d)))

        f_sm = np.heaviside(self.u_sm - self.theta_sm, 1)
        f_hat_sm = np.fft.fft(f_sm)
        conv_sm = self.dx * \
            np.fft.ifftshift(np.real(np.fft.ifft(f_hat_sm * self.w_hat_sm)))

        f_sm_2 = np.heaviside(self.u_sm_2 - self.theta_sm_2, 1)
        f_hat_sm_2 = np.fft.fft(f_sm_2)
        conv_sm_2 = self.dx * \
            np.fft.ifftshift(np.real(np.fft.ifft(f_hat_sm_2 * self.w_hat_sm)))

        self.h_u_d += self.dt / self.tau_h_d * f_d
        self.u_d += self.dt * (-self.u_d + conv_d + input_d + self.h_u_d)

        self.h_u_sm += self.dt / self.tau_h_sm * f_sm
        self.u_sm += self.dt * (-self.u_sm + conv_sm +
                                input_agent1 + self.h_u_sm)

        self.h_u_sm_2 += self.dt / self.tau_h_sm * f_sm_2
        self.u_sm_2 += self.dt * (-self.u_sm_2 + conv_sm_2 +
                                  input_agent2 + self.h_u_sm_2)

        # List of input positions
        input_positions = [-40, 0, 40]

        # Convert `input_positions` to indices in `self.x`
        input_indices = [np.argmin(np.abs(self.x - pos))
                         for pos in input_positions]

        # Store the values of u_sm at the specified positions in u_sm_history
        u_sm_values_at_positions = [self.u_sm[idx] for idx in input_indices]
        self.u_sm_history.append(u_sm_values_at_positions)

        u_sm_values_at_positions_2 = [self.u_sm_2[idx]
                                      for idx in input_indices]
        self.u_sm_2_history.append(u_sm_values_at_positions_2)

        center_index = len(self.u_d) // 2
        u_d_values_at_position = self.u_d[center_index]
        self.u_d_history.append(u_d_values_at_position)

        # Check if time counter has reached the next full step
        if int(self.time_counter) >= self.current_step:
            print(
                f"Received input at time step {self.current_step} with max amplitude: {max_value:.2f}")
            self.current_step += 1  # Move to the next full step

    # # Check if time counter exceeds or is very close to time limit
    #     if int(self.time_counter) == int(self.t_lim):  # Using a small tolerance
    #         print("Learning finished.")
    #         self.save_sequence_memory()  # Save data before shutting down
    #         rclpy.shutdown()  # Terminate the node

    def plt_func(self, _):

        # Update the plot with the latest data
        with self._lock:

            self.line_u_sm_1.set_ydata(self.u_sm)
            self.line_u_sm_2.set_ydata(self.u_sm_2)

        return self.line_u_sm_1, self.line_u_sm_2
        # Ensure data is of expected shape and type
        # self.u_sm = self.u_sm.reshape(-1)  # Ensure 1D
        # # Ensure 1D
        # self.latest_input_slice = self.latest_input_slice.reshape(-1)

        # # Set the data for plotting
        # self.line.set_ydata(self.u_sm)  # Update u_field line
        # # Update latest_input_slice line
        # self.line_input.set_ydata(self.latest_input_slice)
        # self.line_u_d.set_ydata(self.u_d)  # Update u_d line
        # return self.line, self.line_input, self.line_u_d

    def _plt(self):
        # Start the animation
        self.ani = anim.FuncAnimation(self.fig, self.plt_func, interval=100)
        plt.show()

    def kernel_osc(self, a, b, alpha):
        return a * (np.exp(-b * abs(self.x)) * ((b * np.sin(abs(alpha * self.x))) + np.cos(alpha * self.x)))

    def kernel_gauss(self, a_ex, s_ex, w_in):
        return a_ex * np.exp(-0.5 * self.x ** 2 / s_ex ** 2) - w_in

    def gaussian(self, center=0, amplitude=1.0, width=1.0):
        return amplitude * (np.exp(-((self.x - center) ** 2) / (2 * (width ** 2))))

    def save_sequence_memory(self):
        # Create directory if it doesn't exist
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Get current date and time for file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{data_dir}/sequence_memory_{timestamp}.npy"

        filename_sm_2 = f"{data_dir}/sequence_2_{timestamp}.npy"

        filename_d = f"{data_dir}/task_duration_{timestamp}.npy"

        filename_h = f"{data_dir}/sequence_history_{timestamp}.npy"

        filename_h_2 = f"{data_dir}/sequence_history2_{timestamp}.npy"

        filename_dur_h = f"{data_dir}/duration_history_{timestamp}.npy"

        # Save u_sm data as a .npy file
        np.save(filename, self.u_sm)
        print(f"Sequence memory saved to {filename}")

        np.save(filename_sm_2, self.u_sm_2)
        print(f"Sequence memory 2 saved to {filename}")

        np.save(filename_d, self.u_d)
        print(f"Task duration saved to {filename_d}")

        np.save(filename_h, self.u_sm_history)
        print(f"Sequence history saved to {filename_h}")

        np.save(filename_h_2, self.u_sm_2_history)
        print(f"Sequence history 2 saved to {filename_h_2}")

        np.save(filename_dur_h, self.u_d_history)
        print(f"Task duration history saved to {filename_dur_h}")


def main(args=None):
    rclpy.init(args=args)
    node = DNFModel()

    # Multi-threaded executor to handle ROS spinning
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    # Run executor in a separate thread
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    try:
        node._plt()  # Start the plotting function
    except KeyboardInterrupt:
        pass
    finally:
        node.save_sequence_memory()
        node.destroy_node()
        rclpy.shutdown()
        plt.close()


if __name__ == '__main__':
    main()
