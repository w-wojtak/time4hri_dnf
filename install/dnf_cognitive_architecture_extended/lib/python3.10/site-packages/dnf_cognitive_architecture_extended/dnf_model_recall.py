#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import threading
from datetime import datetime
import os
import time


class DNFModelWM(Node):
    def __init__(self):
        super().__init__("dnf_model_recall")

        # Declare the 'trial_number' parameter
        self.declare_parameter('trial_number', 1)  # Default value is 1

        # Get the value of 'trial_number'
        self.trial_number = self.get_parameter(
            'trial_number').get_parameter_value().integer_value

        # Log the trial number
        self.get_logger().info(
            f"Recall node started with trial_number: {self.trial_number}")

        # Persistently track threshold crossings
        self.threshold_crossed = {pos: False for pos in [-40, 0, 40]}

        # Spatial and temporal parameters
        self.x_lim = 80
        self.t_lim = 15
        self.dx = 0.2
        self.dt = 0.1

        # Spatial grid
        self.x = np.arange(-self.x_lim, self.x_lim + self.dx, self.dx)

        # Lock for threading
        self._lock = threading.Lock()

        # Publisher
        self.publisher = self.create_publisher(
            Float32MultiArray, 'threshold_crossings', 10)

        # Create a subscriber to the 'input_matrices_combined' topic
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'input_matrices_combined',
            self.process_inputs,
            10
        )

        # Variable to store the latest input slice
        self.latest_input_slice = np.zeros_like(self.x)

        # Timer to publish every 1 second
        self.timer = self.create_timer(1.0, self.process_inputs)

        # Initialize figure and axes for plotting
        # self.fig, (self.ax1, self.ax2, self.ax3,
        #            self.ax4) = plt.subplots(2, 2, figsize=(10, 10))
        # Initialize figure and axes for plotting
        self.fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6 = axes.flatten()

        # Plot for u_act on ax1
        self.line_act, = self.ax1.plot(
            self.x, np.zeros_like(self.x), label="u_act")
        self.ax1.set_xlim(-self.x_lim, self.x_lim)
        self.ax1.set_ylim(-5, 5)  # Adjust based on expected amplitude
        self.ax1.set_xlabel("x")
        self.ax1.set_ylabel("u_act(x)")
        self.ax1.set_title("Action Onset Field")
        self.ax1.legend()

        # Plot for u_wm on ax2
        self.line_wm, = self.ax3.plot(
            self.x, np.zeros_like(self.x), label="u_wm")
        self.ax3.set_xlim(-self.x_lim, self.x_lim)
        self.ax3.set_ylim(-3, 5)  # Adjust based on expected amplitude
        self.ax3.set_xlabel("x")
        self.ax3.set_ylabel("u_wm(x)")
        self.ax3.set_title("Working Memory Field")
        self.ax3.legend()

        self.line_f1, = self.ax4.plot(
            self.x, np.zeros_like(self.x), label="u_act")

        self.ax4.set_xlim(-self.x_lim, self.x_lim)
        self.ax4.set_ylim(-5, 5)  # Adjust based on expected amplitude
        self.ax4.set_xlabel("x")
        self.ax4.set_ylabel("u_f1(x)")
        self.ax4.set_title("Feedback 1 Field")
        self.ax4.legend()

        self.line_f2, = self.ax5.plot(
            self.x, np.zeros_like(self.x), label="u_act")

        self.ax5.set_xlim(-self.x_lim, self.x_lim)
        self.ax5.set_ylim(-5, 5)  # Adjust based on expected amplitude
        self.ax5.set_xlabel("x")
        self.ax5.set_ylabel("u_f2(x)")
        self.ax5.set_title("Feedback 2 Field")
        self.ax5.legend()

        self.line_sim, = self.ax2.plot(
            self.x, np.zeros_like(self.x), label="u_sim")
        self.ax2.set_xlim(-self.x_lim, self.x_lim)
        self.ax2.set_ylim(-5, 5)  # Adjust based on expected amplitude
        self.ax2.set_xlabel("x")
        self.ax2.set_ylabel("u_sim(x)")
        self.ax2.set_title("Simulation Field")
        self.ax2.legend()

        self.line_error, = self.ax6.plot(
            self.x, np.zeros_like(self.x), label="u_error")
        self.ax6.set_xlim(-self.x_lim, self.x_lim)
        self.ax6.set_ylim(-5, 5)  # Adjust based on expected amplitude
        self.ax6.set_xlabel("x")
        self.ax6.set_ylabel("u_error(x)")
        self.ax6.set_title("Error Field")
        self.ax6.legend()

        # Initialize `u_act` and `u_d` with loaded data or default
        try:
            self.u_d = load_task_duration().flatten()
            self.h_d_initial = max(self.u_d)

            if self.trial_number == 1:

                # self.h_d_initial = 4.1067860962773075

                # Ensure it's 1D and shift as needed
                # u0 = max(load_sequence_memory().flatten())
                self.u_act = load_sequence_memory().flatten() - self.h_d_initial + 1.5
                self.input_action_onset = load_sequence_memory().flatten()
                self.h_u_act = -self.h_d_initial * \
                    np.ones(np.shape(self.x)) + 1.5

                self.u_sim = load_sequence_memory_2().flatten() - self.h_d_initial + 1.5
                self.input_action_onset_2 = load_sequence_memory_2().flatten()
                self.h_u_sim = -self.h_d_initial * \
                    np.ones(np.shape(self.x)) + 1.5
                # self.get_logger().info(f"Initial h: {-self.h_d_initial}")
                # self.get_logger().info(f"Initial u0: {u0}")
                # self.get_logger().info(f"Initial u act : {max(self.u_act)}")
            else:
                data_dir = os.path.join(
                    os.getcwd(), 'dnf_architecture_extended/data')
                self.get_logger().info(f"Loading from {data_dir}")
                latest_h_amem_file = get_latest_file(data_dir, 'h_amem')
                latest_h_amem = np.load(latest_h_amem_file, allow_pickle=True)

                self.u_act = load_sequence_memory().flatten() - self.h_d_initial + \
                    1.5 + latest_h_amem
                self.input_action_onset = load_sequence_memory().flatten() + latest_h_amem
                self.h_u_act = -self.h_d_initial * \
                    np.ones(np.shape(self.x)) + 1.5

        except FileNotFoundError:
            # print("No previous sequence memory found, initializing with default values.")
            # self.u_act = np.zeros(np.shape(self.x))  # Default initialization
            # self.h_0_act = -3.2
            # self.h_u_act = self.h_0_act * np.ones(np.shape(self.x))
            self.get_logger().info(f"No previous sequence memory found.")

        # Parameters specific to working memory
        self.h_0_wm = -1.0
        self.theta_wm = 0.8

        # self.kernel_pars_wm = (1.5, 0.5, 0.75)
        # self.kernel_pars_wm = (2, 0.5, 0.6)
        self.kernel_pars_wm = (1.75, 0.5, 0.8)
        self.w_hat_wm = np.fft.fft(self.kernel_osc(*self.kernel_pars_wm))

        # self.kernel_pars_inhib = (2.5, 0.5, 0.8)
        # # Fourier transform of the kernel function
        # self.w_hat_inhib = np.fft.fft(self.kernel_osc(*self.kernel_pars_inhib))

        # initialization
        self.u_wm = self.h_0_wm * np.ones(np.shape(self.x))
        self.h_u_wm = self.h_0_wm * np.ones(np.shape(self.x))

        # Parameters specific to action onset

        self.tau_h_act = 20
        self.theta_act = 1.5

        self.tau_h_sim = 10
        self.theta_sim = 1.5

        self.theta_error = 1.5

        self.kernel_pars_act = (1.5, 0.8, 0.0)
        self.w_hat_act = np.fft.fft(self.kernel_gauss(*self.kernel_pars_act))

        self.kernel_pars_sim = (1.7, 0.8, 0.7)
        self.w_hat_sim = np.fft.fft(self.kernel_gauss(*self.kernel_pars_sim))

        # feedback fields - decision fields, similar to u_act
        self.h_f = -1.0
        self.w_hat_f = self.w_hat_act

        self.tau_h_f = self.tau_h_act
        self.theta_f = self.theta_act

        self.u_f1 = self.h_f * np.ones(np.shape(self.x))
        self.u_f2 = self.h_f * np.ones(np.shape(self.x))

        self.u_error = self.h_f * np.ones(np.shape(self.x))

        self.u_act_history = []  # Lists to store values at each time step
        self.u_sim_history = []
        self.u_wm_history = []
        self.u_f1_history = []
        self.u_f2_history = []
        self.u_error_history = []

        # initialize h level for the adaptation
        self.h_u_amem = np.zeros(np.shape(self.x))
        self.beta_adapt = 0.01

    def process_inputs(self, msg=None):
        """Process recall by receiving msg from subscription or by timer."""

        if msg:
            # Handle incoming message from subscriber
            received_data = np.array(msg.data)
            self.latest_input_slice = received_data
            # self.get_logger().info(
            #     f"Received data from subscription: {self.latest_input_slice}")

            # Handle the logic for both subscription and timer (without msg)
            self.perform_recall()

    def perform_recall(self):

        # received_data = np.array(msg.data)
        # # Extract the two matrices from the flattened data
        # # Assuming each matrix has the same length as the x-dimension of the grid
        # Since both matrices are of equal size
        n = len(self.latest_input_slice) // 3

        # Split the received data back into two matrices
        input_agent1 = self.latest_input_slice[:n]
        input_agent2 = self.latest_input_slice[n:2*n]

        input_agent_robot_feedback = self.latest_input_slice[2*n:]

        # self.get_logger().info(
        #     f"Received size input: {len(self.u_act)}")

        f_f1 = np.heaviside(self.u_f1 - self.theta_f, 1)
        f_hat_f1 = np.fft.fft(f_f1)
        conv_f1 = self.dx * \
            np.fft.ifftshift(np.real(np.fft.ifft(f_hat_f1 * self.w_hat_f)))

        f_f2 = np.heaviside(self.u_f2 - self.theta_f, 1)
        f_hat_f2 = np.fft.fft(f_f2)
        conv_f2 = self.dx * \
            np.fft.ifftshift(np.real(np.fft.ifft(f_hat_f2 * self.w_hat_f)))

        f_act = np.heaviside(self.u_act - self.theta_act, 1)
        f_hat_act = np.fft.fft(f_act)
        conv_act = self.dx * \
            np.fft.ifftshift(np.real(np.fft.ifft(f_hat_act * self.w_hat_act)))

        f_sim = np.heaviside(self.u_sim - self.theta_sim, 1)
        f_hat_sim = np.fft.fft(f_sim)
        conv_sim = self.dx * \
            np.fft.ifftshift(np.real(np.fft.ifft(f_hat_sim * self.w_hat_sim)))

        f_wm = np.heaviside(self.u_wm - self.theta_wm, 1)
        f_hat_wm = np.fft.fft(f_wm)
        conv_wm = self.dx * \
            np.fft.ifftshift(np.real(np.fft.ifft(f_hat_wm * self.w_hat_wm)))

        f_error = np.heaviside(self.u_error - self.theta_error, 1)
        f_hat_error = np.fft.fft(f_error)
        conv_error = self.dx * \
            np.fft.ifftshift(
                np.real(np.fft.ifft(f_hat_error * self.w_hat_act)))

        # conv_inhib = self.dx * \
        #     np.fft.ifftshift(np.real(np.fft.ifft(f_hat_wm * self.w_hat_inhib)))

        # Update field states
        self.h_u_act += self.dt / self.tau_h_act
        # self.u_act += self.dt * (-self.u_act + conv_act + self.input_action_onset +
        #                          self.h_u_act - 1.9 * f_wm * self.u_wm)
        self.h_u_sim += self.dt / self.tau_h_sim

        self.u_act += self.dt * (-self.u_act + conv_act + self.input_action_onset +
                                 self.h_u_act - 6.0 * f_wm * conv_wm)

        self.u_sim += self.dt * (-self.u_sim + conv_sim + self.input_action_onset_2 +
                                 self.h_u_sim - 6.0 * f_wm * conv_wm)

        # self.u_wm += self.dt * (-self.u_wm + conv_wm +
        #                         self.h_u_wm + f_act * self.u_act)
        self.u_wm += self.dt * \
            (-self.u_wm + conv_wm + 6*((f_f1*self.u_f1)*(f_f2*self.u_f2)) + self.h_u_wm)

        self.u_f1 += self.dt * (-self.u_f1 + conv_f1 + input_agent_robot_feedback +
                                self.h_f - 1 * f_wm * conv_wm)

        self.u_f2 += self.dt * (-self.u_f2 + conv_f2 + input_agent2 +
                                self.h_f - 1 * f_wm * conv_wm)

        self.u_error += self.dt * (-self.u_error + conv_error +
                                   self.h_f - 2 * f_sim * conv_sim)

        self.h_u_amem += self.beta_adapt*(1 - (f_f2 * f_f1)) * (f_f1 - f_f2)

        # List of input positions where we previously applied inputs
        input_positions = [-40, 0, 40]

        # Convert `input_positions` to indices in `self.x`
        input_indices = [np.argmin(np.abs(self.x - pos))
                         for pos in input_positions]

        # Store the values of u_sm at the specified positions in u_sm_history
        u_act_values_at_positions = [self.u_act[idx] for idx in input_indices]
        self.u_act_history.append(u_act_values_at_positions)

        u_sim_values_at_positions = [self.u_sim[idx] for idx in input_indices]
        self.u_sim_history.append(u_sim_values_at_positions)

        u_wm_values_at_positions = [self.u_wm[idx] for idx in input_indices]
        self.u_wm_history.append(u_wm_values_at_positions)

        u_f1_values_at_positions = [self.u_f1[idx] for idx in input_indices]
        self.u_f1_history.append(u_f1_values_at_positions)

        u_f2_values_at_positions = [self.u_f2[idx] for idx in input_indices]
        self.u_f2_history.append(u_f2_values_at_positions)

        # Track current time
        current_time = self.get_clock().now().to_msg().sec

        # Check `u_act` values at exact input indices for threshold crossings
        for i, idx in enumerate(input_indices):
            position = input_positions[i]

            # Only proceed if the threshold has not yet been crossed for this input position
            if not self.threshold_crossed[position] and self.u_act[idx] > self.theta_act:
                # Debugging line
                print(
                    f"Threshold crossed at position {position} with u_act = {self.u_act[idx]}")
                threshold_msg = Float32MultiArray()
                threshold_msg.data = [float(position)]
                self.publisher.publish(threshold_msg)
                self.threshold_crossed[position] = True

        # Debug to confirm changes to the threshold state
        # print(f"Threshold states after check: {self.threshold_crossed}")

    def plt_func(self, _):
        # Update the plot with the latest data for both fields
        self.line_act.set_ydata(self.u_act)
        self.line_sim.set_ydata(self.u_sim)
        self.line_wm.set_ydata(self.u_wm)
        self.line_f1.set_ydata(self.u_f1)
        self.line_f2.set_ydata(self.u_f2)
        self.line_error.set_ydata(self.u_error)
        return self.line_act, self.line_wm, self.line_f1, self.line_f2, self.line_sim, self.line_error

    def _plt(self):
        # Start the animation
        self.ani = anim.FuncAnimation(self.fig, self.plt_func, interval=100)
        plt.show()

    def kernel_osc(self, a, b, alpha):
        return a * (np.exp(-b * abs(self.x)) * ((b * np.sin(abs(alpha * self.x))) + np.cos(alpha * self.x)))

    def kernel_gauss(self, a_ex, s_ex, w_in):
        return a_ex * np.exp(-0.5 * self.x ** 2 / s_ex ** 2) - w_in

    def save_working_memory(self):
        # Create directory if it doesn't exist
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Get current date and time for file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{data_dir}/working_memory_{timestamp}.npy"

        # Save u_wm data as a .npy file
        np.save(filename, self.u_wm)
        print(f"Working memory saved to {filename}")

    def save_history(self):

        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        self.get_logger().info(f"SAVING HISTORY to {data_dir}")
        self.get_logger().info(
            f"SAVING HISTORY SIZE U ACT {len(self.u_act_history)}")
        # Create directory if it doesn't exist

        # Get current date and time for file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save data as a .npy files
        filename_act = f"{data_dir}/act_history_{timestamp}.npy"
        np.save(filename_act, self.u_act_history)

        filename_sim = f"{data_dir}/sim_history_{timestamp}.npy"
        np.save(filename_sim, self.u_sim_history)

        filename_wm = f"{data_dir}/wm_history_{timestamp}.npy"
        np.save(filename_wm, self.u_wm_history)

        filename_f1 = f"{data_dir}/f1_history_{timestamp}.npy"
        np.save(filename_f1, self.u_f1_history)

        filename_f2 = f"{data_dir}/f2_history_{timestamp}.npy"
        np.save(filename_f2, self.u_f2_history)

        filename_h_amem = f"{data_dir}/h_amem_{timestamp}.npy"
        np.save(filename_h_amem, self.h_u_amem)

        print(f"History saved.")


def load_sequence_memory(filename=None):
    data_dir = "data"
    if filename is None:
        # Filter files with the "sequence_memory_" prefix
        files = [f for f in os.listdir(data_dir) if f.startswith(
            "sequence_memory_") and f.endswith('.npy')]

        if not files:
            raise FileNotFoundError(
                "No 'sequence_memory_' files found in the 'data' folder.")

        # Get the latest file by modification time
        latest_file = max([os.path.join(data_dir, f)
                          for f in files], key=os.path.getmtime)
        filename = latest_file

        data = np.load(filename)
        print(f"Loaded sequence memory from {filename}")

        # Ensure data is 1D
        data = data.flatten()

        return data


def load_sequence_memory_2(filename=None):
    data_dir = "data"
    if filename is None:
        # Filter files with the "sequence_memory_" prefix
        files = [f for f in os.listdir(data_dir) if f.startswith(
            "sequence_2_") and f.endswith('.npy')]

        if not files:
            raise FileNotFoundError(
                "No 'sequence_2_' files found in the 'data' folder.")

        # Get the latest file by modification time
        latest_file = max([os.path.join(data_dir, f)
                          for f in files], key=os.path.getmtime)
        filename = latest_file

        data = np.load(filename)
        print(f"Loaded sequence memory from {filename}")

        # Ensure data is 1D
        data = data.flatten()

        return data


def load_task_duration(filename=None):
    data_dir = "data"
    if filename is None:
        # Filter files with the "sequence_memory_" prefix
        files = [f for f in os.listdir(data_dir) if f.startswith(
            "task_duration_") and f.endswith('.npy')]

        if not files:
            raise FileNotFoundError(
                "No 'task_duration_' files found in the 'data' folder.")

        # Get the latest file by modification time
        latest_file = max([os.path.join(data_dir, f)
                          for f in files], key=os.path.getmtime)
        filename = latest_file

    # Load the data from the selected file
    data = np.load(filename)
    print(f"Loaded sequence memory from {filename}")

    # Ensure data is 1D
    data = data.flatten()

    # Print size and max value of the loaded data
    print(f"Data size: {data.size}")
    print(f"Max value: {data.max()}")

    return data


def get_latest_file(data_dir, pattern):
    """Retrieve the latest file in the data directory matching the pattern."""
    files = [f for f in os.listdir(data_dir) if f.startswith(
        pattern) and f.endswith('.npy')]
    if not files:
        return None
    # Sort files by modified time
    files.sort(key=lambda f: os.path.getmtime(
        os.path.join(data_dir, f)), reverse=True)
    return os.path.join(data_dir, files[0])


def main(args=None):
    rclpy.init(args=args)
    node = DNFModelWM()

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
        node.save_history()
        node.destroy_node()
        rclpy.shutdown()
        plt.close()
    # rclpy.init(args=args)
    # node = DNFModelWM()

    # # Multi-threaded executor to handle ROS spinning
    # executor = MultiThreadedExecutor()
    # executor.add_node(node)

    # try:
    #     # Run the executor in a separate thread
    #     thread = threading.Thread(target=executor.spin, daemon=True)
    #     thread.start()

    #     # Start the plotting function
    #     node._plt()

    #     # Main loop to handle executor
    #     while rclpy.ok():
    #         # Run a single spin cycle
    #         executor.spin_once()
    #         time.sleep(0.1)  # Sleep to allow for smooth spinning

    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     # Ensure history is saved before shutting down
    #     node.save_history()  # Save history
    #     node.destroy_node()   # Destroy the node properly
    #     rclpy.shutdown()      # Shutdown ROS2
    #     plt.close()           # Close the plot


if __name__ == '__main__':
    main()
