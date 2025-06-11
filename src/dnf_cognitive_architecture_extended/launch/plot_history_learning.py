import numpy as np
import matplotlib.pyplot as plt
import os

# Function to load the u_sm history


def load_u_sm_history(filename):
    u_sm_history = np.load(filename, allow_pickle=True)
    return u_sm_history


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

# Function to plot the u_sm history


def plot_u_sm_history(u_sm_history, input_positions, u_d_history):
    u_sm_history = np.array(u_sm_history)

    plt.figure(figsize=(10, 6))
    for i, position in enumerate(input_positions):
        plt.plot(
            u_sm_history[:, i], label=f"Position {position} (u_sm)", linestyle='-', marker='o')

    plt.plot(u_d_history)
    plt.title("u_sm History at Different Positions")
    plt.xlabel("Time Step")
    plt.ylabel("u_sm Value")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def main():

    # Adjust the file path using os.path to go one level up to reach the 'data' folder
    data_dir = os.path.join(
        os.getcwd(), 'src/dnf_cognitive_architecture_extended/dnf_cognitive_architecture_extended/data')
    # filename_h = os.path.join(data_dir, 'sequence_history_20241216_165951.npy')
    filename_h = get_latest_file(data_dir, 'sequence_history_')

    filename_d = get_latest_file(data_dir, 'duration_history_')

    print(f"Trying to load file: {filename_h}")

    # Check if the file exists
    if os.path.exists(filename_h):
        u_sm_history = load_u_sm_history(filename_h)
        u_d_history = load_u_sm_history(filename_d)
        input_positions = [-40, 0, 40]
        print(u_sm_history.shape)
        print(u_d_history.shape)
        plot_u_sm_history(u_sm_history, input_positions, u_d_history)

    else:
        print(f"File {filename_h} does not exist.")


if __name__ == "__main__":
    main()
