import os
import numpy as np
import matplotlib.pyplot as plt

# Function to load the history


def load_history(filename):
    history = np.load(filename, allow_pickle=True)
    return history

# Function to plot the history in subplots


def plot_history(history, input_positions, ax, title):
    history = np.array(history)
    print("History shape:", history.shape)

    for i, position in enumerate(input_positions):
        ax.plot(history[:, i],
                label=f"Position {position}", linestyle='-', marker='o')

    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    ax.grid(True)


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


def main():
    # Directory containing saved data
    data_dir = os.path.join(
        os.getcwd(), 'src/dnf_cognitive_architecture_extended/dnf_cognitive_architecture_extended/data')

    # Ensure the data directory exists
    if not os.path.exists(data_dir):
        print("Data directory does not exist.")
        return

    # Get the latest files
    filenames = {
        "act": get_latest_file(data_dir, "act_history_"),
        "wm": get_latest_file(data_dir, "wm_history_"),
        "f1": get_latest_file(data_dir, "f1_history_"),
        "f2": get_latest_file(data_dir, "f2_history_"),
    }

    # Create a figure with a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    for (key, filepath), ax in zip(filenames.items(), axs.flat):
        if filepath:
            print(f"Loading {key} history from: {filepath}")
            history = load_history(filepath)
            input_positions = [-40, 0, 40]  # Adjust these based on your setup
            plot_history(history, input_positions,
                         ax, f"{key.upper()} History")
        else:
            print(f"No {key} history files found.")

    # Adjust layout for better spacing between subplots
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
