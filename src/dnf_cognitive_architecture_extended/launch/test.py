import numpy as np
import matplotlib.pyplot as plt
import os

# import matplotlib
# matplotlib.use('Agg')
# import tkinter
# print("Tkinter is available")


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

# x = np.linspace(0, 10, 100)
# y = np.sin(x)
# # plt.figure()
# print("lol")
# plt.plot(x, y)
# plt.title("Test Plot")
# plt.xlabel("x")
# plt.ylabel("sin(x)")
# plt.show()
# print("lol")


data_dir = os.path.join(
    os.getcwd(), 'src/dnf_architecture_extended/dnf_architecture_extended/data')
# filename_h = os.path.join(data_dir, 'sequence_history_20241216_165951.npy')
print(data_dir)
filename = get_latest_file(data_dir, 'task_duration_')
print(filename)

u_d = np.load(filename, allow_pickle=True)

print(max(u_d))


filename2 = get_latest_file(data_dir, 'sequence_memory_')

u_act = np.load(filename2, allow_pickle=True)
# u_act = load_sequence_memory()

print(max(u_d) - max(u_act))

x_lim = 80
dx = 0.2

# Spatial grid
# x = np.arange(-x_lim, x_lim + dx, dx)
# plt.plot(x, u_d-max(u_act))
# plt.show()

filenameh = get_latest_file(data_dir, 'h_amem')

h = np.load(filenameh, allow_pickle=True)

print(max(h))

x = np.arange(-x_lim, x_lim + dx, dx)
plt.plot(x, h)
plt.show()
