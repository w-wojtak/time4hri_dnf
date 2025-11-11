
# Dynamic Neural Field Cognitive Architecture (ROS 2)

This package implements a **Dynamic Neural Field (DNF)** cognitive architecture for simulating **sequence learning and recall** in human–robot interaction (HRI) scenarios.
It reproduces the computational models presented in:

> Wojtak, W., Ferreira, F., Louro, L., Bicho, E., & Erlhagen, W. (2023). Adaptive timing in a dynamic field architecture for natural human–robot interactions. Cognitive Systems Research, 82, 101148. https://www.sciencedirect.com/science/article/pii/S1389041723000761


This version is implemented in **ROS 2 (Humble)** using pure **Python**, and runs entirely in simulation.
It allows the study of **temporal adaptation**, **error detection**, and other cognitive mechanisms without requiring real robot hardware.

---

## Overview

The architecture consists of several interconnected neural fields that model cognitive processes of **sequence learning** and **memory-based recall**.

* **Learning phase:** The system observes a sequence of stimuli (representing human actions) and encodes them as dynamic memory traces.
* **Recall phase:** The system replays the stored sequence autonomously, simulating timing and order of execution.

### Extensions

* **Temporal adaptation** – Adjusts internal timing between learned actions.
* **Error monitoring** – Detects unexpected or out-of-order events during recall.

---

## System Requirements

* Ubuntu 22.04 LTS (tested on WSL 2)
* ROS 2 Humble
* Python 3.10
* Build system: `colcon`

---

## Package Structure

```
dnf_cognitive_architecture_extended/
├── launch/
│   ├── dnf_learn_launch.py
│   ├── dnf_learn_basic_launch.py
│   ├── dnf_recall_launch.py
│   ├── dnf_recall_basic_launch.py
│   └── dnf_experiment_launch.py
├── dnf_model_learning.py              # Full learning (with adaptation)
├── dnf_model_learning_basic.py        # Simplified learning
├── dnf_model_recall.py                # Full recall
├── dnf_model_recall_basic.py          # Basic recall
├── dnf_model_recall_with_error.py     # Recall with error detection
├── input_matrix.py                    # Input preprocessing (full)
├── input_matrix_basic.py              # Simplified input preprocessing
├── input_matrix_error.py              # Inputs for error detection
├── output_node.py                     # Simulated output (action execution)
├── udp_listener_node.py               # Optional UDP input stream
├── udp_response_sender_node.py        # Sends UDP feedback
└── ...
```

---

## Features

* DNF-based **learning and recall** of action sequences
* **Temporal adaptation** of timing between actions
* **Error detection** for sequence violations
* Modular **ROS 2 node** architecture
* **Real-time plotting** and visualization of neural field activity
* Fully simulation-based (no hardware required)

---

## Installation & Build

This package should be placed inside a ROS 2 workspace and built with `colcon`.

### Prerequisites

* Ubuntu 22.04 or WSL2 Ubuntu 22.04
* ROS 2 Humble (`source /opt/ros/humble/setup.bash`)
* `colcon` build tool
* Python 3.10 with NumPy, SciPy, Matplotlib

### Clone the Repository

```bash
git clone https://github.com/w-wojtak/time4hri_dnf.git
cd time4hri_dnf
```

### Build the Workspace

```bash
colcon build
```

### Source the Workspace

```bash
source install/setup.bash
```

---

## Usage

This package provides launch files for both **learning** and **recall** phases, as well as combined experiment launches.

### Learning Phase

Run the DNF learning phase to encode a sequence:

```bash
ros2 launch dnf_cognitive_architecture_extended dnf_learn_launch.py
```

Nodes launched:

* `input_matrix` — Processes simulated input events
* `dnf_model_learning` — Core DNF learning node

Simplified version (temporal adaptation only):

```bash
ros2 launch dnf_cognitive_architecture_extended dnf_learn_basic_launch.py
```

---

### Recall Phase

Run the DNF recall phase to replay a stored sequence:

```bash
ros2 launch dnf_cognitive_architecture_extended dnf_recall_launch.py
```

Optionally specify a trial number:

```bash
ros2 launch dnf_cognitive_architecture_extended dnf_recall_launch.py trial_number:=3
```

Nodes launched:

* `dnf_model_recall` — Recall node
* `output_node` — Publishes simulated action commands

Simplified version:

```bash
ros2 launch dnf_cognitive_architecture_extended dnf_recall_basic_launch.py
```

---

### Full Experiment

Run a full experiment consisting of one learning phase followed by multiple recall trials:

```bash
ros2 launch dnf_cognitive_architecture_extended dnf_experiment_launch.py
```

Basic version (temporal adaptation only):

```bash
ros2 launch dnf_cognitive_architecture_extended dnf_experiment_basic_launch.py
```

---

## Customization Notes

* Input sequences and parameters can be modified in `input_matrix*.py`.
* Visualization and plotting options can be enabled in the DNF scripts (`plot_activity=True`).
* UDP interfaces (`udp_listener_node.py`, `udp_response_sender_node.py`) can be used to exchange data with external simulators or sensors.
 
