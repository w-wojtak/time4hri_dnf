# dnf_cognitive_architecture_extended

## Package Overview

**dnf_cognitive_architecture_extended** is a cognitive architecture consisting of several interconnected dynamic neural fields.  
It is designed for human-robot interaction experiments with two main phases:  

- **Learning phase:** The robot observes and encodes a sequence of human actions.  
- **Recall phase:** The robot executes the learned action sequence from memory.  

### System Requirements

- Ubuntu 22.04.5 LTS on WSL2  
- ROS 2 Humble  
- Python 3.10.12  
- Build system: Colcon  

### Features

- Encoding memory of sequences of observed human actions using dynamic neural fields  
- Executing learned sequences by sending commands to the robot controller  
- Adaptive timing to synchronize robot actions with human partner timing  
- Real-time plotting and visualization of neural field activity  
- Detection of human errors, such as actions performed out of order  


### Project Structure

```

dnf_cognitive_architecture_extended/
├── launch/
│   ├── dnf_learn_launch.py
│   └── dnf_recall_launch.py
├── dnf_model_learning.py
├── dnf_model_recall.py
├── input_matrix.py
├── output_node.py
├── ...
```


## Installation and Build Instructions

This package is designed for ROS 2 Humble on Ubuntu 22.04 (tested on WSL2). It is built using `colcon` in a ROS 2 workspace.

### Prerequisites

- Ubuntu 22.04 or compatible (e.g., WSL2 Ubuntu 22.04)  
- ROS 2 Humble installed and sourced 
- Python 3.10  
- `colcon` build tool installed  

### Clone the Repository

```bash
git clone https://github.com/w-wojtak/time4hri_dnf.git
cd time4hri_dnf
```


### Build the Workspace

`colcon build`

### Source the Workspace
Before running any ROS 2 commands or launching nodes:

`source install/setup.bash`


## Usage
This package includes launch files for different versions of the two experimental phases.

### Learning Phase
Launch nodes to observe and encode a sequence of actions:

`ros2 launch dnf_cognitive_architecture_extended dnf_learn_launch.py`

Nodes launched:

* `input_matrix` — processes input from human activity
* `dnf_model_learning` — core learning node using dynamic neural fields

### Recall Phase
Launch nodes to execute the previously learned sequence:

`ros2 launch dnf_cognitive_architecture_extended dnf_recall_launch.py`

You can optionally specify a trial number:

`ros2 launch dnf_cognitive_architecture_extended dnf_recall_launch.py trial_number:=3`

Nodes launched:

* `dnf_model_recall` — recall node that uses the trial number to select memory
* `output_node` — sends action commands to the robot

There are other launch files for launching different versions of learnign and recall, e.g.

`dnf_learn_basic_launch.py`, `dnf_recall_basic_launch.py` 

To launch an experiment with one learning trial and three recall trials, run:

`ros2 launch dnf_cognitive_architecture_extended dnf_experiment_basic_launch.py`

or

`ros2 launch dnf_cognitive_architecture_extended dnf_experiment_extended_launch.py`



## License
To be specified.
