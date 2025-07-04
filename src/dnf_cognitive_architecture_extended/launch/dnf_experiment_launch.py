from launch import LaunchDescription
from launch.actions import TimerAction, IncludeLaunchDescription, Shutdown
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir
import os


def generate_launch_description():
    launch_dir = os.path.join(
        os.path.dirname(__file__)
    )

    # --- Learning Phase ---
    learning_nodes = [
        Node(
            package='dnf_cognitive_architecture_extended',
            executable='input_matrix',
            name='input_matrix_learning'
        ),
        Node(
            package='dnf_cognitive_architecture_extended',
            executable='dnf_model_learning',
            name='dnf_model_learning'
        )
    ]

    # --- Recall Phase Template ---
    def recall_phase(trial_num, delay_sec):
        return TimerAction(
            period=delay_sec,
            actions=[
                Node(
                    package='dnf_cognitive_architecture_extended',
                    executable='input_matrix',
                    name=f'input_matrix_recall_{trial_num}'
                ),
                Node(
                    package='dnf_cognitive_architecture_extended',
                    executable='dnf_model_recall',
                    name=f'dnf_model_recall_{trial_num}',
                    parameters=[{'trial_number': trial_num}]
                ),
                Node(
                    package='dnf_cognitive_architecture_extended',
                    executable='output_node',
                    name=f'output_node_{trial_num}'
                )
            ]
        )

    # Delay to allow learning phase to complete (e.g. 10 seconds)
    recall_1 = recall_phase(1, delay_sec=100.0)
    # Assumes each recall takes ~15s
    recall_2 = recall_phase(2, delay_sec=200.0)
    recall_3 = recall_phase(3, delay_sec=300.0)

    return LaunchDescription(
        learning_nodes + [recall_1, recall_2, recall_3]
    )
