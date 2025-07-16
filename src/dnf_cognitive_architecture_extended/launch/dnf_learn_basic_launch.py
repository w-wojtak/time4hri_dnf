from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='dnf_cognitive_architecture_extended',
            executable='input_matrix_basic',
            name='input_matrix_basic'
        ),
        Node(
            package='dnf_cognitive_architecture_extended',
            executable='dnf_model_learning_basic',
            name='dnf_model_learning_basic'
        )
    ])
