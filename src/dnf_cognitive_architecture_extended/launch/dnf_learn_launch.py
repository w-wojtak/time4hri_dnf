from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='dnf_cognitive_architecture_extended',
            executable='rabbitmq_input_matrix',
            name='rabbitmq_input_matrix'
        ),
        Node(
            package='dnf_cognitive_architecture_extended',
            executable='dnf_model_learning',  # Replace this with your second node
            name='dnf_model_learning'
        )
    ])
