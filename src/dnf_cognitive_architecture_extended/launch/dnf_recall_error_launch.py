from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'trial_number',
            default_value='1',
            description='Trial number for the recall mode'
        ),
        Node(
            package='dnf_cognitive_architecture_extended',
            executable='input_matrix_error',
            name='input_matrix_error'
        ),
        Node(
            package='dnf_cognitive_architecture_extended',
            executable='dnf_model_recall_with_error',
            name='dnf_model_recall_with_error',
            parameters=[{'trial_number': LaunchConfiguration('trial_number')}]
        ),
        Node(
            package='dnf_cognitive_architecture_extended',
            executable='output_node',
            name='output_node'
        )
    ])
