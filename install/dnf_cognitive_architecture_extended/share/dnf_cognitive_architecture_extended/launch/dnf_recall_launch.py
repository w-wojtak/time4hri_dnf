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
            executable='rabbitmq_input_matrix',
            name='rabbitmq_input_matrix'
        ),
        Node(
            package='dnf_cognitive_architecture_extended',
            executable='dnf_model_recall',
            name='dnf_model_recall',
            parameters=[{'trial_number': LaunchConfiguration('trial_number')}]
        ),
        Node(
            package='dnf_cognitive_architecture_extended',
            executable='output_node',
            name='output_node'
        )
    ])


#  from launch import LaunchDescription
# from launch_ros.actions import Node


# def generate_launch_description():
#     return LaunchDescription([
#         Node(
#             package='dnf_architecture_extended',
#             executable='input_matrix',
#             name='input_matrix'
#         ),
#         Node(
#             package='dnf_architecture_extended',
#             executable='dnf_model_recall',
#             name='dnf_model_recall'
#         ),
#         Node(
#             package='dnf_architecture_extended',
#             executable='output_node',
#             name='output_node'
#         )
#     ])
