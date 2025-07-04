from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    trial_number = LaunchConfiguration('trial_number')

    # Learning phase nodes
    learning_input_node = Node(
        package='dnf_cognitive_architecture_extended',
        executable='input_matrix',
        name='input_matrix_learning'
    )

    learning_dnf_node = Node(
        package='dnf_cognitive_architecture_extended',
        executable='dnf_model_learning',
        name='dnf_model_learning'
    )

    # Recall phase (launches 3 trials one after the other)
    recall_nodes = []
    for i in range(1, 4):
        recall_nodes.extend([
            Node(
                package='dnf_cognitive_architecture_extended',
                executable='input_matrix',
                name=f'input_matrix_recall_{i}'
            ),
            Node(
                package='dnf_cognitive_architecture_extended',
                executable='dnf_model_recall',
                name=f'dnf_model_recall_{i}',
                parameters=[{
                    'trial_number': LaunchConfiguration('trial_number')
                }],
            ),
            Node(
                package='dnf_cognitive_architecture_extended',
                executable='output_node',
                name=f'output_node_{i}'
            ),
            TimerAction(period=2.0, actions=[])  # Small delay between trials
        ])

    # Only launch recall after learning_dnf_node exits
    return LaunchDescription([
        DeclareLaunchArgument(
            'trial_number',
            default_value='1',
            description='Trial number for the recall mode'
        ),

        learning_input_node,
        learning_dnf_node,

        RegisterEventHandler(
            OnProcessExit(
                target_action=learning_dnf_node,
                on_exit=recall_nodes
            )
        )
    ])
