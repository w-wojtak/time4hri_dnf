from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    # Declare trial_number parameter (not actually used in this code, each node gets its own)
    declare_trial_number_arg = DeclareLaunchArgument(
        'trial_number',
        default_value='1',
        description='Trial number for the recall mode'
    )

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

    # Create recall trial nodes
    def create_recall_nodes(trial):
        input_node = Node(
            package='dnf_cognitive_architecture_extended',
            executable='input_matrix',
            name=f'input_matrix_recall_{trial}'
        )
        dnf_node = Node(
            package='dnf_cognitive_architecture_extended',
            executable='dnf_model_recall',
            name=f'dnf_model_recall_{trial}',
            parameters=[{'trial_number': trial}]
        )
        output_node = Node(
            package='dnf_cognitive_architecture_extended',
            executable='output_node',
            name=f'output_node_{trial}'
        )
        return input_node, dnf_node, output_node

    # Generate nodes for each trial
    input1, dnf1, output1 = create_recall_nodes(1)
    input2, dnf2, output2 = create_recall_nodes(2)
    input3, dnf3, output3 = create_recall_nodes(3)

    # Event chaining
    recall_1 = [input1, dnf1, output1]
    recall_2 = [input2, dnf2, output2]
    recall_3 = [input3, dnf3, output3]

    return LaunchDescription([
        declare_trial_number_arg,
        learning_input_node,
        learning_dnf_node,

        # After learning ends, run recall trial 1
        RegisterEventHandler(
            OnProcessExit(
                target_action=learning_dnf_node,
                on_exit=recall_1
            )
        ),

        # After recall trial 1 ends, run recall trial 2
        RegisterEventHandler(
            OnProcessExit(
                target_action=dnf1,
                on_exit=recall_2
            )
        ),

        # After recall trial 2 ends, run recall trial 3
        RegisterEventHandler(
            OnProcessExit(
                target_action=dnf2,
                on_exit=recall_3
            )
        ),
    ])
