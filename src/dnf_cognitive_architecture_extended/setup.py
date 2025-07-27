from setuptools import setup

package_name = 'dnf_cognitive_architecture_extended'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/dnf_recall_launch.py']),
        ('share/' + package_name + '/launch', ['launch/dnf_learn_launch.py']),
        ('share/' + package_name + '/launch',
         ['launch/dnf_recall_basic_launch.py']),
        ('share/' + package_name + '/launch',
         ['launch/dnf_recall_error_launch.py']),
        ('share/' + package_name + '/launch',
         ['launch/dnf_learn_basic_launch.py']),
        ('share/' + package_name + '/launch',
         ['launch/dnf_experiment_launch.py']),
        ('share/' + package_name + '/launch',
         ['launch/dnf_experiment_basic_launch.py']),
        # Add any other launch files as needed
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'numpy',
        'matplotlib'
    ],
    zip_safe=True,
    maintainer='wwojtak',
    maintainer_email='w.wojtak@gmail.com',
    description='Test package for ROS2 Python build',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "input_matrix = dnf_cognitive_architecture_extended.input_matrix:main",
            "input_matrix_basic = dnf_cognitive_architecture_extended.input_matrix_basic:main",
            "input_matrix_basic_speech = dnf_cognitive_architecture_extended.input_matrix_basic_speech:main",
            "input_matrix_error = dnf_cognitive_architecture_extended.input_matrix_error:main",
            "dnf_model_learning = dnf_cognitive_architecture_extended.dnf_model_learning:main",
            "output_node = dnf_cognitive_architecture_extended.output_node:main",
            "dnf_model_recall = dnf_cognitive_architecture_extended.dnf_model_recall:main",
            "dnf_model_learning_basic = dnf_cognitive_architecture_extended.dnf_model_learning_basic:main",
            "dnf_model_recall_basic = dnf_cognitive_architecture_extended.dnf_model_recall_basic:main",
            "dnf_model_recall_with_error = dnf_cognitive_architecture_extended.dnf_model_recall_with_error:main",
            'udp_listener_node = dnf_cognitive_architecture_extended.udp_listener_node:main',
            'udp_response_sender_node = dnf_cognitive_architecture_extended.udp_response_sender_node:main'
        ],
    },
)
