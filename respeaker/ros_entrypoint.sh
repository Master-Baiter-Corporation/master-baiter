#!/bin/bash
set -e

# Source l'environnement ROS2 global
source "/opt/ros/humble/setup.bash"

# Source le workspace ROS2 nouvellement créé (s'il a été compilé avec succès)
if [ -f "/root/ros2_ws/install/setup.bash" ]; then
  source "/root/ros2_ws/install/setup.bash"
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
fi

# Source le workspace Turtlebot3 nouvellement créé (s'il a été compilé avec succès)
if [ -f "/root/turtlebot3_ws/install/setup.bash" ]; then
  source "/root/turtlebot3_ws/install/setup.bash"
  export LDS_MODEL=LDS-02
fi

# Exécute la commande passée au conteneur (par défaut "bash")
exec "$@"