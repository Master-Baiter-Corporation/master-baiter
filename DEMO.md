# Instructions pour la démo

## Prérequis

- ROS2 Humble
- Gazebo Harmonic
- ros_gz
- rviz2

Vous êtes légèrement livré à vous même pour ce qui est de l'installation de ROS2 et compagnie. Nous avons personnellement utilisé une [image docker](./Dockerfile) custom basée sur celle de ROS Humble mais je ne garantis pas la reproductibilité de la chose. En cas d'extrême urgence vous pouvez toujours vous référez à la [section concernant l'installation Docker](#utilisation-de-docker).

## Mise en place de l'environnement

```sh
git clone git@github.com:Master-Baiter-Corporation/master-baiter.git
cd master-baiter
git submodule update --init --recursive
```

## Compilation et lancement

```sh
cd .. # pour ne pas mélanger les dossiers générés avec la repo
colcon build
```

Les commandes suivantes sont à exécuter dans chaque nouveau shell :

```sh
source install/setup.bash
export TURTLEBOT3_MODEL="burger"
export GZ_SIM_RESOURCE_PATH="$GZ_SIM_RESOURCE_PATH:$(realpath ./master-baiter/gazebo_shit/)"
```

Lancer le monde turtlebot3 :

```sh
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

Lancer notre orchestrateur :

```sh
ros2 run orchestrator listener
```

Maintenant que la simulation tourne ainsi que notre orchestrateur alors il est possible d'envoyer des commandes (mockées) à notre Robot Lucien telles que :

```sh
ros2 topic pub --once /baiter_cmd std_msgs/String 'data: avance'
ros2 topic pub --once /baiter_cmd std_msgs/String 'data: recule'
ros2 topic pub --once /baiter_cmd std_msgs/String 'data: droite'
ros2 topic pub --once /baiter_cmd std_msgs/String 'data: gauche'
```

Mais aussi en l'appelant par son prénom et en lui ayant donné des coordonnées vers lesquelles se diriger. Les coordonnées sont relatives au robot, l'axe X allant face à lui et l'axe Y vers sa gauche. Donc par exemple (1, 0) se situe devant lui, (0, -2) se situe derrière lui et (-2, 2) se situe derrière à gauche.

```sh
ros2 topic pub --once /baiter_goal_pos geometry_msgs/Point "{x: 1.5, y: -1.5}"
ros2 topic pub --once /baiter_cmd std_msgs/String 'data: lucien'
```

## Utilisation de Docker

```sh
cd master-baiter
# Autoriser tous les utilisateurs et processus locaux à se connecter au serveur X
# Afin de pouvoir ouvrir les fenêtres de Gazebo et RVIZ sur la machine hôte
xhost +local:
# Construire l'image Docker
docker build -t baiter:latest .
# Script pour lancer un conteneur
./baiter-run.sh
# Une fois dans le conteneur
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
echo 'export TURTLEBOT3_MODEL="burger"' >> ~/.bashrc
echo 'export GZ_SIM_RESOURCE_PATH="$GZ_SIM_RESOURCE_PATH:/home/dev/app/gazebo_shit"' >> ~/.bashrc
source ~/.bashrc
colcon build
```
