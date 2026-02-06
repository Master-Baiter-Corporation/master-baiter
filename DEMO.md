# Instructions pour la démo

## Partie 1 : Simulation sur Gazebo

### Prérequis

- ROS2 Humble
- Gazebo Harmonic
- ros_gz
- rviz2

Vous êtes légèrement livré à vous même pour ce qui est de l'installation de ROS2 et compagnie. Nous avons personnellement utilisé une [image docker](./Dockerfile) custom basée sur celle de ROS Humble mais je ne garantis pas la reproductibilité de la chose. En cas d'extrême urgence vous pouvez toujours vous référez à la [section concernant l'installation Docker](#utilisation-de-docker).

### Mise en place de l'environnement

```sh
git clone git@github.com:Master-Baiter-Corporation/master-baiter.git
cd master-baiter
git submodule update --init --recursive
```

### Compilation et lancement

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

### Utilisation de Docker

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
## Partie 2 : Utilisation avec le Turtlebot3

### Prérequis

- ROS2 Humble

Il est également nécessaire d'avoir mené à bien la connection de la Raspberry Pi avec la plateforme Respeaker (voir [guide de mise en place du respeaker](./respeaker/SETUP_RESPEAKER.md)).

### Mise en place de l'environnement

 - Sur votre machine, cloner ce repository puis copier les dossiers "bridge_odas_ai", "orchestrator" et "speech_recognition" dans votre espace de travail ROS2 (je présuppose que celui-ci est nommé "ros2_ws" et qu'il se situe dans votre dossier personnel ~)
```sh
git clone git@github.com:Master-Baiter-Corporation/master-baiter.git
cd master-baiter
cp -r bridge_odas_ai orchestrator speech_recognition ~/ros2_ws/src
```

### Compilation et lancement

 - Dans votre espace de travail ROS2, compiler les nœuds que vous venez de copier puis lancer les nœuds "bridge_odas_ai" et "speech_recognition" grâce au fichier de launch de ce-dernier :
```sh
cd ~/ros2_ws
colcon build
source ~/.bashrc
ros2 launch speech_recognition speech_recognition.launch.py
```

 - Dans un autre terminal, se placer dans l'espace de travail ROS2 et lancer le nœud "orchestrator" :
```sh
cd ~/ros2_ws
source install/setup.bash
ros2 run orchestrator listener
```

 - Si vous venez de réaliser l'initialisation de la Raspberry Pi grâce au guide et que vous ne l'avez pas éteinte depuis, il n'y a plus rien à faire. \
   Sinon, basculer sur la Raspberry Pi et suivre les instructions présentées dans le paragraphe "Lorsque vous revenez ensuite sur la Raspberry Pi" de la section "**Au cas où**" du [guide de mise en place du respeaker](./respeaker/SETUP_RESPEAKER.md).
 
 - Ça y est, le démarrage de tous les nœuds ROS2 indispensables au bon fonctionnement de notre système est finalisé !

### Utilisation en pratique

Maintenant que tous les nœuds sont lancés, vous pouvez commecer à parler à côté du Turtlebot3 pour lui donner des instructions. Lorsque vous dites "avance", il se met à avancer tout droit à une vitesse de 0,1 m/s pendant 3,5 secondes. Lorsque vous dites "gauche" ou "droite", il s'oriente alors dans la direction demandée en effectuant un quart de tour. Ensuite, à tout moment, si vous souhaitez immobiliser le robot pendant qu'il effectue une action, vous n'avez qu'à dire "recule".

Seul cas particulier, la commande "lucien", dernière instruction reconnue par le robot (son prénom), nécessite d'effectuer une action en amont. En effet, lorsque vous l'appelez, le robot se dirige vers la direction qui lui a été donnée sur le topic ROS2 "/baiter_goal_pos". Par exemple, dans un autre terminal sur votre machine où tournent les 3 nœuds "bridge_odas_ai", "orchestrator" et "speech_recognition", vous pouvez lancer la commande `ros2 topic pub --once /baiter_goal_pos geometry_msgs/Point "{x: -1.0, y: 0.0, z: 0.0}"`. Celle-ci va renseigner la position à atteindre pour le robot s'il entend son prénom, dans l'exemple elle se trouve tout droit derrière lui. Vous n'avez alors plus qu'à dire "lucien" à côté du robot pour qu'il se dirige vers la position renseignée précédemment.
