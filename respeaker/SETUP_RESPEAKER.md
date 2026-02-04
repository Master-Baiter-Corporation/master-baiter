# Guide pour faire fonctionner le capteur Respeaker "6-Mic circular Array"

> Par Élian DELMAS SI5 (IoT-CPS)

1. Télécharger "raspios" (version de mai 2021) [ICI](https://downloads.raspberrypi.com/raspios_arm64/images/raspios_arm64-2021-05-28/2021-05-07-raspios-buster-arm64.zip)

2. Flasher la carte SD de la Raspberry Pi pour y mettre l'OS précédemment téléchargé

3. Se connecter en SSH sur la Pi

4. Installer dkms (version 2.6.1) grâce au repository GitHub
```sh
cd ~/Desktop
git clone https://github.com/dkms-project/dkms.git
cd dkms
git reset --hard 08dd4e13aceae4f37120aee7d8d55c8e22987740
make install-debian
```

5. Lancer la commande `dkms` pour vérifier que l'installation s'est bien passée

6. Utiliser le repository de Mr LAVIROTTE pour installer les drivers de la plateforme
```sh
cd ~/Desktop
export GIT_SSL_NO_VERIFY=true
git clone https://ubinas.polytech.unice.fr:38443/platform/respeaker/mics_hat.git
```

7. Modifier le fichier 'install.sh' qui se trouve à la racine du répertoire 'mics_hat' avec `nano ~/mics_hat/install.sh` : corriger la ligne 18, puis, entre les lignes 20 et 21 du fichier, ajouter la ligne `git reset --hard c8d97904ceacf848346029f176c23cc74addb733` pour revenir à la version de mai 2021 du driver de la plateforme Respeaker. \
Une fois cela fait, le fichier devrait ressembler à ça :
```
18 | cd ~/Desktop
19 | git clone https://github.com/respeaker/seeed-voicecard.git
20 | cd seeed-voicecard
21 | git reset --hard c8d97904ceacf848346029f176c23cc74addb733
22 | sudo ./install.sh
```

8. Lancer les commandes :
```sh
cd ~/mics_hat/
sudo ./install.sh
```

9. Une fois l'installation terminée et la raspberry Pi redémarrée, lancer les commandes :
```sh
cd ~/mics_hat/
pip3 install -r requirements.txt
python3 interfaces/pixels_demo.py
```
Cela devrait faire clignoter les LED présentes sur la plateforme.

10. Lancer les commandes `arecord -L` et `aplay -L` pour vérifier la bonne installation des drivers \
Les résultats devraient ressembler à ça :
```
pi@raspberrypi:~ $ arecord -L
null
    Discard all samples (playback) or generate zero samples (capture)
default
ac108
dmixer
ac101
sysdefault:CARD=seeed8micvoicec
    seeed-8mic-voicecard,
    Default Audio Device
dmix:CARD=seeed8micvoicec,DEV=0
    seeed-8mic-voicecard,
    Direct sample mixing device
dsnoop:CARD=seeed8micvoicec,DEV=0
    seeed-8mic-voicecard,
    Direct sample snooping device
hw:CARD=seeed8micvoicec,DEV=0
    seeed-8mic-voicecard,
    Direct hardware device without any conversions
plughw:CARD=seeed8micvoicec,DEV=0
    seeed-8mic-voicecard,
    Hardware device with all software conversions 
```
et ça :
```
pi@raspberrypi:~ $ aplay -L
null
    Discard all samples (playback) or generate zero samples (capture)
default
ac108
dmixer
ac101
sysdefault:CARD=ALSA
    bcm2835 ALSA, bcm2835 ALSA
    Default Audio Device
dmix:CARD=ALSA,DEV=0
    bcm2835 ALSA, bcm2835 ALSA
    Direct sample mixing device
dmix:CARD=ALSA,DEV=1
    bcm2835 ALSA, bcm2835 IEC958/HDMI
    Direct sample mixing device
dsnoop:CARD=ALSA,DEV=0
    bcm2835 ALSA, bcm2835 ALSA
    Direct sample snooping device
dsnoop:CARD=ALSA,DEV=1
    bcm2835 ALSA, bcm2835 IEC958/HDMI
    Direct sample snooping device
hw:CARD=ALSA,DEV=0
    bcm2835 ALSA, bcm2835 ALSA
    Direct hardware device without any conversions
hw:CARD=ALSA,DEV=1
    bcm2835 ALSA, bcm2835 IEC958/HDMI
    Direct hardware device without any conversions
plughw:CARD=ALSA,DEV=0
    bcm2835 ALSA, bcm2835 ALSA
    Hardware device with all software conversions
plughw:CARD=ALSA,DEV=1
    bcm2835 ALSA, bcm2835 IEC958/HDMI
    Hardware device with all software conversions
sysdefault:CARD=seeed8micvoicec
    seeed-8mic-voicecard,
    Default Audio Device
dmix:CARD=seeed8micvoicec,DEV=0
    seeed-8mic-voicecard,
    Direct sample mixing device
dsnoop:CARD=seeed8micvoicec,DEV=0
    seeed-8mic-voicecard,
    Direct sample snooping device
hw:CARD=seeed8micvoicec,DEV=0
    seeed-8mic-voicecard,
    Direct hardware device without any conversions
plughw:CARD=seeed8micvoicec,DEV=0
    seeed-8mic-voicecard,
    Hardware device with all software conversions
```

11. Tester que l'enregistrement se passe bien avec la commande `arecord -Dac108 -f S32_LE -r 16000 -c 8 -d 3 ~/a.wav` qui va démarrer un enregistrement audio via les micros de la plateforme respeaker d'une durée de 3 secondes (VS Code peut lire les fichiers .wav donc vous pouvez l'écouter juste après pour vérifier le bon déroulé de l'enregistrement)

12. Modifier le fichier '/etc/apt/sources.list' avec la commande `sudo nano /etc/apt/sources.list` pour ajouter la ligne `deb https://download.docker.com/linux/debian/ buster main contrib non-free` en début de fichier

13. Exécuter ces commandes pour préparer l'installation de Docker :
```sh
# Add Docker's official GPG key:
sudo apt update
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
```

14. Vérifier le contenu du fichier '/etc/apt/sources.list.d/docker.list' avec la commande `cat /etc/apt/sources.list.d/docker.list` \
Il doit contenir la ligne : `deb [arch=arm64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian buster stable`

15. Installer Docker puis vérifier qu'il fonctionne correctement avec les commandes :
```sh
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl status docker
sudo systemctl start docker
sudo docker run hello-world
```

16. Modifier la taille de la swap du Raspberry Pi (nécessaire pour la compilation du package ROS plus tard) : 
    * Désactiver la swap actuelle : `sudo dphys-swapfile swapoff`
    * Modifier la taille dans le fichier '/etc/dphys-swapfile' : `sudo nano /etc/dphys-swapfile` pour changer la valeur de la variable `CONF_SWAPSIZE` à `4096` et décommenter la variable `CONF_MAXSWAP` pour mettre sa valeur à `4096`
    * Recréer le fichier d'échange : `sudo dphys-swapfile setup`
    * Activer notre nouvelle swap : `sudo dphys-swapfile swapon`

17. Copier les fichiers "Dockerfile", "configuration.cfg" et "ros_entrypoint.sh" donnés dans un même dossier sur la Raspberry Pi puis se placer dans celui-ci

18. Construire l'image Docker : `docker build -t ros:aptgetter .` \
**Attention cela prend au moins une trentaine de minutes (partez faire autre chose en attendant)**

19. Créer le conteneur Docker : `docker run -d -it --name mon_conteneur --device /dev/snd --device /dev/ttyACM0 --net=host --privileged ros:aptgetter`

20. Ouvrir un terminal sur le conteneur en cours d'exécution (plusieurs terminaux sur le même conteneur peuvent être ouverts grâce à cette méthode) : `docker exec -it mon_conteneur /bin/bash`

21. Modifier les fichiers '/root/ros2_ws/src/odas_ros/odas_ros/scripts/odas_visualization_node.py' et '/root/ros2_ws/install/odas_ros/lib/odas_ros/odas_visualization_node.py' en remplaçant dans les deux la ligne n°13 (qui contient originallement "import sensor_msgs.point_cloud2 as pcl2  # type: ignore") par "from sensor_msgs_py import point_cloud2 as pcl2" :
`nano /root/ros2_ws/src/odas_ros/odas_ros/scripts/odas_visualization_node.py` \
puis \
`nano /root/ros2_ws/install/odas_ros/lib/odas_ros/odas_visualization_node.py`

22. Exécuter le fichier "flash_opencr.sh" qui se trouve dans le dossier "/root" (qui est aussi le dossier ~) :
```sh
cd /root/
./flash_opencr.sh
```

23. Se placer dans le dossier "/root/ros2_ws/" et lancer le nœud ROS ODAS :
```sh
cd /root/ros2_ws/
ros2 launch odas_ros odas.launch.xml configuration_path:=$PWD/src/odas_ros/odas_ros/config/configuration.cfg rviz:=false visualization:=true
```

24. Vérifier que le nœud ROS tourne bien en lançant les commandes suivantes dans un autre terminal (voir étape 20. pour ouvrir un autre terminal sur le même conteneur) :
```sh
ros2 node list
ros2 topic list
```
Les résultats devraient ressembler à ça :
```
root@respeaker:~/ros2_ws# ros2 node list
/odas_server_node
/odas_visualization_node
```
et
```
root@respeaker:~/ros2_ws# ros2 topic list
/parameter_events
/rosout
/ssl
/ssl_pcl2
/sss
/sst
/sst_poses
```

25. Lancer la commande `ros2 topic echo /ssl` pour écouter le topic sur lequel est publiée la Localisation de Sources Sonores (SSL - Sound Source Localization) qui détecte les sources sonores potentielles sur une sphère unitaire autour des microphones. Cela fourni des données sur sa position (x,y,z) et son énergie e.\
Le résultat devrait ressembler à ça :
```
root@respeaker:~/ros2_ws# ros2 topic echo /ssl
header:
  stamp:
    sec: 1767117697
    nanosec: 789954432
  frame_id: odas
sources:
- x: 0.827
  y: 0.381
  z: 0.413
  e: 0.205
- x: 0.266
  y: 0.574
  z: 0.774
  e: 0.149
- x: 0.647
  y: 0.681
  z: 0.342
  e: 0.084
- x: 0.386
  y: 0.196
  z: 0.902
  e: 0.08
---
```

26. Dans un autre terminal, effectuer l'étape 20. pour ouvrir un second terminal sur le conteneur et exécuter le fichier "ros2_turtlebot_start.sh" qui se trouve dans le dossier "/root" (qui est aussi le dossier ~) :
```sh
cd /root/
./ros2_turtlebot_start.sh
```

27. Et là, comme m'a dit Gemini suite à notre très longue conversation :
 > "Bravo pour votre persévérance ! Vous avez maintenant un système de localisation sonore fonctionnel sur votre Raspberry Pi."

### Au cas où

 - Lorsque vous avez terminé et que vous souhaitez éteindre la Raspberry Pi :
    1. Quitter le terminal du conteneur Docker (et revenir à un terminal de la Raspberry Pi) : `exit`
    2. Stopper le conteneur Docker avant d'éteindre la Pi : `docker stop mon_conteneur`
    3. Éteindre la Pi avec la commande `sudo shutdown now`

 - Lorsque vous revenez ensuite sur la Raspberry Pi :
    1. Se replacer dans le dossier où se trouvent les fichiers "Dockerfile", "configuration.cfg" et "ros_entrypoint.sh"
    2. Relancer le conteneur une fois que l'on revient sur la Pi : `docker start mon_conteneur`
    3. Effectuer de nouveau les étapes **20.**, **23.** et **26.** de ce guide pour remettre en fonctionnement ODAS et les packages ROBOTIS (permettant les mouvements du robot)

 - Sinon, pour supprimer le conteneur qui a été stoppé : `docker rm mon_conteneur`
