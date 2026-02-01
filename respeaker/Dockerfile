FROM ros:humble-ros-core as aptgetter

SHELL ["/bin/bash", "-c"]

# ---------------------------------------------------
# 0. Pre-configure Architecture for OpenCR
# ---------------------------------------------------
RUN dpkg --add-architecture armhf

# ---------------------------------------------------
# 1. Install system packages and basic tools
# ---------------------------------------------------
RUN apt-get update && apt-get install -y \
      curl \
      wget \
      unzip \
      bzip2 \
      git \
      sudo \
      nano \
      software-properties-common \
      build-essential \
      cmake \
      gcc \
      gfortran \
      texinfo \
      perl \
      libfftw3-dev \
      libconfig-dev \
      libasound2-dev \
      libpulse-dev \
      libgfortran-*-dev \
      libboost-system-dev \
      libudev-dev \
      libc6:armhf \
      python3-colcon-common-extensions \
      python3-rosdep \
      python3-pip \
      python3-argcomplete \
      ros-humble-ros-base \
      ros-humble-rmw-cyclonedds-cpp \
      ros-humble-sensor-msgs-py \
      ros-humble-hls-lfcd-lds-driver \
      ros-humble-turtlebot3-msgs \
      ros-humble-dynamixel-sdk \
      ros-humble-xacro \
      ros-humble-demo-nodes-cpp \
      ros-humble-demo-nodes-py && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------
# 2. Python dependencies
# ---------------------------------------------------
RUN pip3 install libconf

# ---------------------------------------------------
# 3. Setup ROS2 Workspace & Clone Repositories
# ---------------------------------------------------
WORKDIR /root

RUN mkdir -p /root/ros2_ws/src

WORKDIR /root/ros2_ws/src

RUN git clone https://github.com/introlab/audio_utils.git && \
    git clone https://github.com/introlab/odas_ros.git

WORKDIR /root/ros2_ws/src/audio_utils

RUN git submodule update --init --recursive

WORKDIR /root/ros2_ws/src/odas_ros

RUN git submodule update --init --recursive

COPY configuration.cfg /root/ros2_ws/src/odas_ros/odas_ros/config/configuration.cfg

# ---------------------------------------------------
# 4. Build the ROS2 Workspace
# ---------------------------------------------------
WORKDIR /root/ros2_ws

RUN source /opt/ros/humble/setup.bash && \
    rosdep init && \
    rosdep update && \
    colcon build --executor sequential

# ---------------------------------------------------
# 5. Setup Turtlebot3 Workspace & Clone Repositories
# ---------------------------------------------------
WORKDIR /root

RUN mkdir -p /root/turtlebot3_ws/src

WORKDIR /root/turtlebot3_ws/src

RUN git clone --depth 1 -b humble https://github.com/ROBOTIS-GIT/turtlebot3.git && \
    git clone --depth 1 -b humble https://github.com/ROBOTIS-GIT/ld08_driver.git && \
    git clone --depth 1 -b humble https://github.com/ROBOTIS-GIT/coin_d4_driver.git

WORKDIR /root/turtlebot3_ws/src/turtlebot3

RUN rm -r turtlebot3_cartographer turtlebot3_navigation2

# ---------------------------------------------------
# 6. Build the Turtlebot3 Workspace
# ---------------------------------------------------
WORKDIR /root/turtlebot3_ws

RUN source /opt/ros/humble/setup.bash && \
    source /root/ros2_ws/install/setup.bash && \
    colcon build --symlink-install --parallel-workers 1

# ---------------------------------------------------
# 7. Prepare OpenCR Firmware Tools
# ---------------------------------------------------
ENV OPENCR_PORT=/dev/ttyACM0
ENV OPENCR_MODEL=burger

WORKDIR /root

RUN wget https://github.com/ROBOTIS-GIT/OpenCR-Binaries/raw/master/turtlebot3/ROS2/latest/opencr_update.tar.bz2 && \
    tar -xvf opencr_update.tar.bz2 && \
    rm opencr_update.tar.bz2

RUN echo -e "#!/bin/bash\n" > /root/flash_opencr.sh && \
    echo "cd /root/opencr_update" >> /root/flash_opencr.sh && \
    echo "./update.sh \$OPENCR_PORT \$OPENCR_MODEL.opencr" >> /root/flash_opencr.sh && \
    chmod +x /root/flash_opencr.sh

# ---------------------------------------------------
# 8. Entrypoint & Default Command
# ---------------------------------------------------
RUN echo -e "#!/bin/bash\n" > /root/ros2_turtlebot_start.sh && \
    echo "export ROS_DOMAIN_ID=<id>" >> /root/ros2_turtlebot_start.sh && \
    echo "export TURTLEBOT3_MODEL=burger" >> /root/ros2_turtlebot_start.sh && \
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /root/ros2_turtlebot_start.sh && \
    echo -e "export LDS_MODEL=LDS-02\n" >> /root/ros2_turtlebot_start.sh && \
    echo "source /opt/ros/humble/setup.bash" >> /root/ros2_turtlebot_start.sh && \
    echo -e "source /root/turtlebot3_ws/install/setup.bash\n" >> /root/ros2_turtlebot_start.sh && \
    echo "exec ros2 launch turtlebot3_bringup robot.launch.py" >> /root/ros2_turtlebot_start.sh && \
    chmod +x /root/ros2_turtlebot_start.sh

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /root/ros2_ws/install/setup.bash" >> /root/.bashrc && \
    echo "source /root/turtlebot3_ws/install/setup.bash" >> /root/.bashrc && \
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /root/.bashrc && \
    echo "export LDS_MODEL=LDS-02" >> /root/.bashrc

COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
