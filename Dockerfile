FROM osrf/ros:humble-desktop

# ---------------------------------------------------
# Install basic tools
# ---------------------------------------------------
RUN apt-get update && apt-get install -y \
    curl \
    lsb-release \
    gnupg \
    git \
    sudo \
    xz-utils \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------
# Add OSRF Gazebo repository keys & sources
# ---------------------------------------------------
RUN curl https://packages.osrfoundation.org/gazebo.gpg \
        --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
        https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/gazebo-stable.list && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
        https://packages.osrfoundation.org/gazebo/ubuntu-prerelease $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/gazebo-prerelease.list

# ---------------------------------------------------
# Install Gazebo Harmonic + RViz2
# ---------------------------------------------------
RUN apt-get update && apt-get install -y \
    gz-harmonic \
    ros-humble-ros-gz \
    ros-humble-cartographer \
    ros-humble-cartographer-ros \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-rviz2 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------
# Create non-root user "dev"
# ---------------------------------------------------
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    usermod -aG sudo $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME

# Switch to root temporarily to install system-wide Neovim
USER root

# ---------------------------------------------------
# Install latest Neovim from GitHub (tarball method)
# ---------------------------------------------------
RUN curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux-x86_64.tar.gz && \
    tar xzf nvim-linux-x86_64.tar.gz && \
    mv nvim-linux-x86_64 /usr/local/nvim && \
    ln -s /usr/local/nvim/bin/nvim /usr/local/bin/nvim && \
    rm nvim-linux-x86_64.tar.gz

# Switch back to dev user
USER $USERNAME
WORKDIR /home/$USERNAME

# ---------------------------------------------------
# Install LazyVim
# ---------------------------------------------------
RUN git clone https://github.com/LazyVim/starter ~/.config/nvim && \
    rm -rf ~/.config/nvim/.git

# ---------------------------------------------------
# Default command
# ---------------------------------------------------
CMD ["/bin/bash"]
