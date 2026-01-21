#!/bin/bash

IMAGE="baiter"
CONTAINER_NAME="baiter_container"

# Check if a container already exists
EXISTING=$(docker ps -a --filter "name=$CONTAINER_NAME" --format "{{.ID}}")

if [ -n "$EXISTING" ]; then
    STATE=$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME")

    if [ "$STATE" = "exited" ]; then
        echo "[*] Starting existing container..."
        docker start "$CONTAINER_NAME"
        docker attach "$CONTAINER_NAME"
        exit $?
    elif [ "$STATE" = "running" ]; then
        echo "[*] Container already running. Opening a new terminal..."
        docker exec -it "$CONTAINER_NAME" /bin/bash
        exit $?
    fi
fi

echo "[*] No existing container, creating a new one..."

docker run -it \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --device /dev/dri \
  --device /dev/nvidia0 \
  --device /dev/nvidiactl \
  --device /dev/nvidia-modeset \
  --device /dev/nvidia-uvm \
  --device /dev/nvidia-uvm-tools \
  --env="__NV_PRIME_RENDER_OFFLOAD=1" \
  --env="__VK_LAYER_NV_optimus=NVIDIA_only" \
  --env="__GLX_VENDOR_LIBRARY_NAME=nvidia" \
  --env="DISPLAY=$DISPLAY" \
  -e WAYLAND_DISPLAY=wayland-0 \
  -p 2222:22 \
  --mount type=bind,source=../master-baiter,target=/home/dev/app \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -e XDG_RUNTIME_DIR=/tmp \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /run/user/$UID/wayland-0:/tmp/wayland-0 \
  "$IMAGE"
