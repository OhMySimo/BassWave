#!/bin/bash
CONTAINER_NAME="rocm_tf_ddsp_v2"
LOCAL_PATH="/media/simone/NVME/BassModelv30/ddsp"

# Remove existing container if it exists
sudo docker rm -f $CONTAINER_NAME >/dev/null 2>&1

sudo docker run -it \
  --name $CONTAINER_NAME \
  --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host \
  --shm-size=512m \
  --cpus=6.0 \
  --cpu-shares=512 \
  --memory=12g \
  -e TF_NUM_INTEROP_THREADS=2 \
  -e TF_NUM_INTRAOP_THREADS=4 \
  -e MALLOC_TRIM_THRESHOLD_=0 \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v "$LOCAL_PATH":/app \
  -v /media/simone/NVME:/media/simone/NVME \
  -w /app \
  basswave-training:rocm5.5-tf2.11-20260509 \
  /bin/bash
