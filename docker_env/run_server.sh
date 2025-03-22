#!/bin/bash
echo "================================================================"
HOST_CODE_DIR="$HOME/polar-face/polar-face-code"
HOST_DATA_DIR="/mnt/datassd/polar-face"

if [ -d "$HOST_CODE_DIR" ]
then
    echo "Using code under $HOST_CODE_DIR"
    echo "================================================================"
else
    echo "Code dir not exsist. Git clone first"
    echo "================================================================"
    exit
fi

NAME="pface"
DEVICE="device=0"
PORT="16000"

docker run -ti --ipc=host \
--name $NAME \
--gpus $DEVICE \
-e NVIDIA_DRIVER_CAPABILITIES=all \
-v $HOST_CODE_DIR:/root/code \
-v $HOST_DATA_DIR:/root/datassd \
-p $PORT:$PORT \
polar_face:latest
