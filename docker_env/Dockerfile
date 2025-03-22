FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
RUN DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
SHELL ["/bin/bash", "-c"] 

RUN apt update && apt install -y git nano libgl1-mesa-glx x11-apps
RUN conda init bash

WORKDIR /root
COPY install.sh /root/install.sh
RUN cd && chmod +x install.sh && ./install.sh


