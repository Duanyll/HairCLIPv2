FROM pcloud/gdown:latest as downloader

RUN gdown 1g8S81ZybmrF86OjvjLYJzx-wx83ZOiIw
RUN gdown 1OG6t7q4PpHOoYNdP-ipoxuqYbfMSgPta
RUN gdown 1c-SgUUQj0X1mIl-W-_2sMboI2QS7GzfK
RUN gdown 1sa732uBfX1739MFsvtRCKWCN54zYyltC
RUN gdown 1qk0ZIfA1VmrFUzDJ0g8mK8nx0WtF-5sY

FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

RUN rm /etc/apt/sources.list.d/cuda.list \
    && sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list \
    && sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai apt-get install -y git ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    ftfy regex tqdm matplotlib jupyter ipykernel opencv-python scikit-image kornia==0.6.7 face-alignment==1.3.5 dlib==19.22.1 \
    ninja gradio redis

RUN pip install git+https://github.com/openai/CLIP.git

COPY . /workspace
COPY --from=downloader /data/* /workspace/pretrained_models/

CMD python server.py

EXPOSE 7860
