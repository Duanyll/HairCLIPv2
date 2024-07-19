FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

RUN rm /etc/apt/sources.list.d/cuda.list \
    && sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list \
    && sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai apt-get install -y git gdb vim curl wget tmux zip cmake ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    ftfy regex tqdm matplotlib jupyter ipykernel opencv-python scikit-image kornia==0.6.7 face-alignment==1.3.5 dlib==19.22.1

RUN pip install git+https://github.com/openai/CLIP.git
