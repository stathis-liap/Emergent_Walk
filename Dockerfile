# Emergent Walk — reference-free sampling-based MPC on the Go2.
#
# Planning runs entirely on the CPU via MuJoCo's threaded `rollout` module, so
# no CUDA/JAX is needed for control. We keep an NVIDIA CUDA base only so the
# nvidia-container runtime can provide hardware EGL for MuJoCo's renderer
# (video recording in simulate.py). To run on a machine without an NVIDIA GPU,
# set MUJOCO_GL=osmesa at runtime for software rendering, or EW_RECORD=0 to
# skip rendering entirely.
FROM nvidia/cuda:12.8.1-base-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# Python 3.11 + GL/EGL/OSMesa runtime libs for the MuJoCo renderer and OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        libgl1 \
        libegl1 \
        libgles2 \
        libglfw3 \
        libosmesa6 \
        ffmpeg \
        git \
        ca-certificates \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && python3 -m pip install --upgrade pip

# Python deps (install first for layer caching).
COPY requirements.txt /tmp/requirements.txt
RUN pip install --default-timeout=1000 -r /tmp/requirements.txt

# Robot assets. The project loads ../unitree_mujoco/unitree_robots/go2/scene.xml
# relative to its own directory, so the two repos must be siblings.
WORKDIR /workspace
RUN git clone --depth 1 https://github.com/unitreerobotics/unitree_mujoco.git

# Project source.
COPY . /workspace/Emergent_Walk_MJX
WORKDIR /workspace/Emergent_Walk_MJX

# Default: run the 30 s simulation. Override the command (or EW_* env vars) as
# needed, e.g. `docker run ... python simulate.py` after `-e EW_SIM_SECONDS=60`.
CMD ["python", "simulate.py"]
