# Jetson Thor (JetPack 7, sm_110): NVIDIA's PyTorch container plus what RPent,
# LIBERO-PRO and MuJoCo need. PyTorch comes from the container; the venvs made by
# env/thor/setup.sh see it through --system-site-packages and never reinstall it.
#   sudo docker build -t rpent-thor env/thor
FROM nvcr.io/nvidia/pytorch:26.07-py3
RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        libegl1 libgl1 libglib2.0-0 libosmesa6 tmux git curl numactl && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir uv
ENV MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_ENABLE_HF_TRANSFER=0
