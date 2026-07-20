# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04
ARG PYTHON_VERSION=3.10

ARG PYTORCH_VERSION=2.6.0
ARG TORCHVISION_VERSION=0.21.0
ARG PYTORCH_CUDA_INDEX=cu124
ARG FLASH_ATTN_VERSION=2.7.3
ARG FLASH_ATTN_WHEEL_URL=auto

ARG APP_USER=trellis
ARG APP_UID=1000
ARG APP_GID=1000
ARG APP_PORT=7860
ARG MAX_JOBS=8
ARG TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0"


FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS python-deps

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG PYTHON_VERSION
ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION
ARG PYTORCH_CUDA_INDEX
ARG FLASH_ATTN_VERSION
ARG FLASH_ATTN_WHEEL_URL
ARG MAX_JOBS
ARG TORCH_CUDA_ARCH_LIST

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    MAX_JOBS=${MAX_JOBS} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH}

WORKDIR /build/trellis2

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOF
set -eux
apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    git \
    git-lfs \
    libegl1 \
    libeigen3-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libjpeg-dev \
    libpng-dev \
    libx11-6 \
    libxext6 \
    ninja-build \
    pkg-config \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    build-essential
rm -rf /var/lib/apt/lists/*
EOF

RUN python${PYTHON_VERSION} -m venv "${VIRTUAL_ENV}"

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked <<'EOF'
set -eux
python -m pip install --upgrade pip setuptools wheel packaging
EOF

# README default: PyTorch 2.6.0 with CUDA 12.4.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked <<EOF
set -eux
python -m pip install \
    torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA_INDEX}
EOF

COPY requirements.docker.txt ./

# Mirrors `setup.sh --basic` without sudo/conda assumptions.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked <<'EOF'
set -eux
python -m pip install --requirement requirements.docker.txt
python -m pip uninstall -y pillow || true
python -m pip install --no-binary=:all: pillow-simd
EOF

# README default attention backend is flash-attn. Installing the release wheel
# directly avoids flash-attn's setup.py wheel-download path, which can fail
# under BuildKit cache mounts with "Invalid cross-device link".
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked <<EOF
set -eux
if [ "${FLASH_ATTN_WHEEL_URL}" = "auto" ]; then
    FLASH_ATTN_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
fi
python -m pip install "${FLASH_ATTN_WHEEL_URL}"
EOF

FROM python-deps AS builder

# Mirrors setup.sh extension flags:
#   --nvdiffrast --nvdiffrec --cumesh --flexgemm
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked <<'EOF'
set -eux
mkdir -p /tmp/extensions

git clone --depth=1 -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
python -m pip install /tmp/extensions/nvdiffrast --no-build-isolation

git clone --depth=1 -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec
python -m pip install /tmp/extensions/nvdiffrec --no-build-isolation

git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh
python -m pip install /tmp/extensions/CuMesh --no-build-isolation

git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM
python -m pip install /tmp/extensions/FlexGEMM --no-build-isolation

rm -rf /tmp/extensions
EOF

# O-Voxel is shipped in this repo and should be installed from local source.
COPY o-voxel ./o-voxel
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked <<'EOF'
set -eux
python -m pip install ./o-voxel --no-build-isolation --no-deps
EOF


FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG PYTHON_VERSION
ARG APP_USER
ARG APP_UID
ARG APP_GID
ARG APP_PORT

LABEL org.opencontainers.image.title="TRELLIS.2 Low-VRAM CUDA Runtime" \
      org.opencontainers.image.description="CUDA 12.4 runtime for TRELLIS.2 with low-VRAM flow block offload defaults." \
      org.opencontainers.image.licenses="MIT"

ENV APP_USER=${APP_USER} \
    APP_UID=${APP_UID} \
    APP_GID=${APP_GID}

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    HOME=/home/${APP_USER} \
    PATH=/opt/venv/bin:${PATH} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_MODULE_LOADING=LAZY \
    ATTN_BACKEND=flash_attn \
    TRELLIS2_FLOW_BLOCK_OFFLOAD=1 \
    TOKENIZERS_PARALLELISM=false \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    XDG_CACHE_HOME=/home/${APP_USER}/.cache \
    TRITON_CACHE_DIR=/home/${APP_USER}/.cache/triton \
    HF_HOME=/home/${APP_USER}/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/home/${APP_USER}/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/home/${APP_USER}/.cache/huggingface/transformers \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=${APP_PORT} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOF
set -eux
apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ffmpeg \
    git \
    git-lfs \
    gosu \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libjpeg-turbo8 \
    libpng16-16 \
    libsm6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-venv \
    tini
rm -rf /var/lib/apt/lists/*
if ! getent group "${APP_GID}" >/dev/null; then
    groupadd --gid ${APP_GID} ${APP_USER}
fi
if ! id --user ${APP_USER} >/dev/null 2>&1; then
    useradd --uid ${APP_UID} --gid ${APP_GID} --create-home --shell /bin/bash ${APP_USER}
fi
EOF

COPY --from=builder /opt/venv /opt/venv
COPY --chown=${APP_UID}:${APP_GID} . /app

RUN <<EOF
set -eux
mkdir -p \
    /app/tmp \
    /app/outputs \
    ${TRITON_CACHE_DIR} \
    /home/${APP_USER}/.cache/huggingface/hub \
    /home/${APP_USER}/.cache/huggingface/transformers
chown -R ${APP_UID}:${APP_GID} /app /home/${APP_USER}
EOF

RUN chmod +x /app/docker/entrypoint.sh

VOLUME /home/${APP_USER}/.cache/huggingface ${TRITON_CACHE_DIR} /app/tmp /app/outputs
EXPOSE ${APP_PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"GRADIO_SERVER_PORT\", \"7860\")}/', timeout=5)" || exit 1

ENTRYPOINT ["tini", "--", "/app/docker/entrypoint.sh"]
CMD ["python", "app.py"]
