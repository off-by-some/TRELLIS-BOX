# syntax=docker/dockerfile:1.6

# =============================================================================
# Build Configuration Variables
# =============================================================================
# You can override these at build time using --build-arg
#
# Example:
#   docker build \
#     --build-arg CUDA_VERSION=12.3.2 \
#     --build-arg PYTHON_VERSION=3.11 \
#     --build-arg APP_PORT=8080 \
#     --build-arg CACHE_DIR=/data/.cache \
#     -t trellis-box:latest .
#
# Or use docker-compose with build args in docker-compose.yml
# See docker.env.example for all available configuration options
# =============================================================================
# CUDA and System
ARG CUDA_VERSION=12.3.2
ARG CUDNN_VERSION=9
ARG UBUNTU_VERSION=22.04
ARG PYTHON_VERSION=3.10

# Python Package Versions
ARG POETRY_VERSION=1.8.3
ARG TORCH_VERSION=2.4.0
ARG KAOLIN_VERSION=0.17.0
ARG KAOLIN_INDEX_URL=https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

# CUDA Architecture List for compiling PyTorch extensions
# Specify which GPU architectures to compile for (space-separated compute capabilities)
ARG TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"

# Application Configuration
ARG APP_USER=appuser
ARG APP_UID=1000
ARG APP_PORT=8501
ARG STREAMLIT_SERVER_ADDRESS=0.0.0.0
ARG STREAMLIT_SERVER_HEADLESS=true

# Cache Directories (inside container)
ARG CACHE_DIR=/home/appuser/.cache
ARG HF_CACHE_DIR=/home/appuser/.cache/huggingface
ARG REMBG_CACHE_DIR=/home/appuser/.u2net
ARG TRELLIS_OUTPUT_DIR=/tmp/Trellis-demo

# =============================================================================
# Builder Stage
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Re-declare ARGs needed in this stage
ARG PYTHON_VERSION
ARG POETRY_VERSION
ARG KAOLIN_VERSION
ARG KAOLIN_INDEX_URL
ARG TORCH_CUDA_ARCH_LIST

# Set working directory
WORKDIR /app

# Install system dependencies and Python
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOF
set -e
apt-get update
apt-get install -y --no-install-recommends \
    git \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    build-essential
rm -rf /var/lib/apt/lists/*
EOF

# Upgrade pip and install poetry
RUN --mount=type=cache,target=/root/.cache/pip <<EOF
set -e
pip install --upgrade pip setuptools wheel
pip install poetry==${POETRY_VERSION}
EOF

# Configure poetry
RUN <<EOF
set -e
poetry config virtualenvs.create false
poetry config installer.max-workers 10
EOF

# Copy dependency files (including poetry.lock if it exists)
COPY pyproject.toml poetry.lock* ./
COPY extensions/ ./extensions/

# Install application dependencies. We keep this separate from the other dependencies to
# avoid re-installing the same dependencies if kaolin fails to install.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry <<EOF
set -e
poetry install --only main --no-interaction --no-ansi
EOF

# Install Kaolin from NVIDIA's repository
RUN --mount=type=cache,target=/root/.cache/pip <<EOF
set -e
pip install --no-cache-dir \
    --find-links ${KAOLIN_INDEX_URL} \
    kaolin==${KAOLIN_VERSION}
EOF

# Install flash-attention
RUN --mount=type=cache,target=/root/.cache/pip <<EOF
set -e
if pip install --no-cache-dir flash-attn; then
    echo "Flash attention wheel installed successfully"
else
    echo "Flash attention wheel not available, attempting source build..."
    pip install --no-cache-dir flash-attention
fi
EOF

# Install diff-gaussian-rasterization (download wheel if not present locally)
RUN --mount=type=cache,target=/root/.cache/pip <<EOF
set -e
if [ -f "wheels/diff_gaussian_rasterization-*.whl" ]; then
    echo "Installing diff-gaussian-rasterization from local wheel..."
    pip install --no-cache-dir wheels/diff_gaussian_rasterization-*.whl
else
    echo "Local wheel not found, downloading from HuggingFace..."
    pip install --no-cache-dir \
        https://huggingface.co/spaces/JeffreyXiang/TRELLIS/resolve/main/wheels/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl?download=true
fi
EOF

# Build and install nvdiffrast from source (ensures CUDA compatibility)
# Set TORCH_CUDA_ARCH_LIST to compile for common GPU architectures
# Format: "compute_capability1 compute_capability2 ..."
# 7.0,7.5 = Volta/Turing (V100, RTX 2080)
# 8.0,8.6 = Ampere (A100, RTX 3090)
# 8.9 = Ada Lovelace (RTX 4090)
# 9.0 = Hopper (H100)
RUN --mount=type=cache,target=/root/.cache/pip <<EOF
set -e
cd extensions/nvdiffrast
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" pip install --no-cache-dir .
EOF

# =============================================================================
# Runtime Stage
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# Re-declare ARGs needed in runtime stage
ARG PYTHON_VERSION
ARG APP_USER
ARG APP_UID
ARG APP_PORT
ARG CACHE_DIR
ARG HF_CACHE_DIR
ARG REMBG_CACHE_DIR
ARG TRELLIS_OUTPUT_DIR

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/usr/local/bin:$PATH" \
    APP_PORT=${APP_PORT} \
    CACHE_DIR=${CACHE_DIR} \
    HF_HOME=${HF_CACHE_DIR} \
    HUGGINGFACE_HUB_CACHE=${HF_CACHE_DIR} \
    TRANSFORMERS_CACHE=${HF_CACHE_DIR} \
    U2NET_HOME=${REMBG_CACHE_DIR} \
    TRELLIS_OUTPUT_DIR=${TRELLIS_OUTPUT_DIR} \
    STREAMLIT_SERVER_ADDRESS=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0} \
    STREAMLIT_SERVER_HEADLESS=${STREAMLIT_SERVER_HEADLESS:-true}

WORKDIR /app

# Install only runtime dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOF
set -e
apt-get update
apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python3-pip \
    git \
    libgomp1
rm -rf /var/lib/apt/lists/*
EOF

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python${PYTHON_VERSION}/dist-packages /usr/local/lib/python${PYTHON_VERSION}/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY trellis/ ./trellis/
COPY extensions/ ./extensions/
COPY assets/ ./assets/
COPY app.py ./

# Create non-root user for security
RUN <<EOF
set -e
useradd -m -u ${APP_UID} -s /bin/bash ${APP_USER}
chown -R ${APP_USER}:${APP_USER} /app
mkdir -p ${CACHE_DIR} ${HF_CACHE_DIR} ${REMBG_CACHE_DIR} ${TRELLIS_OUTPUT_DIR}
chown -R ${APP_USER}:${APP_USER} ${CACHE_DIR} ${REMBG_CACHE_DIR} ${TRELLIS_OUTPUT_DIR}
EOF

USER ${APP_USER}

# Expose Streamlit port
EXPOSE ${APP_PORT}

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:${APP_PORT}/_stcore/health')" || exit 1

# Command to run the Streamlit app
CMD streamlit run app.py \
    --server.port=${APP_PORT} \
    --server.address=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0} \
    --server.headless=${STREAMLIT_SERVER_HEADLESS:-true}
