# syntax=docker/dockerfile:1.4

# =============================================================================
# Build Configuration Variables
# =============================================================================
# You can override these at build time using --build-arg
# 
# Example:
#   docker build \
#     --build-arg CUDA_VERSION=12.2.0 \
#     --build-arg PYTHON_VERSION=3.11 \
#     --build-arg APP_PORT=8080 \
#     --build-arg CACHE_DIR=/data/.cache \
#     -t trellis-box:latest .
#
# Or use docker-compose with build args in docker-compose.yml
# See docker.env.example for all available configuration options
# =============================================================================
# CUDA and System
ARG CUDA_VERSION=12.1.0
ARG CUDNN_VERSION=8
ARG UBUNTU_VERSION=22.04
ARG PYTHON_VERSION=3.10

# Python Package Versions
ARG POETRY_VERSION=1.8.3
ARG TORCH_VERSION=2.4.0
ARG KAOLIN_VERSION=0.17.0
ARG KAOLIN_INDEX_URL=https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

# Application Configuration
ARG APP_USER=appuser
ARG APP_UID=1000
ARG APP_PORT=8501

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

# Set working directory
WORKDIR /app

# Install system dependencies and Python
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install poetry
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install poetry==${POETRY_VERSION}

# Configure poetry
RUN poetry config virtualenvs.create false && \
    poetry config installer.max-workers 10

# Copy dependency files (including poetry.lock if it exists)
COPY pyproject.toml poetry.lock* ./
COPY wheels/ ./wheels/
COPY extensions/ ./extensions/

# Install application dependencies. We keep this separate from the other dependencies to 
# avoid re-installing the same dependencies if wheels or kaolin fail to install.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    poetry install --only main --no-interaction --no-ansi

# Install Kaolin from NVIDIA's repository
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
        --find-links ${KAOLIN_INDEX_URL} \
        kaolin==${KAOLIN_VERSION}

# Install custom wheels (diff_gaussian_rasterization, etc.)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir wheels/*.whl

# Build and install nvdiffrast from source (ensures CUDA compatibility)
RUN --mount=type=cache,target=/root/.cache/pip \
    cd extensions/nvdiffrast && \
    pip install --no-cache-dir .

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
    TRELLIS_OUTPUT_DIR=${TRELLIS_OUTPUT_DIR}

WORKDIR /app

# Install only runtime dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python3-pip \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python${PYTHON_VERSION}/dist-packages /usr/local/lib/python${PYTHON_VERSION}/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY trellis/ ./trellis/
COPY extensions/ ./extensions/
COPY assets/ ./assets/
COPY app.py ./

# Create non-root user for security
RUN useradd -m -u ${APP_UID} -s /bin/bash ${APP_USER} && \
    chown -R ${APP_USER}:${APP_USER} /app && \
    mkdir -p ${CACHE_DIR} ${HF_CACHE_DIR} ${REMBG_CACHE_DIR} ${TRELLIS_OUTPUT_DIR} && \
    chown -R ${APP_USER}:${APP_USER} ${CACHE_DIR} ${REMBG_CACHE_DIR} ${TRELLIS_OUTPUT_DIR}

USER ${APP_USER}

# Expose Streamlit port
EXPOSE ${APP_PORT}

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:${APP_PORT}/_stcore/health')" || exit 1

# Command to run the Streamlit app
CMD streamlit run app.py \
    --server.port=${APP_PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true
