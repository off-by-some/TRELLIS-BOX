# syntax=docker/dockerfile:1.4
# Use CUDA 12.1 devel image for better compatibility with build requirements
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies and Python 3.10
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install poetry
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install poetry==1.8.3

# Configure poetry
RUN poetry config virtualenvs.create false && \
    poetry config installer.max-workers 10

# Copy dependency files (including poetry.lock if it exists)
COPY pyproject.toml poetry.lock* ./
COPY wheels/ ./wheels/

# Install Python dependencies with caching
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    poetry install --only main --no-interaction --no-ansi && \
    pip install --no-cache-dir kaolin==0.17.0 && \
    pip install --no-cache-dir wheels/*.whl

# Runtime stage - smaller image
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/usr/local/bin:$PATH"

WORKDIR /app

# Install only runtime dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY trellis/ ./trellis/
COPY extensions/ ./extensions/
COPY assets/ ./assets/
COPY app.py ./

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose Streamlit port
EXPOSE 8501

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8501/_stcore/health')" || exit 1

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
