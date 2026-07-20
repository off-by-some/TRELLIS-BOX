#!/bin/bash
# Build script for Trellis 3D Docker image
#
# Usage:
#   ./scripts/build.sh                    # Build with default settings
#   ./scripts/build.sh --no-cache         # Force rebuild without cache
#
# To customize build arguments, edit the variables below or pass them directly:
#   CUDA_VERSION=12.2.2 ./scripts/build.sh

set -e  # Exit on error

# =============================================================================
# Configuration - Matches docker-compose.yml and Dockerfile
# =============================================================================
# Load .env file if it exists
if [ -f ".env" ]; then
    set -a  # Export all variables
    source .env
    set +a
fi

# Default build arguments (matches docker-compose.yml defaults)
CUDA_VERSION=${CUDA_VERSION:-12.3.2}
CUDNN_VERSION=${CUDNN_VERSION:-9}
UBUNTU_VERSION=${UBUNTU_VERSION:-22.04}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
POETRY_VERSION=${POETRY_VERSION:-1.8.3}
KAOLIN_VERSION=${KAOLIN_VERSION:-0.17.0}
KAOLIN_INDEX_URL=${KAOLIN_INDEX_URL:-https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html}
TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-7.0 7.5 8.0 8.6 8.9 9.0}
APP_USER=${APP_USER:-appuser}
APP_UID=${APP_UID:-1000}
APP_PORT=${APP_PORT:-8501}
STREAMLIT_SERVER_ADDRESS=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}
STREAMLIT_SERVER_HEADLESS=${STREAMLIT_SERVER_HEADLESS:-true}

# Image configuration
IMAGE_NAME=${IMAGE_NAME:-trellis-box}
IMAGE_TAG=${IMAGE_TAG:-latest}

# Build arguments - matches docker-compose.yml
BUILD_ARGS=(
    --build-arg CUDA_VERSION="${CUDA_VERSION}"
    --build-arg CUDNN_VERSION="${CUDNN_VERSION}"
    --build-arg UBUNTU_VERSION="${UBUNTU_VERSION}"
    --build-arg PYTHON_VERSION="${PYTHON_VERSION}"
    --build-arg POETRY_VERSION="${POETRY_VERSION}"
    --build-arg KAOLIN_VERSION="${KAOLIN_VERSION}"
    --build-arg KAOLIN_INDEX_URL="${KAOLIN_INDEX_URL}"
    --build-arg TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
    --build-arg APP_USER="${APP_USER}"
    --build-arg APP_UID="${APP_UID}"
    --build-arg APP_PORT="${APP_PORT}"
    --build-arg STREAMLIT_SERVER_ADDRESS="${STREAMLIT_SERVER_ADDRESS}"
    --build-arg STREAMLIT_SERVER_HEADLESS="${STREAMLIT_SERVER_HEADLESS}"
)

# Parse command line arguments
EXTRA_ARGS=()
for arg in "$@"; do
    case $arg in
        --no-cache)
            EXTRA_ARGS+=(--no-cache)
            shift
            ;;
        --progress=*)
            EXTRA_ARGS+=("$arg")
            shift
            ;;
        *)
            EXTRA_ARGS+=("$arg")
            shift
            ;;
    esac
done

echo "=========================================="
echo "Building TRELLIS Docker Image"
echo "=========================================="
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "CUDA Version: ${CUDA_VERSION}"
echo "cuDNN Version: ${CUDNN_VERSION}"
echo "Python Version: ${PYTHON_VERSION}"
echo "App Port: ${APP_PORT}"
echo "=========================================="

# Build the Docker image with BuildKit
DOCKER_BUILDKIT=1 docker build \
    "${BUILD_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    .

echo ""
echo "=========================================="
echo "Build completed successfully"
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  ./scripts/run.sh"
echo "  # Or with Docker Compose:"
echo "  docker-compose up"
echo "  # Access at: http://localhost:${APP_PORT}"