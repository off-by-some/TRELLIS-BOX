#!/bin/bash
# Build script for Trellis 3D Docker image
# 
# Usage:
#   ./scripts/build.sh                    # Build with default settings
#   ./scripts/build.sh --no-cache         # Force rebuild without cache
#
# To customize build arguments, edit the variables below or pass them directly:
#   CUDA_VERSION=12.2.0 ./scripts/build.sh

set -e  # Exit on error

# Default build arguments (can be overridden by environment variables)
CUDA_VERSION=${CUDA_VERSION:-12.4.1}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
APP_PORT=${APP_PORT:-8501}

# Image name and tag
IMAGE_NAME=${IMAGE_NAME:-trellis-box}
IMAGE_TAG=${IMAGE_TAG:-latest}

# Build arguments
BUILD_ARGS=(
    --build-arg CUDA_VERSION="${CUDA_VERSION}"
    --build-arg PYTHON_VERSION="${PYTHON_VERSION}"
    --build-arg APP_PORT="${APP_PORT}"
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
echo "Building Trellis 3D Docker Image"
echo "=========================================="
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "CUDA Version: ${CUDA_VERSION}"
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
echo "Build complete!"
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "=========================================="
echo ""
echo "To run the container:"
echo "  docker run --gpus all -p ${APP_PORT}:${APP_PORT} ${IMAGE_NAME}:${IMAGE_TAG}"