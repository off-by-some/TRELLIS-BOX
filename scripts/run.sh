#!/bin/bash

set -e  # Exit on any error
export DOCKER_BUILDKIT=1

# =============================================================================
# Configuration - Matches docker-compose.yml and Dockerfile
# =============================================================================
# Load .env file if it exists
if [ -f ".env" ]; then
    set -a  # Export all variables
    source .env
    set +a
fi

# Default values (matches docker-compose.yml defaults)
CUDA_VERSION=${CUDA_VERSION:-12.3.2}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
APP_PORT=${APP_PORT:-8501}
HOST_PORT=${HOST_PORT:-8501}
APP_USER=${APP_USER:-appuser}
APP_UID=${APP_UID:-1000}
CACHE_DIR=${CACHE_DIR:-/home/appuser/.cache}
HF_CACHE_DIR=${HF_CACHE_DIR:-/home/appuser/.cache/huggingface}
REMBG_CACHE_DIR=${REMBG_CACHE_DIR:-/home/appuser/.u2net}
TRELLIS_OUTPUT_DIR=${TRELLIS_OUTPUT_DIR:-/tmp/Trellis-demo}
OUTPUTS_HOST_DIR=${OUTPUTS_HOST_DIR:-./outputs}

# Use bind mounts instead of named volumes for better compatibility
HOST_CACHE_DIR=${HOST_CACHE_DIR:-$HOME/.cache/trellis}
HOST_HF_CACHE_DIR=${HOST_HF_CACHE_DIR:-$HOME/.cache/huggingface}
HOST_REMBG_CACHE_DIR=${HOST_REMBG_CACHE_DIR:-$HOME/.cache/rembg}

# Streamlit configuration
STREAMLIT_SERVER_ADDRESS=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}
STREAMLIT_SERVER_HEADLESS=${STREAMLIT_SERVER_HEADLESS:-true}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to prompt yes/no
prompt_yes_no() {
    local prompt="$1"
    local default="${2:-n}"
    local response

    if [[ "$default" == "y" ]]; then
        echo -n "$prompt [Y/n]: "
    else
        echo -n "$prompt [y/N]: "
    fi

    read response
    response=${response,,}  # Convert to lowercase

    if [[ -z "$response" ]]; then
        response="$default"
    fi

    [[ "$response" == "y" || "$response" == "yes" ]]
}

# Function to check if container is running
is_container_running() {
    docker ps --filter "name=trellis-box" --filter "status=running" | grep -q trellis-box
}

# Function to check GPU memory
check_gpu_memory() {
    print_status "Checking GPU memory..."

    # Check if nvidia-smi is available
    if ! command -v nvidia-smi &> /dev/null; then
        print_warning "nvidia-smi not found. Skipping GPU memory check."
        return 0
    fi

    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
    TOTAL_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)

    echo "GPU Memory: ${FREE_MEM}MB free / ${TOTAL_MEM}MB total"

    if [ "$FREE_MEM" -lt 8000 ]; then
        print_warning "Less than 8GB GPU memory free (${FREE_MEM}MB)"
        print_warning "TRELLIS needs ~8-10GB during generation"

        if ! prompt_yes_no "Continue anyway?"; then
            print_status "Exiting. Free up GPU memory and try again."
            exit 1
        fi
    else
        print_success "Sufficient GPU memory available"
    fi
}

# Function to clean up old images
cleanup_old_images() {
    print_status "Cleaning up old TRELLIS images..."

    # Remove dangling images
    docker image prune -f > /dev/null 2>&1

    # Remove old trellis-box images (keep only the latest)
    local image_count=$(docker images trellis-box --format "table {{.Repository}}:{{.Tag}}" | grep -c trellis-box || true)

    if [ "$image_count" -gt 1 ]; then
        print_status "Found $image_count TRELLIS images. Keeping latest, removing others..."
        # Get all image IDs except the latest
        local old_images=$(docker images trellis-box --format "{{.ID}}" | tail -n +2)
        if [ -n "$old_images" ]; then
            echo "$old_images" | xargs docker rmi -f > /dev/null 2>&1
            print_success "Cleaned up old images"
        fi
    fi
}

# Function to stop running container
stop_container() {
    if is_container_running; then
        print_warning "TRELLIS container is already running"
        if prompt_yes_no "Stop the existing container?"; then
            print_status "Stopping container..."
            docker stop trellis-box > /dev/null 2>&1
            print_success "Container stopped"
        else
            print_status "Keeping existing container. Use './stop.sh' to stop it manually."
            exit 1
        fi
    fi
}

# Function to remove existing container
remove_container() {
    if docker ps -a --filter "name=trellis-box" | grep -q trellis-box; then
        print_status "Removing existing container..."
        docker rm trellis-box > /dev/null 2>&1
    fi
}

# Function to build the image
build_image() {
    print_status "Building TRELLIS Docker image..."
    print_status "This may take 10-15 minutes on first run..."

    if docker build -t trellis-box . ; then
        print_success "Image built successfully"
    else
        print_error "Failed to build image"
        exit 1
    fi
}

# Function to check GPU access
check_gpu_access() {
    print_status "Checking GPU access..."

    # Test if nvidia-docker is working with --gpus all
    if docker run --rm --gpus all nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
        print_success "GPU access with --gpus all: WORKING"
        return 0
    fi

    # Test if nvidia-docker is working with --runtime=nvidia
    if docker run --rm --runtime=nvidia nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
        print_warning "GPU access with --runtime=nvidia: WORKING (older Docker setup)"
        print_warning "Consider upgrading to Docker with --gpus support"
        return 0
    fi

    # Check if basic nvidia runtime works at all
    if docker run --rm nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
        print_warning "Basic NVIDIA runtime works, but GPU passthrough is not configured"
        print_error "Try: sudo apt-get install nvidia-docker2 && sudo systemctl restart docker"
    else
        print_error "NVIDIA Container Toolkit not properly installed or GPU not available"
        print_error "Install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        print_error "Or check that your GPU is properly installed: nvidia-smi"
    fi

    print_error "Cannot proceed without GPU access. TRELLIS requires CUDA."
    exit 1
}

# Function to start the container
start_container() {
    print_status "Starting TRELLIS container..."
    print_status "Image: trellis-box:latest"
    print_status "Port: ${HOST_PORT} -> ${APP_PORT}"
    print_status "GPU Memory: ${CUDA_VISIBLE_DEVICES:-all available}"
    print_status "================================"

    # Create host directories
    mkdir -p "$OUTPUTS_HOST_DIR"
    mkdir -p "$HOST_CACHE_DIR"
    mkdir -p "$HOST_HF_CACHE_DIR"
    mkdir -p "$HOST_REMBG_CACHE_DIR"

    # Use --gpus all (modern Docker) or --runtime=nvidia (legacy) based on what works
    if docker run --gpus all --rm nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
        GPU_FLAG="--gpus all"
        print_status "Using --gpus flag for GPU access"
    elif docker run --runtime=nvidia --rm nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
        GPU_FLAG="--runtime=nvidia"
        print_warning "Using --runtime=nvidia (consider upgrading Docker for --gpus support)"
    else
        print_error "GPU access test failed - cannot start container"
        print_error "Run './scripts/check_gpu.sh' to diagnose GPU issues"
        exit 1
    fi

    # Use bind mounts instead of named volumes for better compatibility
    docker run $GPU_FLAG \
        -it \
        --rm \
        -p "${HOST_PORT}:${APP_PORT}" \
        --name trellis-box \
        -v "${HOST_CACHE_DIR}:${CACHE_DIR}" \
        -v "${HOST_HF_CACHE_DIR}:${HF_CACHE_DIR}" \
        -v "${HOST_REMBG_CACHE_DIR}:${REMBG_CACHE_DIR}" \
        -v "$(pwd)/${OUTPUTS_HOST_DIR}:${TRELLIS_OUTPUT_DIR}" \
        -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        -e STREAMLIT_SERVER_PORT="${APP_PORT}" \
        -e STREAMLIT_SERVER_ADDRESS="${STREAMLIT_SERVER_ADDRESS}" \
        -e STREAMLIT_SERVER_HEADLESS="${STREAMLIT_SERVER_HEADLESS}" \
        trellis-box

    print_success "Container stopped"
}

# Main execution
main() {
    echo "🚀 TRELLIS Docker Launcher"
    echo "========================="

    # Check GPU access first
    check_gpu_access

    # Check GPU memory
    check_gpu_memory

    # Stop any running container
    stop_container

    # Remove existing container
    remove_container

    # Clean up old images
    cleanup_old_images

    # Always rebuild (since it copies local code)
    build_image

    # Start the container
    start_container
}

# Run main function
main "$@"
