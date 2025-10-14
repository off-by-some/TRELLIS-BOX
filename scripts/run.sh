#!/bin/bash

set -e  # Exit on any error
export DOCKER_BUILDKIT=1

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

# Function to start the container
start_container() {
    print_status "Starting TRELLIS container..."

    # Create persistent cache directories
    mkdir -p ~/.cache/trellis-box
    mkdir -p ~/.cache/rembg

    # Start the container
    docker run --gpus all \
        -it \
        -p 7860:7860 \
        --name trellis-box \
        -v ~/.cache/trellis-box:/root/.cache \
        -v ~/.cache/rembg:/root/.u2net \
        -v $(pwd)/outputs:/tmp/Trellis-demo \
        trellis-box
}

# Main execution
main() {
    echo "🚀 TRELLIS Docker Launcher"
    echo "========================="

    # Check GPU memory first
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
