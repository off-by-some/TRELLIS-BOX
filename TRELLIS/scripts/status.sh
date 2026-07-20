#!/bin/bash

# =============================================================================
# Configuration - Matches docker-compose.yml and other scripts
# =============================================================================
# Load .env file if it exists
if [ -f ".env" ]; then
    set -a  # Export all variables
    source .env
    set +a
fi

# Default values (matches docker-compose.yml defaults)
APP_PORT=${APP_PORT:-8501}
HOST_PORT=${HOST_PORT:-8501}
OUTPUTS_HOST_DIR=${OUTPUTS_HOST_DIR:-./outputs}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo "TRELLIS Docker Status"
echo "====================="
echo "TRELLIS models and rembg background removal"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running or accessible"
    exit 1
fi

# Check container status
if docker ps -a --filter "name=trellis-box" | grep -q trellis-box; then
    if docker ps --filter "name=trellis-box" --filter "status=running" | grep -q trellis-box; then
        print_success "TRELLIS container running"
        echo "  Web interface: http://localhost:${HOST_PORT}"
        echo "  Container: trellis-box"
    else
        print_warning "TRELLIS container exists but stopped"
        echo "  Start with: ./scripts/restart.sh"
    fi
else
    print_status "No TRELLIS container found"
    echo "  Create with: ./scripts/run.sh"
fi

echo ""

# Check image status
if docker images trellis-box | grep -q trellis-box; then
    local image_info=$(docker images trellis-box --format "table {{.Size}}\t{{.CreatedAt}}" | tail -n 1)
    print_success "TRELLIS image exists: $image_info"
else
    print_warning "TRELLIS image not found"
    echo "  Will be built automatically when running './run.sh'"
fi

echo ""

# GPU Memory check
if command -v nvidia-smi &> /dev/null; then
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
    TOTAL_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    echo "💾 GPU Memory: ${FREE_MEM}MB free / ${TOTAL_MEM}MB total"

    if [ "$FREE_MEM" -lt 8000 ]; then
        print_warning "Low GPU memory (TRELLIS needs ~8-10GB)"
    fi
else
    print_warning "nvidia-smi not found - cannot check GPU memory"
fi

echo ""

# Cache directories check (bind mounts)
if [ -d "$HOME/.cache/trellis" ]; then
    cache_size=$(du -sh "$HOME/.cache/trellis" 2>/dev/null | cut -f1)
    print_success "TRELLIS cache: $HOME/.cache/trellis (${cache_size})"
else
    print_status "TRELLIS cache directory not created"
fi

if [ -d "$HOME/.cache/huggingface" ]; then
    hf_size=$(du -sh "$HOME/.cache/huggingface" 2>/dev/null | cut -f1)
    print_success "Hugging Face cache: $HOME/.cache/huggingface (${hf_size})"
else
    print_status "Hugging Face cache directory not created"
fi

if [ -d "$HOME/.cache/rembg" ]; then
    rembg_size=$(du -sh "$HOME/.cache/rembg" 2>/dev/null | cut -f1)
    print_success "rembg cache: $HOME/.cache/rembg (${rembg_size})"
else
    print_status "rembg cache directory not created"
fi

echo ""

# Outputs directory check
if [ -d "$OUTPUTS_HOST_DIR" ]; then
    local output_count=$(find "$OUTPUTS_HOST_DIR" -type f | wc -l)
    print_success "Outputs directory exists: $OUTPUTS_HOST_DIR (${output_count} files)"
else
    print_status "Outputs directory not created yet: $OUTPUTS_HOST_DIR"
fi
