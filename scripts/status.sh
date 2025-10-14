#!/bin/bash

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

echo "🚀 TRELLIS Docker Status"
echo "========================"
echo "Includes TRELLIS models and rembg background removal models"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running or accessible"
    exit 1
fi

# Check container status
if docker ps -a --filter "name=trellis-box" | grep -q trellis-box; then
    if docker ps --filter "name=trellis-box" --filter "status=running" | grep -q trellis-box; then
        print_success "TRELLIS container is RUNNING"
        echo "  🌐 Web interface: http://localhost:7860"
        echo "  🐳 Container name: trellis-box"
    else
        print_warning "TRELLIS container exists but is STOPPED"
        echo "  Run './restart.sh' to start it again"
    fi
else
    print_status "No TRELLIS container found"
    echo "  Run './run.sh' to create and start one"
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

# TRELLIS cache directory check
if [ -d ~/.cache/trellis-box ]; then
    local cache_size=$(du -sh ~/.cache/trellis-box 2>/dev/null | cut -f1)
    print_success "TRELLIS cache: ~/.cache/trellis-box (${cache_size})"
else
    print_status "TRELLIS cache directory not created yet"
fi

# rembg cache directory check
if [ -d ~/.cache/rembg ]; then
    local rembg_size=$(du -sh ~/.cache/rembg 2>/dev/null | cut -f1)
    local rembg_files=$(find ~/.cache/rembg -type f | wc -l)
    print_success "rembg cache: ~/.cache/rembg (${rembg_size}, ${rembg_files} files)"
else
    print_status "rembg cache directory not created yet"
fi

echo ""

# Outputs directory check
if [ -d "./outputs" ]; then
    local output_count=$(find ./outputs -type f | wc -l)
    print_success "Outputs directory exists with ${output_count} files"
else
    print_status "Outputs directory not created yet"
fi
