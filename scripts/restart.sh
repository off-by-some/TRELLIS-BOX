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

# Check if container exists
if ! docker ps -a --filter "name=trellis-box" | grep -q trellis-box; then
    print_error "No TRELLIS container found. Run './run.sh' first to create one."
    exit 1
fi

# Check if container is already running
if docker ps --filter "name=trellis-box" --filter "status=running" | grep -q trellis-box; then
    print_status "TRELLIS container is already running at http://localhost:7860"
    exit 0
fi

print_status "Restarting TRELLIS container..."
if docker start -ai trellis-box; then
    print_success "TRELLIS restarted successfully"
else
    print_error "Failed to restart TRELLIS container"
    exit 1
fi

