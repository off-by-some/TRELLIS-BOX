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

# Check if container exists
if ! docker ps -a --filter "name=trellis-box" | grep -q trellis-box; then
    print_warning "No TRELLIS container found to stop."
    exit 0
fi

# Check if container is running
if ! docker ps --filter "name=trellis-box" --filter "status=running" | grep -q trellis-box; then
    print_status "TRELLIS container is not currently running."
    exit 0
fi

print_status "Stopping TRELLIS container..."
if docker stop trellis-box > /dev/null 2>&1; then
    print_success "TRELLIS container stopped successfully"
    print_status "Run './restart.sh' to start it again (keeps cached models)"
else
    print_error "Failed to stop TRELLIS container"
    exit 1
fi

