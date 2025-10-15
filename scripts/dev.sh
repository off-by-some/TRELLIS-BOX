#!/bin/bash

# Quick development mode launcher
# Directly uses docker compose without all the checks

set -e
export DOCKER_BUILDKIT=1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

echo "🔥 TRELLIS Development Mode"
echo "============================"
echo ""

# Detect docker compose command
COMPOSE_CMD=""
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
    print_status "Using docker-compose (standalone)"
elif docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
    print_status "Using docker compose (plugin)"
else
    print_error "Neither 'docker-compose' nor 'docker compose' found"
    print_error "Install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    print_status "No .env file found (using defaults)"
fi

print_status "Starting development mode with hot reloading..."
print_status "Files mounted:"
print_status "  - app.py (hot-reloadable)"
print_status "  - webui/ (hot-reloadable)"
print_status "  - docs/ (hot-reloadable)"
print_status "  - assets/ (hot-reloadable)"
echo ""
print_status "Press Ctrl+C to stop"
echo ""

# Stop existing containers
$COMPOSE_CMD down > /dev/null 2>&1

# Start in development mode
$COMPOSE_CMD -f docker-compose.yml -f docker-compose.dev.yml up --build

print_success "Development container stopped"

