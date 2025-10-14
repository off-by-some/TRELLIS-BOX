#!/bin/bash

# TRELLIS-BOX Docker Hub Publishing Script
# Builds and publishes the Docker image to Docker Hub

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_status() {
    echo -e "${BLUE}➜ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Get Docker Hub username
if [ -z "$DOCKER_USERNAME" ]; then
    print_status "Enter your Docker Hub username:"
    read DOCKER_USERNAME
fi

if [ -z "$DOCKER_USERNAME" ]; then
    print_error "Docker Hub username is required"
    exit 1
fi

# Get version tag (default to 'latest')
VERSION="${1:-latest}"
print_status "Using version tag: $VERSION"

# Image names
LOCAL_IMAGE="trellis-box"
REMOTE_IMAGE="$DOCKER_USERNAME/trellis-box"

echo ""
print_status "Building TRELLIS-BOX Docker image..."
echo ""

# Build the image
if docker build -t "$LOCAL_IMAGE:$VERSION" .; then
    print_success "Image built successfully"
else
    print_error "Failed to build Docker image"
    exit 1
fi

echo ""

# Tag for Docker Hub
print_status "Tagging image for Docker Hub..."
docker tag "$LOCAL_IMAGE:$VERSION" "$REMOTE_IMAGE:$VERSION"

# Also tag as latest if version is not 'latest'
if [ "$VERSION" != "latest" ]; then
    docker tag "$LOCAL_IMAGE:$VERSION" "$REMOTE_IMAGE:latest"
    print_success "Tagged as $REMOTE_IMAGE:$VERSION and $REMOTE_IMAGE:latest"
else
    print_success "Tagged as $REMOTE_IMAGE:latest"
fi

echo ""

# Check if logged in to Docker Hub
if ! docker info 2>/dev/null | grep -q "Username: $DOCKER_USERNAME"; then
    print_status "Logging in to Docker Hub..."
    if ! docker login; then
        print_error "Failed to login to Docker Hub"
        exit 1
    fi
fi

echo ""
print_status "Pushing to Docker Hub..."
echo ""

# Push the versioned tag
if docker push "$REMOTE_IMAGE:$VERSION"; then
    print_success "Pushed $REMOTE_IMAGE:$VERSION"
else
    print_error "Failed to push $REMOTE_IMAGE:$VERSION"
    exit 1
fi

# Push latest tag if applicable
if [ "$VERSION" != "latest" ]; then
    if docker push "$REMOTE_IMAGE:latest"; then
        print_success "Pushed $REMOTE_IMAGE:latest"
    else
        print_warning "Failed to push $REMOTE_IMAGE:latest"
    fi
fi

echo ""
print_success "Successfully published TRELLIS-BOX to Docker Hub!"
echo ""
echo "To pull this image, users can run:"
echo "  docker pull $REMOTE_IMAGE:$VERSION"
if [ "$VERSION" != "latest" ]; then
    echo "  docker pull $REMOTE_IMAGE:latest"
fi
echo ""
echo "View on Docker Hub: https://hub.docker.com/r/$REMOTE_IMAGE"

