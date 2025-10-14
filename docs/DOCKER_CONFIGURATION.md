# Docker Configuration Guide

This guide explains how to configure and customize the Trellis 3D Docker image build and runtime.

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration Variables](#configuration-variables)
- [Configuration Methods](#configuration-methods)
- [Common Use Cases](#common-use-cases)
- [Build Optimization](#build-optimization)

## Quick Start

### Default Build

```bash
# Using the build script (recommended)
./scripts/build.sh

# Or using docker-compose
docker-compose build

# Or using docker directly
DOCKER_BUILDKIT=1 docker build -t trellis-box:latest .
```

### Custom Configurations

Quick start with custom configuration:
```bash
# Copy the example configuration
cp docker.env.example .env

# Edit to your preferences
nano .env

# Build and run with your configuration
docker-compose up --build
```

You may also set environment variables via export and --build-arg
```bash
# Option 2: Using environment variables with the build script
CUDA_VERSION=12.2.0 PYTHON_VERSION=3.11 ./scripts/build.sh

# Option 3: Using --build-arg with docker
docker build \
  --build-arg CUDA_VERSION=12.2.0 \
  --build-arg PYTHON_VERSION=3.11 \
  --build-arg APP_PORT=8080 \
  -t trellis-box:custom .
```

## Configuration Variables

### CUDA and System

| Variable | Default | Description | Example Values |
|----------|---------|-------------|----------------|
| `CUDA_VERSION` | `12.3.2` | NVIDIA CUDA version | `12.3.2`, `12.2.2`, `12.1.1` |
| `CUDNN_VERSION` | `9` | cuDNN version | `8`, `9` |
| `UBUNTU_VERSION` | `22.04` | Ubuntu base version | `20.04`, `22.04` |
| `PYTHON_VERSION` | `3.10` | Python version | `3.10`, `3.11` |

### Python Packages

| Variable | Default | Description |
|----------|---------|-------------|
| `POETRY_VERSION` | `1.8.3` | Poetry package manager version |
| `TORCH_VERSION` | `2.4.0` | PyTorch version (for reference) |
| `KAOLIN_VERSION` | `0.17.0` | NVIDIA Kaolin library version |
| `KAOLIN_INDEX_URL` | `https://nvidia-kaolin...` | Kaolin pip index URL |
| `TORCH_CUDA_ARCH_LIST` | `7.0 7.5 8.0 8.6 8.9 9.0` | GPU architectures to compile for |

### Application

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_USER` | `appuser` | Non-root user inside container |
| `APP_UID` | `1000` | User ID (match with host for permissions) |
| `APP_PORT` | `8501` | Streamlit application port |
| `HOST_PORT` | `8501` | Port exposed on host (compose only) |
| `STREAMLIT_SERVER_ADDRESS` | `0.0.0.0` | Streamlit bind address |
| `STREAMLIT_SERVER_HEADLESS` | `true` | Run without browser auto-open |

### Cache Directories

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_DIR` | `/home/appuser/.cache` | Main cache directory inside container |
| `HF_CACHE_DIR` | `/home/appuser/.cache/huggingface` | Hugging Face models cache |
| `REMBG_CACHE_DIR` | `/home/appuser/.u2net` | Rembg background removal models |
| `TRELLIS_OUTPUT_DIR` | `/tmp/Trellis-demo` | Generated outputs directory |
| `CACHE_VOLUME` | `trellis-cache` | Docker volume name for main cache |
| `HF_CACHE_VOLUME` | `huggingface-cache` | Docker volume name for HF cache |
| `REMBG_CACHE_VOLUME` | `rembg-cache` | Docker volume name for rembg cache |
| `OUTPUTS_HOST_DIR` | `./outputs` | Host directory for outputs (bind mount) |

### GPU Runtime

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `all` | Which GPUs to use |
| `GPU_COUNT` | `all` | Number of GPUs to allocate |

### Docker Image

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_NAME` | `trellis-box` | Docker image name |
| `IMAGE_TAG` | `latest` | Docker image tag |

## Configuration Methods

### Method 1: Environment Variables (Build Script)

```bash
# Set variables before running the script
export CUDA_VERSION=12.2.0
export PYTHON_VERSION=3.11
export APP_PORT=8080

./scripts/build.sh
```

Or inline:

```bash
CUDA_VERSION=12.2.0 PYTHON_VERSION=3.11 ./scripts/build.sh
```

### Method 2: .env File (Docker Compose)

```bash
# Create your .env file
cp docker.env.example .env

# Edit the .env file
nano .env  # or vim, code, etc.

# Build and run
docker-compose up --build
```

### Method 3: Build Args (Direct Docker)

```bash
docker build \
  --build-arg CUDA_VERSION=12.2.0 \
  --build-arg PYTHON_VERSION=3.11 \
  --build-arg APP_PORT=8080 \
  -t trellis-box:custom .
```

### Method 4: Edit Dockerfile Directly

Modify the default values in the `Dockerfile`:

```dockerfile
ARG CUDA_VERSION=12.2.0  # Change from 12.1.0
ARG PYTHON_VERSION=3.11   # Change from 3.10
```

## Common Use Cases

### Use Case 1: Different CUDA Version

Your system has CUDA 12.2.2 instead of 12.3.2:

```bash
# Method A: Build script
CUDA_VERSION=12.2.2 ./scripts/build.sh

# Method B: Docker compose .env
echo "CUDA_VERSION=12.2.2" >> .env
docker-compose build
```

**Note:** Check [NVIDIA CUDA Docker Hub](https://hub.docker.com/r/nvidia/cuda/tags) for available versions and their support status.

### Use Case 2: Match Host User ID

Fix file permission issues by matching container UID to host UID:

```bash
# Get your user ID
MY_UID=$(id -u)

# Build with your UID
APP_UID=$MY_UID ./scripts/build.sh

# Or add to .env
echo "APP_UID=$MY_UID" >> .env
```

### Use Case 3: Run on Different Port

You have another service on port 8501:

```bash
# Using .env file
cat > .env << EOF
APP_PORT=8080
HOST_PORT=8080
EOF

docker-compose up --build
```

Then access at `http://localhost:8080`

### Use Case 4: Multi-GPU with Specific GPUs

Use only GPUs 0 and 1 out of 4 available:

```bash
# Add to .env
cat > .env << EOF
CUDA_VISIBLE_DEVICES=0,1
GPU_COUNT=2
EOF

docker-compose up
```

### Use Case 5: Development vs Production Builds

```bash
# Development (faster rebuilds)
IMAGE_TAG=dev ./scripts/build.sh

# Production (specific version)
IMAGE_TAG=v1.0.0 ./scripts/build.sh --no-cache
```

### Use Case 6: Custom Cache Directories

Run multiple instances with separate caches:

```bash
# Instance 1: Development
cat > .env.dev << EOF
CACHE_VOLUME=trellis-dev-cache
HF_CACHE_VOLUME=huggingface-dev-cache
REMBG_CACHE_VOLUME=rembg-dev-cache
OUTPUTS_HOST_DIR=./outputs-dev
HOST_PORT=8501
EOF

# Instance 2: Production
cat > .env.prod << EOF
CACHE_VOLUME=trellis-prod-cache
HF_CACHE_VOLUME=huggingface-prod-cache
REMBG_CACHE_VOLUME=rembg-prod-cache
OUTPUTS_HOST_DIR=./outputs-prod
HOST_PORT=8502
EOF

# Run dev instance
docker-compose --env-file .env.dev up -d

# Run prod instance (in a different directory or with different container name)
docker-compose --env-file .env.prod -p trellis-prod up -d
```

### Use Case 7: Centralized Cache Storage

Use a shared cache location across multiple projects:

```bash
# Mount a centralized cache directory
cat > .env << EOF
CACHE_DIR=/shared/cache/trellis
HF_CACHE_DIR=/shared/cache/huggingface
REMBG_CACHE_DIR=/shared/cache/rembg
OUTPUTS_HOST_DIR=/shared/outputs/trellis
EOF

docker-compose build
docker-compose up
```

### Use Case 8: Store Caches on Fast Storage

Put caches on SSD for better performance:

```bash
# Use absolute paths to NVMe/SSD storage
cat > .env << EOF
OUTPUTS_HOST_DIR=/mnt/nvme/trellis-outputs
EOF

docker-compose up
```

## Build Optimization

### Faster Rebuilds with BuildKit Cache

The Dockerfile uses BuildKit cache mounts for faster rebuilds:

```bash
# Enable BuildKit (usually enabled by default)
export DOCKER_BUILDKIT=1

# Build with cache
./scripts/build.sh

# Subsequent builds reuse cached layers
./scripts/build.sh  # Much faster!
```

### Force Clean Build

```bash
# Using build script
./scripts/build.sh --no-cache

# Using docker-compose
docker-compose build --no-cache

# Using docker
docker build --no-cache -t trellis-box .
```

### Multi-Stage Build Benefits

The Dockerfile uses multi-stage builds:

- **Builder stage**: Has all development tools and build dependencies
- **Runtime stage**: Only includes what's needed to run the app

This results in:
- ✅ Smaller final image (~30-40% reduction)
- ✅ Faster deployments
- ✅ Better security (fewer packages = smaller attack surface)

### Layer Caching Strategy

Layers are ordered by change frequency:

1. System dependencies (rarely change)
2. Poetry and Python packages (change when pyproject.toml changes)
3. Application code (changes frequently)

This means editing your Python code won't trigger a full dependency reinstall.

## Cache Management

### Understanding Cache Directories

The Trellis 3D application uses several cache directories:

1. **Main Cache** (`CACHE_DIR`): General Python/pip cache
2. **Hugging Face Cache** (`HF_CACHE_DIR`): Downloaded model weights from Hugging Face
3. **Rembg Cache** (`REMBG_CACHE_DIR`): U2-Net models for background removal
4. **Outputs** (`TRELLIS_OUTPUT_DIR`): Generated 3D models and outputs

### Cache Volume Types

**Docker Named Volumes** (default for caches):
- Pros: Managed by Docker, persistent across container rebuilds
- Cons: Not directly accessible from host filesystem
- Use for: Model caches that don't need direct host access

**Bind Mounts** (default for outputs):
- Pros: Direct access from host, easy backup
- Cons: Permission issues if UID mismatches
- Use for: Output files you want to access from host

### Managing Cache Volumes

```bash
# List all volumes
docker volume ls

# Inspect a volume
docker volume inspect trellis-cache

# Remove a volume (will re-download models on next run)
docker volume rm trellis-cache

# Backup a volume
docker run --rm \
  -v trellis-cache:/data \
  -v $(pwd):/backup \
  ubuntu tar czf /backup/trellis-cache-backup.tar.gz /data

# Restore a volume
docker run --rm \
  -v trellis-cache:/data \
  -v $(pwd):/backup \
  ubuntu tar xzf /backup/trellis-cache-backup.tar.gz -C /
```

### Pre-populating Cache

To avoid downloading models on first run, you can pre-populate the cache:

```bash
# Start container and let it download models
docker-compose up

# Models are now cached in Docker volumes
# Next startup will be faster

# Alternative: Copy from another machine's cache
docker run --rm -v trellis-cache:/target -v /path/to/source:/source \
  ubuntu cp -r /source/* /target/
```

## Troubleshooting

### Issue: Permission Denied on Volumes

**Solution**: Match APP_UID to your host user ID

```bash
APP_UID=$(id -u) ./scripts/build.sh
```

### Issue: CUDA Version Mismatch

**Error**: `CUDA version mismatch` or `driver version is insufficient`

**Solution**: Check your NVIDIA driver version and use compatible CUDA:

```bash
nvidia-smi  # Check driver version

# Use appropriate CUDA version
CUDA_VERSION=12.0.1 ./scripts/build.sh
```

### Issue: Build Fails on Kaolin

**Solution**: Kaolin requires specific CUDA and PyTorch versions. Ensure compatibility:

- PyTorch 2.4.0 → CUDA 12.1
- Check [Kaolin compatibility](https://github.com/NVIDIAGameWorks/kaolin)

### Issue: Port Already in Use

**Error**: `port is already allocated`

**Solution**: Use a different host port

```bash
HOST_PORT=8502 docker-compose up
```

### Issue: Disk Space - Cache Too Large

**Error**: Running out of disk space due to large model caches

**Solution 1**: Move caches to a larger disk

```bash
# In .env
CACHE_VOLUME=trellis-cache-large
# Then create volume on different disk (Docker daemon config)
```

**Solution 2**: Use bind mounts to specific disk

```bash
# In .env
CACHE_DIR=/mnt/large-disk/trellis/.cache
HF_CACHE_DIR=/mnt/large-disk/trellis/huggingface
```

**Solution 3**: Clean up unused caches

```bash
# Remove all unused volumes
docker volume prune

# Or remove specific volumes
docker-compose down
docker volume rm trellis-cache huggingface-cache rembg-cache
```

### Issue: Slow Model Loading

**Problem**: Models take too long to load from cache

**Solution 1**: Use faster storage (NVMe/SSD) for caches

```bash
# Mount cache on fast storage
cat > .env << EOF
CACHE_DIR=/mnt/nvme/.cache
HF_CACHE_DIR=/mnt/nvme/huggingface
EOF
```

**Solution 2**: Increase Docker's cache size limits

Edit Docker daemon config (`/etc/docker/daemon.json`):

```json
{
  "data-root": "/mnt/fast-disk/docker"
}
```

### Issue: No CUDA Device Found

**Problem**: Container shows "No CUDA runtime is found" or "no CUDA-capable device is detected"

**Solution 1**: Verify NVIDIA Container Toolkit installation

```bash
# Check if NVIDIA Container Toolkit is installed
docker run --rm --gpus all nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 nvidia-smi

# If that fails, install NVIDIA Container Toolkit
# Ubuntu/Debian:
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Solution 2**: Check GPU visibility on host

```bash
# Verify GPU is visible on host
nvidia-smi

# Check Docker can see GPUs
docker run --rm --gpus all ubuntu nvidia-smi
```

**Solution 3**: For remote GPU machines (like via SSH)

If running on a remote machine (like 10.0.0.217), ensure:

1. SSH into the GPU machine: `ssh user@10.0.0.217`
2. Run the Docker commands on the GPU machine directly
3. Or set up Docker context for remote machine

**Solution 4**: Environment-specific GPU access

```bash
# Force specific GPU
CUDA_VISIBLE_DEVICES=0 ./scripts/run.sh

# Or in .env file
echo "CUDA_VISIBLE_DEVICES=0" >> .env

# For multiple GPUs
echo "CUDA_VISIBLE_DEVICES=0,1" >> .env
```

## Advanced Configuration

### Custom Wheel Files

If you have custom `.whl` files, place them in the `wheels/` directory before building:

```bash
ls wheels/
# your-custom-package.whl

./scripts/build.sh
```

**Note**: `diff_gaussian_rasterization` is always built from source for CUDA compatibility.

**Note on Package Installation**: The Dockerfile handles package installation with intelligent fallbacks:

- **flash-attention**: Tries wheel first, falls back to source build from PyPI
- **diff-gaussian-rasterization**: Always built from [GitHub source](https://github.com/graphdeco-inria/differentiable-gaussian-rasterization) for CUDA compatibility
- **nvdiffrast**: Always built from source in `extensions/` directory for CUDA compatibility

This ensures maximum compatibility across different CUDA versions and hardware configurations.

### Modify Poetry Dependencies

1. Edit `pyproject.toml` to add/update dependencies
2. Rebuild the image:

```bash
./scripts/build.sh
```

The build automatically runs `poetry install` with the updated dependencies.

## Best Practices

1. **Version Pin Everything**: Use specific versions for reproducible builds
2. **Use .env for Secrets**: Never commit `.env` files with sensitive data
3. **Match Host UID**: Set `APP_UID=$(id -u)` to avoid permission issues
4. **Tag Your Images**: Use meaningful tags like `v1.0.0` or `prod` instead of just `latest`
5. **Test Before Deploy**: Build and test locally before deploying to production
6. **Document Changes**: If you modify defaults, document why in your project

## Resources

- [Docker BuildKit Documentation](https://docs.docker.com/build/buildkit/)
- [Multi-stage Builds](https://docs.docker.com/build/building/multi-stage/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)

