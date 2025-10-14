#!/bin/bash

# Enhanced GPU diagnostics for TRELLIS Docker setup

echo "=== TRELLIS GPU Diagnostics ==="
echo ""

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA drivers not found!"
    echo "Install NVIDIA drivers first:"
    echo "  Ubuntu: sudo apt install nvidia-driver-XXX"
    echo "  Check your GPU: lspci | grep -i nvidia"
    exit 1
fi

echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader

echo ""
echo "=== GPU Memory Status ==="
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader

echo ""
echo "=== Processes Using GPU ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

echo ""
echo "=== TRELLIS Memory Requirements ==="
echo "Peak usage: ~8-10 GB during generation"
echo "Recommended: 12GB+ GPU memory"

# Check memory
FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
TOTAL_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo ""
echo "Memory: ${FREE_MEM}MB free / ${TOTAL_MEM}MB total"

if [ "$FREE_MEM" -lt 8000 ]; then
    echo "⚠ Low memory: Less than 8GB free"
    echo "Recommendation: Terminate unnecessary GPU processes before running TRELLIS"
    echo ""
    echo "To terminate a process: sudo kill -9 <PID>"
    echo "GPU processes are listed above"
else
    echo "✓ Sufficient memory available for TRELLIS"
fi

echo ""
echo "=== Docker GPU Access Test ==="
echo "Testing NVIDIA Container Toolkit..."

    # Test Docker GPU access
    if docker run --rm --gpus all nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 nvidia-smi --query-gpu=name --format=csv,noheader > /dev/null 2>&1; then
        echo "✓ Docker GPU access: Available"
        GPU_NAME=$(docker run --rm --gpus all nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        echo "GPU in container: $GPU_NAME"
    else
        echo "✗ Docker GPU access: Unavailable"
        echo ""
        echo "Troubleshooting steps:"
        echo "1. Install NVIDIA Container Toolkit:"
        echo "   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
        echo "   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
        echo "   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
        echo "   sudo systemctl restart docker"
        echo ""
        echo "2. Test again: docker run --rm --gpus all nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 nvidia-smi"
    fi

echo ""
echo "=== System Information ==="
echo "Docker version: $(docker --version)"
echo "NVIDIA driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo "CUDA version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1-2).0"

echo ""
echo "=== TRELLIS System Readiness ==="
if [ "$FREE_MEM" -ge 8000 ] && docker run --rm --gpus all nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "✓ System ready for TRELLIS"
    echo "Command: ./scripts/run.sh"
else
    echo "✗ System requires configuration before running TRELLIS"
fi

