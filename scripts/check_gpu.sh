#!/bin/bash

# Check GPU memory usage and help identify processes to kill

echo "=== GPU Memory Status ==="
nvidia-smi

echo ""
echo "=== Processes Using GPU ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

echo ""
echo "=== TRELLIS Memory Requirements ==="
echo "Peak usage: ~8-10 GB during generation"
echo "Your GPU: 12 GB total"
echo ""

# Check if there's enough free memory
FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
echo "Currently free: ${FREE_MEM} MB"

if [ "$FREE_MEM" -lt 8000 ]; then
    echo ""
    echo "⚠️  WARNING: Less than 8GB free!"
    echo "Recommendation: Kill unnecessary processes before running TRELLIS"
    echo ""
    echo "To kill a process: sudo kill -9 <PID>"
    echo "PIDs using GPU are listed above"
else
    echo ""
    echo "✓ Sufficient memory available to run TRELLIS"
fi

