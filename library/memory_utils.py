"""
Memory utilities for TRELLIS application.
Pure utility functions without UI dependencies.
"""

import os
import gc
import glob
import time
import torch


def cleanup_temp_files(max_age_hours=24):
    """Clean up old temporary files."""
    temp_dirs = ["/tmp", "/tmp/Trellis-demo"]
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    for temp_dir in temp_dirs:
        if not os.path.exists(temp_dir):
            continue
        try:
            for pattern in ["*.png", "*.jpg", "*.glb", "*.ply"]:
                for file_path in glob.glob(os.path.join(temp_dir, pattern)):
                    try:
                        if current_time - os.path.getmtime(file_path) > max_age_seconds:
                            os.remove(file_path)
                    except (OSError, PermissionError):
                        pass
        except Exception:
            pass


def defragment_memory():
    """Aggressive memory defragmentation for PyTorch CUDA allocator."""
    if not torch.cuda.is_available():
        return

    # Force garbage collection first
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Try to trigger memory defragmentation
    if hasattr(torch.cuda, 'consolidate_memory'):
        torch.cuda.consolidate_memory()

    # Clear cache again
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def reduce_memory_usage():
    """Critical memory management: clears cache, forces GC, and optimizes memory layout."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if hasattr(torch.cuda, 'consolidate_memory'):
            torch.cuda.consolidate_memory()

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        if hasattr(torch.cuda, 'reset_max_memory_allocated'):
            torch.cuda.reset_max_memory_allocated()
        if hasattr(torch.cuda, 'reset_max_memory_cached'):
            torch.cuda.reset_max_memory_cached()

        temp = torch.zeros(1, device='cuda')
        del temp

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Force Python garbage collection
    gc.collect()
