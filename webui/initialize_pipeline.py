"""Pipeline initialization and memory management for TRELLIS 3D Generator."""

import os
import gc
import glob
import time
import warnings
import torch
from pathlib import Path
from trellis.pipelines import TrellisImageTo3DPipeline

# Suppress warnings during pipeline initialization
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*deprecated.*")
warnings.filterwarnings("ignore", message=".*xFormers is available.*")
warnings.filterwarnings("ignore", message=".*torch.library.impl_abstract.*renamed.*")
warnings.filterwarnings("ignore", message=".*torch.library.register_fake.*")


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

    for _ in range(3):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if hasattr(torch.cuda, 'consolidate_memory'):
            torch.cuda.consolidate_memory()

    total_memory = torch.cuda.get_device_properties(0).total_memory
    try:
        for size_mb in [16, 64, 256]:
            size_bytes = size_mb * 1024 * 1024
            if size_bytes < total_memory * 0.1:
                temp = torch.zeros(size_bytes // 4, dtype=torch.float32, device='cuda')
                del temp
                torch.cuda.empty_cache()
    except RuntimeError:
        pass


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

    gc.collect(generation=2)
    gc.collect(generation=1)
    gc.collect(generation=0)
    gc.collect()
    gc.set_threshold(300, 5, 5)

    cleanup_temp_files(max_age_hours=1)
    defragment_memory()


def load_pipeline():
    """Load and configure the TRELLIS pipeline with memory optimizations."""
    print("Loading TRELLIS pipeline...")
    reduce_memory_usage()

    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()
    reduce_memory_usage()

    # Set models to evaluation mode and convert to half precision where appropriate
    for model_name, model in pipeline.models.items():
        if hasattr(model, 'eval'):
            model.eval()

        if 'flow' in model_name or 'decoder' in model_name:
            model.half()

            # Keep norm layers in fp32 for numerical stability
            from trellis.modules.norm import LayerNorm32, GroupNorm32, ChannelLayerNorm32
            from trellis.modules.sparse.norm import SparseGroupNorm32, SparseLayerNorm32
            from trellis.modules.attention.modules import MultiHeadRMSNorm
            from trellis.modules.sparse.attention.modules import SparseMultiHeadRMSNorm
            
            for module in model.modules():
                if isinstance(module, (LayerNorm32, GroupNorm32, ChannelLayerNorm32,
                                     SparseGroupNorm32, SparseLayerNorm32,
                                     MultiHeadRMSNorm, SparseMultiHeadRMSNorm)):
                    module.float()

    # Enable cuDNN and CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable gradient checkpointing for memory efficiency
        for model in pipeline.models.values():
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()

        # Additional optimizations
        os.environ['CUDNN_CONVOLUTION_BWD_FILTER_ALGO'] = '1'
        
        if hasattr(torch.jit, 'enable_onednn_fusion'):
            torch.jit.enable_onednn_fusion(True)

        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            os.environ['TORCH_USE_CUDA_DSA'] = '1'

    print("TRELLIS pipeline loaded successfully")
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    reduce_memory_usage()
    print(f"GPU Memory after optimization: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")

    return pipeline

