from typing import *

DEBUG = False

def __detect_available_backend():
    """Detect the best available attention backend"""
    import os

    # Check environment variable first
    env_attn_backend = os.environ.get('ATTN_BACKEND')
    if env_attn_backend is not None and env_attn_backend in ['xformers', 'flash_attn', 'sdpa', 'naive']:
        return env_attn_backend

    try:
        from trellis.utils.device import get_trellis_device
        if get_trellis_device().type != 'cuda':
            import torch
            return 'sdpa' if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else 'naive'
    except Exception:
        pass

    # Auto-detect available backends in order of preference
    try:
        import flash_attn
        return 'flash_attn'
    except ImportError:
        pass

    try:
        import xformers.ops
        return 'xformers'
    except ImportError:
        pass

    # Check if SDPA is available (PyTorch 2.0+)
    try:
        import torch
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            return 'sdpa'
    except ImportError:
        pass

    # Fallback to naive implementation
    return 'naive'

BACKEND = __detect_available_backend()

def __from_env():
    import os

    global BACKEND
    global DEBUG

    env_sttn_debug = os.environ.get('ATTN_DEBUG')

    if env_sttn_debug is not None:
        DEBUG = env_sttn_debug == '1'

    print(f"[ATTENTION] Using backend: {BACKEND}")


__from_env()
    

def set_backend(backend: Literal['xformers', 'flash_attn', 'sdpa', 'naive']):
    global BACKEND
    BACKEND = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug


from .full_attn import *
from .modules import *
