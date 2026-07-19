import os
import platform
from contextlib import nullcontext

import torch


def get_trellis_device() -> torch.device:
    """Return the requested TRELLIS runtime device."""
    requested = os.environ.get("TRELLIS_DEVICE", "auto").strip().lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # The portable sparse fallback relies on Conv3d-heavy dense ops. CPU is
        # the most compatible macOS default; MPS remains available explicitly.
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("TRELLIS_DEVICE=cuda requested, but CUDA is unavailable.")
        return torch.device("cuda")

    if requested == "mps":
        if not _mps_available():
            raise RuntimeError("TRELLIS_DEVICE=mps requested, but MPS is unavailable.")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError("TRELLIS_DEVICE must be one of: auto, cuda, mps, cpu.")


def get_trellis_device_type() -> str:
    return get_trellis_device().type


def is_cuda_runtime() -> bool:
    return get_trellis_device().type == "cuda"


def is_portable_runtime() -> bool:
    return get_trellis_device().type in {"cpu", "mps"}


def is_macos() -> bool:
    return platform.system() == "Darwin"


def autocast_context():
    if torch.cuda.is_available() and get_trellis_device().type == "cuda":
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def configure_torch_threads() -> None:
    requested = os.environ.get("TORCH_NUM_THREADS") or os.environ.get("TRELLIS_CPU_THREADS")
    if not requested:
        return
    try:
        threads = int(requested)
    except ValueError:
        return
    if threads > 0:
        torch.set_num_threads(threads)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(max(1, min(threads, 4)))
            except RuntimeError:
                pass


def empty_device_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if _mps_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def synchronize_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if _mps_available() and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _mps_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )
