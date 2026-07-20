import gc
import os
from typing import Optional, Union

import torch


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def empty_device_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if _mps_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def synchronize_device(device: Optional[Union[str, torch.device]] = None) -> None:
    if device is not None:
        device = torch.device(device)
    if torch.cuda.is_available() and (device is None or device.type == "cuda"):
        torch.cuda.synchronize(device)
    if _mps_available() and (device is None or device.type == "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def cleanup_memory(device: Optional[Union[str, torch.device]] = None, synchronize: bool = False) -> None:
    gc.collect()
    if synchronize:
        synchronize_device(device)
    empty_device_cache()


def _mps_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )
