from typing import *
import gc
import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from .. import models


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Pipeline":
        """
        Load a pretrained model.
        """
        import json
        is_local = os.path.exists(f"{path}/{config_file}")
        print(f"[startup] Loading pipeline config: {path}/{config_file}", flush=True)

        if is_local:
            config_file = f"{path}/{config_file}"
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, config_file)

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        model_items = [
            (k, v)
            for k, v in args['models'].items()
            if not hasattr(cls, 'model_names_to_load') or k in cls.model_names_to_load
        ]
        lazy_model_names = set(getattr(cls, 'lazy_model_names', []))
        lazy_env = getattr(cls, 'lazy_model_loading_env', 'TRELLIS_LAZY_LOAD_MODELS')
        lazy_default = getattr(cls, 'lazy_model_loading_default', False)
        lazy_model_loading = env_flag(lazy_env, lazy_default)

        _models = {}
        eager_model_items = [
            (k, v)
            for k, v in model_items
            if not lazy_model_loading or k not in lazy_model_names
        ]
        if lazy_model_loading and lazy_model_names:
            deferred = [k for k, _ in model_items if k in lazy_model_names]
            print(
                "[startup] Lazy model loading enabled. "
                f"Deferred {len(deferred)} model(s): {', '.join(deferred)}",
                flush=True,
            )
        for k, v in tqdm(eager_model_items, desc="Loading TRELLIS.2 models", unit="model"):
            _models[k] = cls._load_pretrained_model(path, k, v)

        new_pipeline = cls(_models)
        new_pipeline._pretrained_args = args
        new_pipeline._pretrained_path = path
        new_pipeline._pretrained_model_specs = dict(model_items)
        new_pipeline._lazy_model_names = lazy_model_names
        new_pipeline._lazy_model_loading = lazy_model_loading
        if lazy_model_loading and lazy_model_names:
            print("[startup] Pipeline model registry ready. Weights will load on first use.", flush=True)
        else:
            print("[startup] Pipeline models loaded.", flush=True)
        return new_pipeline

    @staticmethod
    def _load_pretrained_model(path: str, name: str, model_path: str) -> nn.Module:
        print(f"[startup] Loading model: {name}", flush=True)
        try:
            model = models.from_pretrained(f"{path}/{model_path}", _progress_name=name)
        except Exception:
            print(f"[startup] Falling back to model path: {model_path}", flush=True)
            model = models.from_pretrained(model_path, _progress_name=name)
        model.eval()
        return model

    def ensure_model(self, name: str) -> nn.Module:
        if name not in self.models:
            if not hasattr(self, '_pretrained_model_specs') or name not in self._pretrained_model_specs:
                raise KeyError(f"Unknown pretrained model: {name}")
            self.models[name] = self._load_pretrained_model(
                self._pretrained_path,
                name,
                self._pretrained_model_specs[name],
            )
            if not getattr(self, 'low_vram', False) and hasattr(self, '_device'):
                self.models[name].to(self.device)
        return self.models[name]

    def ensure_models(self, names: Iterable[str]) -> None:
        names = list(names)
        missing = [name for name in names if name not in self.models]
        iterator = tqdm(missing, desc="Loading selected TRELLIS.2 models", unit="model") if missing else []
        for name in iterator:
            self.ensure_model(name)

    def unload_model(self, name: str) -> None:
        model = self.models.pop(name, None)
        if model is None:
            return
        if isinstance(model, nn.Module):
            model.cpu()
        del model
        gc.collect()

    def unload_models(self, names: Iterable[str]) -> None:
        for name in names:
            self.unload_model(name)

    @property
    def device(self) -> torch.device:
        if hasattr(self, '_device'):
            return self._device
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for name, model in tqdm(self.models.items(), desc=f"Moving models to {device}", unit="model"):
            print(f"[startup] Moving model to {device}: {name}", flush=True)
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))
