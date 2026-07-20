from typing import *
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from .. import models


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
        import os
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

        _models = {}
        model_items = [
            (k, v)
            for k, v in args['models'].items()
            if not hasattr(cls, 'model_names_to_load') or k in cls.model_names_to_load
        ]
        for k, v in tqdm(model_items, desc="Loading TRELLIS.2 models", unit="model"):
            print(f"[startup] Loading model: {k}", flush=True)
            try:
                _models[k] = models.from_pretrained(f"{path}/{v}", _progress_name=k)
            except Exception:
                print(f"[startup] Falling back to model path: {v}", flush=True)
                _models[k] = models.from_pretrained(v, _progress_name=k)

        new_pipeline = cls(_models)
        new_pipeline._pretrained_args = args
        print("[startup] Pipeline models loaded.", flush=True)
        return new_pipeline

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
