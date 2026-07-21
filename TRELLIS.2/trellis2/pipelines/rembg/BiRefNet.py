from typing import *
import os
import warnings
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image


DEFAULT_BIREFNET_MODEL = "ZhengPeng7/BiRefNet"
DEFAULT_BIREFNET_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
DEFAULT_BRIA_RMBG_REVISION = "5df4c9c76d8170882c34f6986e848ee07fd0ba43"
REMBG_PRESETS = {
    "auto": DEFAULT_BIREFNET_MODEL,
    "birefnet": DEFAULT_BIREFNET_MODEL,
    "lite": "ZhengPeng7/BiRefNet_lite",
    "bria": "briaai/RMBG-2.0",
}
DISABLED_REMBG_BACKENDS = {"none", "disabled", "off", "false", "0"}


def explain_gated_hf_model_error(model_name: str, error: Exception) -> None:
    message = str(error)
    lower_message = message.lower()
    if (
        "gated repo" not in lower_message
        and "401" not in message
        and "403" not in message
        and "authorized list" not in lower_message
    ):
        return

    model_url = f"https://huggingface.co/{model_name}"
    access_hint = (
        "Your token was found, but this Hugging Face account has not accepted "
        "the model license/access terms yet."
        if "403" in message or "authorized list" in lower_message
        else "A Hugging Face token with access to this gated model is required."
    )

    print(
        "\n[startup] Hugging Face authentication is required for background removal.",
        flush=True,
    )
    print(f"[startup] {access_hint}", flush=True)
    print(f"[startup] Open {model_url} and accept the license/request access.", flush=True)
    print(
        "[startup] Then run `huggingface-cli login` locally or set `HF_TOKEN` "
        "in the repo-root .env file.",
        flush=True,
    )


def resolve_rembg_model_name(default_model_name: str = DEFAULT_BIREFNET_MODEL) -> Optional[str]:
    backend = os.environ.get("TRELLIS2_REMBG_BACKEND", "auto").strip()
    model_name = os.environ.get("TRELLIS2_REMBG_MODEL", "").strip()
    backend_key = backend.lower()

    if model_name.lower() in DISABLED_REMBG_BACKENDS or backend_key in DISABLED_REMBG_BACKENDS:
        return None

    if model_name:
        return REMBG_PRESETS.get(model_name.lower(), model_name)

    if backend_key in REMBG_PRESETS:
        return REMBG_PRESETS[backend_key]

    return backend or default_model_name


def resolve_rembg_revision(model_name: str, revision: Optional[str] = None) -> Optional[str]:
    env_revision = os.environ.get("TRELLIS2_REMBG_REVISION")
    revision = env_revision.strip() if env_revision is not None else revision
    if revision:
        return revision
    if model_name == DEFAULT_BIREFNET_MODEL:
        return DEFAULT_BIREFNET_REVISION
    if model_name == "briaai/RMBG-2.0":
        return DEFAULT_BRIA_RMBG_REVISION
    return None


def resolve_rembg_dtype() -> Optional[torch.dtype]:
    value = os.environ.get("TRELLIS2_REMBG_DTYPE", "float16").strip().lower()
    if value in {"", "none", "auto"}:
        return torch.float16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp32", "float32", "full"}:
        return torch.float32
    raise ValueError(f"Unknown TRELLIS2_REMBG_DTYPE={value}")


def create_rembg_model(default_model_name: str = DEFAULT_BIREFNET_MODEL):
    model_name = resolve_rembg_model_name(default_model_name)
    if model_name is None:
        print("[startup] Background removal disabled. RGBA images with alpha are required.", flush=True)
        return None
    print(f"[startup] Background removal backend: {model_name}", flush=True)
    return LazyBiRefNet(model_name=model_name)


class BiRefNet:
    def __init__(
        self,
        model_name: str = DEFAULT_BIREFNET_MODEL,
        revision: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.preferred_device = (device or os.environ.get("TRELLIS2_REMBG_DEVICE", "auto")).strip().lower()
        revision = resolve_rembg_revision(model_name, revision)
        try:
            warnings.filterwarnings(
                "ignore",
                message=r"Importing from timm\.models\..* is deprecated.*",
                category=FutureWarning,
            )
            self.model = AutoModelForImageSegmentation.from_pretrained(
                model_name,
                revision=revision,
                trust_remote_code=True,
                torch_dtype=resolve_rembg_dtype(),
            )
        except Exception as e:
            explain_gated_hf_model_error(model_name, e)
            raise
        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    def to(self, device: str):
        if self.preferred_device in {"cpu", "cuda"}:
            device = self.preferred_device
        self.model.to(device)
        return self

    def cuda(self):
        return self.to("cuda")

    def cpu(self):
        self.model.cpu()
        return self

    @property
    def device(self):
        return next(self.model.parameters()).device
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        param = next(self.model.parameters())
        input_images = self.transform_image(image).unsqueeze(0).to(
            device=param.device,
            dtype=param.dtype,
        )
        # Prediction
        with torch.inference_mode():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image


class LazyBiRefNet:
    def __init__(self, model_name: str = DEFAULT_BIREFNET_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self) -> BiRefNet:
        if self._model is None:
            print(f"[startup] Loading background remover: {self.model_name}", flush=True)
            self._model = BiRefNet(self.model_name)
        return self._model

    def to(self, device: str):
        self._load().to(device)
        return self

    def cuda(self):
        return self.to("cuda")

    def cpu(self):
        if self._model is not None:
            self._model.cpu()
        return self

    @property
    def device(self):
        return self._load().device

    def __call__(self, image: Image.Image) -> Image.Image:
        return self._load()(image)
    
