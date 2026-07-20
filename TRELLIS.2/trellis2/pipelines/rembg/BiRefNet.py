from typing import *
import os
import warnings
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image


DEFAULT_RMBG_REVISION = "5df4c9c76d8170882c34f6986e848ee07fd0ba43"


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


class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        self.model_name = model_name
        revision = os.environ.get("TRELLIS2_REMBG_REVISION")
        if model_name == "briaai/RMBG-2.0":
            revision = revision or DEFAULT_RMBG_REVISION
        try:
            warnings.filterwarnings(
                "ignore",
                message=r"Importing from timm\.models\..* is deprecated.*",
                category=FutureWarning,
            )
            self.model = AutoModelForImageSegmentation.from_pretrained(
                model_name, trust_remote_code=True, revision=revision
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
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()

    @property
    def device(self):
        return next(self.model.parameters()).device
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to(self.device)
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
    
