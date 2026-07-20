from typing import *
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import DINOv3ViTModel
import timm
import numpy as np
from PIL import Image


ORIGINAL_DINOV3_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
PUBLIC_TIMM_DINOV3_MODEL = "hf_hub:timm/vit_large_patch16_dinov3_qkvb.lvd1689m"


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def resolve_image_cond_model_config(config: dict) -> dict:
    config = {
        "name": config["name"],
        "args": dict(config.get("args", {})),
        **{key: value for key, value in config.items() if key not in {"name", "args"}},
    }

    if config["name"] != "DinoV3FeatureExtractor":
        return config

    image_encoder = os.environ.get("TRELLIS2_IMAGE_ENCODER", "timm-dinov3").strip()
    image_encoder_key = image_encoder.lower()
    if image_encoder_key in {"meta-dinov3", "original-dinov3", "transformers-dinov3"}:
        return config

    if image_encoder_key == "timm-dinov3":
        model_name = os.environ.get("TRELLIS2_DINOV3_MODEL", PUBLIC_TIMM_DINOV3_MODEL)
    elif image_encoder_key == "custom":
        model_name = os.environ.get("TRELLIS2_DINOV3_MODEL")
        if not model_name:
            raise ValueError("TRELLIS2_DINOV3_MODEL is required when TRELLIS2_IMAGE_ENCODER=custom")
    elif image_encoder.startswith("hf_hub:"):
        model_name = image_encoder
    elif "/" in image_encoder:
        model_name = f"hf_hub:{image_encoder}"
    else:
        raise ValueError(
            "Unknown TRELLIS2_IMAGE_ENCODER="
            f"{image_encoder!r}; expected timm-dinov3, meta-dinov3, custom, "
            "or a timm/HF Hub model id"
        )

    image_size = int(os.environ.get("TRELLIS2_DINOV3_IMAGE_SIZE", config["args"].get("image_size", 512)))
    config["name"] = "TimmDinoV3FeatureExtractor"
    config["args"] = {
        "model_name": model_name,
        "image_size": image_size,
        "truncate_rope_periods": env_flag("TRELLIS2_TIMM_DINOV3_TRUNCATE_ROPE", True),
    }
    print(
        f"[startup] Using timm DINOv3 image encoder: {model_name}. "
        "Set TRELLIS2_IMAGE_ENCODER=meta-dinov3 to use the gated Transformers checkpoint.",
        flush=True,
    )
    return config


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
        "\n[startup] Hugging Face authentication is required for this model.",
        flush=True,
    )
    print(
        f"[startup] {access_hint}",
        flush=True,
    )
    print(
        f"[startup] Open {model_url} and accept the license/request access.",
        flush=True,
    )
    print(
        "[startup] Then run `huggingface-cli login` locally or set `HF_TOKEN` "
        "in the repo-root .env file.",
        flush=True,
    )


class DinoV2FeatureExtractor:
    """
    Feature extractor for DINOv2 models.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()

    @property
    def device(self):
        return next(self.model.parameters()).device
    
    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features from the image.
        
        Args:
            image: A batch of images as a tensor of shape (B, C, H, W) or a list of PIL images.
        
        Returns:
            A tensor of shape (B, N, D) where N is the number of patches and D is the feature dimension.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.transform(image).to(self.device)
        features = self.model(image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
    

class DinoV3FeatureExtractor:
    """
    Feature extractor for DINOv3 models.
    """
    def __init__(self, model_name: str, image_size=512):
        self.model_name = model_name
        try:
            self.model = DINOv3ViTModel.from_pretrained(model_name)
        except Exception as e:
            explain_gated_hf_model_error(model_name, e)
            raise
        self.model.eval()
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)

        for i, layer_module in enumerate(self.model.layer):
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        return F.layer_norm(hidden_states, hidden_states.shape[-1:])
        
    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features from the image.
        
        Args:
            image: A batch of images as a tensor of shape (B, C, H, W) or a list of PIL images.
        
        Returns:
            A tensor of shape (B, N, D) where N is the number of patches and D is the feature dimension.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.transform(image).to(self.device)
        features = self.extract_features(image)
        return features


class TimmDinoV3FeatureExtractor:
    """
    DINOv3 ViT-L/16 feature extractor using the public timm HF Hub weights.
    """
    def __init__(
        self,
        model_name: str = PUBLIC_TIMM_DINOV3_MODEL,
        image_size: int = 512,
        truncate_rope_periods: bool = True,
        pretrained: bool = True,
    ):
        self.model_name = model_name
        self.image_size = image_size
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=image_size,
            num_classes=0,
        )
        if truncate_rope_periods and hasattr(self.model, "rope"):
            self.model.rope.periods = self.model.rope.periods.to(torch.bfloat16).to(torch.float32)
        self.model.eval()
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def to(self, device):
        self.model.to(device)
        return self

    def cuda(self):
        self.model.cuda()
        return self

    def cpu(self):
        self.model.cpu()
        return self

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        if isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image)
        elif isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        dtype = next(self.model.parameters()).dtype
        image = self.transform(image).to(device=self.device, dtype=dtype)
        features = self.model.forward_features(image)
        if isinstance(features, dict):
            features = features.get("x")
            if features is None:
                features = features.get("last_hidden_state")
        if not isinstance(features, torch.Tensor):
            raise RuntimeError(f"Unexpected DINOv3 feature output type: {type(features)}")
        if features.ndim != 3 or features.shape[-1] != 1024:
            raise RuntimeError(f"Unexpected DINOv3 feature shape: {tuple(features.shape)}")
        return features
