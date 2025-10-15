"""
Image refinement module for TRELLIS-BOX using SSD-1B (Segmind Stable Diffusion).

This module provides pre-processing image refinement before 3D generation,
improving input quality for better 3D reconstruction results.

SSD-1B is a distilled version of SDXL with 50% less memory usage while maintaining
90-95% of the quality, making it ideal for consumer GPUs.
"""

from typing import Union, Optional
import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusionXLImg2ImgPipeline
from transformers import CLIPProcessor, CLIPModel


class ImageRefiner:
    """
    Refines input images using SSD-1B (Segmind Stable Diffusion) before 3D generation.

    This improves texture quality, reduces artifacts, and enhances details
    in the input image, leading to better 3D reconstruction quality.

    SSD-1B uses ~4-5GB VRAM (vs SDXL's 8-12GB) while maintaining excellent quality.

    Uses CLIP for content-aware prompt generation to ensure minimal content changes
    while improving image quality.

    Args:
        model_id: Hugging Face model ID for the refiner
        device: Device to run the model on ('cuda' or 'cpu')
        use_fp16: Whether to use half precision for memory efficiency (automatically disabled on CPU)
        use_clip_prompts: Whether to use CLIP for generating content-aware prompts (recommended for content preservation)
        clip_model_id: Hugging Face model ID for CLIP model used for prompt generation
    """
    
    def __init__(
        self,
        model_id: str = "segmind/SSD-1B",
        device: str = "cuda",
        use_fp16: bool = True,
        use_clip_prompts: bool = True,
        clip_model_id: str = "openai/clip-vit-base-patch32",
    ):
        self.device = device
        # Force fp32 on CPU to avoid CUDA requirements
        self.use_fp16 = use_fp16 and device == "cuda"
        self.use_clip_prompts = use_clip_prompts

        # Load refiner pipeline (SSD-1B uses SDXL architecture)
        dtype = torch.float16 if self.use_fp16 else torch.float32
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            variant="fp16" if self.use_fp16 else None,
        )
        self.pipe.to(device)

        # Memory optimizations
        self.pipe.enable_attention_slicing()

        # Only enable CPU offloading when using CUDA
        if device == "cuda" and hasattr(self.pipe, 'enable_model_cpu_offload'):
            self.pipe.enable_model_cpu_offload()

        # Initialize CLIP for content-aware prompt generation
        if self.use_clip_prompts:
            self.clip_model = CLIPModel.from_pretrained(clip_model_id)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
            self.clip_model.to(device)
            self.clip_model.eval()
        else:
            self.clip_model = None
            self.clip_processor = None

    def _generate_clip_prompt(self, image: Image.Image) -> str:
        """
        Generate a content-aware prompt using CLIP to describe the image.

        This helps preserve image content while improving quality by creating
        prompts that are faithful to the original image.

        Args:
            image: Input PIL Image

        Returns:
            Descriptive prompt based on CLIP analysis
        """
        if not self.use_clip_prompts or self.clip_model is None:
            return "high quality, detailed, sharp"

        # CLIP candidate prompts that focus on quality improvement without content change
        candidate_prompts = [
            "a high quality photograph",
            "a detailed image",
            "a sharp clear image",
            "a well-lit photograph",
            "a professional quality image",
            "an enhanced detailed photograph",
            "a crisp clear image",
            "a refined detailed image",
        ]

        # Process image for CLIP
        inputs = self.clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get image features
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)

        # Process text prompts
        text_inputs = self.clip_processor(
            text=candidate_prompts,
            return_tensors="pt",
            padding=True
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        # Get text features
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**text_inputs)

        # Calculate similarity scores
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).squeeze(0)

        # Get the best matching prompt
        best_idx = similarity.argmax().item()
        base_prompt = candidate_prompts[best_idx]

        
        # Combine with quality enhancement terms
        return f"{base_prompt}, high quality, detailed, sharp focus"

    def refine(
        self,
        image: Union[Image.Image, np.ndarray],
        strength: float = 0.2,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        prompt: Optional[str] = None,
        negative_prompt: str = "blurry, low quality, distorted",
    ) -> Image.Image:
        """
        Refine an input image to improve quality before 3D generation.

        Uses CLIP-generated content-aware prompts by default to preserve image content
        while enhancing quality. Can override with custom prompts if needed.

        Args:
            image: Input PIL Image or numpy array
            strength: How much to transform the image (0.0-1.0)
                     Lower = more faithful to original (default: 0.2 for content preservation)
                     Higher = more refinement but may change content
            guidance_scale: How strongly to follow the prompt
            num_inference_steps: Number of denoising steps (more = better quality but slower)
            prompt: Positive prompt to guide refinement (if None, uses CLIP-generated prompt)
            negative_prompt: Negative prompt to avoid artifacts

        Returns:
            Refined PIL Image
        """
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Generate CLIP-based prompt if none provided
        if prompt is None:
            prompt = self._generate_clip_prompt(image)
            print(f"Generated CLIP-based prompt: {prompt}")

        # Run refinement
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
        
        return result.images[0]
    
    def refine_batch(
        self,
        images: list[Union[Image.Image, np.ndarray]],
        **kwargs
    ) -> list[Image.Image]:
        """
        Refine multiple images (e.g., for multi-view input).
        
        Args:
            images: List of input images
            **kwargs: Arguments passed to refine()
            
        Returns:
            List of refined images
        """
        return [self.refine(img, **kwargs) for img in images]
    
    def unload(self):
        """Free GPU memory by moving models to CPU."""
        if hasattr(self, 'pipe'):
            self.pipe.to('cpu')
        if hasattr(self, 'clip_model') and self.clip_model is not None:
            self.clip_model.to('cpu')
        torch.cuda.empty_cache()
    
    def cleanup(self):
        """
        Fully cleanup and release all resources.
        Use this when you're done with the refiner entirely.
        """
        if hasattr(self, 'pipe'):
            # Move to CPU first
            self.pipe.to('cpu')
            # Delete the pipeline
            del self.pipe
            self.pipe = None

        if hasattr(self, 'clip_model') and self.clip_model is not None:
            # Move CLIP model to CPU first
            self.clip_model.to('cpu')
            # Delete CLIP models
            del self.clip_model
            del self.clip_processor
            self.clip_model = None
            self.clip_processor = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def __del__(self):
        """Destructor to ensure cleanup on deletion."""
        self.cleanup()

