"""
Image refinement module for TRELLIS-BOX using Stable Diffusion XL Refiner.

This module provides pre-processing image refinement before 3D generation,
improving input quality for better 3D reconstruction results.
"""

from typing import Union
import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusionXLImg2ImgPipeline


class ImageRefiner:
    """
    Refines input images using Stable Diffusion XL Refiner before 3D generation.
    
    This improves texture quality, reduces artifacts, and enhances details
    in the input image, leading to better 3D reconstruction quality.
    
    Args:
        model_id: Hugging Face model ID for the refiner
        device: Device to run the model on ('cuda' or 'cpu')
        use_fp16: Whether to use half precision for memory efficiency
    """
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        self.device = device
        self.use_fp16 = use_fp16
        
        # Load refiner pipeline
        dtype = torch.float16 if use_fp16 else torch.float32
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            variant="fp16" if use_fp16 else None,
        )
        self.pipe.to(device)
        
        # Memory optimizations for consumer GPUs
        self.pipe.enable_attention_slicing()
        if hasattr(self.pipe, 'enable_model_cpu_offload'):
            self.pipe.enable_model_cpu_offload()
    
    def refine(
        self,
        image: Union[Image.Image, np.ndarray],
        strength: float = 0.3,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        prompt: str = "high quality, detailed, sharp",
        negative_prompt: str = "blurry, low quality, distorted",
    ) -> Image.Image:
        """
        Refine an input image to improve quality before 3D generation.
        
        Args:
            image: Input PIL Image or numpy array
            strength: How much to transform the image (0.0-1.0)
                     Lower = more faithful to original
                     Higher = more refinement but may change content
            guidance_scale: How strongly to follow the prompt
            num_inference_steps: Number of denoising steps (more = better quality but slower)
            prompt: Positive prompt to guide refinement
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
        """Free GPU memory by moving model to CPU."""
        if hasattr(self, 'pipe'):
            self.pipe.to('cpu')
            torch.cuda.empty_cache()

