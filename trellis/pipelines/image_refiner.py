"""
Image refinement module for TRELLIS-BOX using SSD-1B (Segmind Stable Diffusion).

This module provides pre-processing image refinement before 3D generation,
improving input quality for better 3D reconstruction results.

SSD-1B is a distilled version of SDXL with 50% less memory usage while maintaining
90-95% of the quality, making it ideal for consumer GPUs.
"""

from typing import Union
import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusionXLImg2ImgPipeline


class ImageRefiner:
    """
    Refines input images using SSD-1B (Segmind Stable Diffusion) before 3D generation.
    
    This improves texture quality, reduces artifacts, and enhances details
    in the input image, leading to better 3D reconstruction quality.
    
    SSD-1B uses ~4-5GB VRAM (vs SDXL's 8-12GB) while maintaining excellent quality.
    
    Args:
        model_id: Hugging Face model ID for the refiner
        device: Device to run the model on ('cuda' or 'cpu')
        use_fp16: Whether to use half precision for memory efficiency (automatically disabled on CPU)
    """
    
    def __init__(
        self,
        model_id: str = "segmind/SSD-1B",
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        self.device = device
        # Force fp32 on CPU to avoid CUDA requirements
        self.use_fp16 = use_fp16 and device == "cuda"

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
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def __del__(self):
        """Destructor to ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass

