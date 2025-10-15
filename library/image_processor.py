"""
Image processing functionality for TRELLIS.
Pure business logic with no UI or state management dependencies.
"""

import uuid
import torch
from typing import List, Tuple, Optional, Any
from PIL import Image

from library.image_refiner import ImageRefiner
from library.models import ProcessingResult


class ImageProcessor:
    """Handles image preprocessing and refinement operations."""

    def __init__(self, pipeline: Optional[Any] = None):
        """Initialize the image processor.

        Args:
            pipeline: TRELLIS pipeline for image preprocessing
        """
        self.pipeline = pipeline
        self.refiner: Optional[ImageRefiner] = None

    def set_pipeline(self, pipeline: Any) -> None:
        """Set the TRELLIS pipeline."""
        self.pipeline = pipeline

    def load_refiner(self) -> None:
        """Load the SSD-1B refiner lazily."""
        if self.refiner is None:
            print("Loading SSD-1B Refiner (Segmind Stable Diffusion)...")
            self.refiner = ImageRefiner(device="cpu", use_fp16=False)

    def unload_refiner(self) -> None:
        """Unload the refiner to free memory."""
        if self.refiner is not None:
            self.refiner.unload()
            self.refiner = None
            torch.cuda.empty_cache()

    def apply_refinement(self, image: Image.Image) -> Image.Image:
        """
        Apply SSD-1B refinement to improve input image quality.

        Args:
            image: Input PIL Image

        Returns:
            Refined PIL Image
        """
        if self.refiner is None:
            self.load_refiner()

        # Apply refinement
        refined_image = self.refiner.refine(
            image,
            strength=0.3,  # Subtle refinement to preserve original
            guidance_scale=7.5,
            num_inference_steps=20,
            prompt="high quality, detailed, sharp, clean",
            negative_prompt="blurry, low quality, distorted, artifacts"
        )
        return refined_image

    def preprocess_single_image(self, image: Image.Image, use_refinement: bool = False, tmp_dir: str = "/tmp/Trellis-demo") -> ProcessingResult:
        """
        Preprocess a single input image with memory-efficient operations.
        Background removal happens first, then refinement if requested.

        Args:
            image: The input image
            use_refinement: Whether to apply SSD-1B refinement after background removal
            tmp_dir: Directory to save processed images

        Returns:
            ProcessingResult with trial_id and processed image
        """
        trial_id = str(uuid.uuid4())

        # Memory-efficient preprocessing with no gradients (background removal first)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                processed_image = self.pipeline.preprocess_image(image)

        # Apply refinement after background removal if requested
        if use_refinement:
            print("Applying image refinement after background removal...")
            processed_image = self.apply_refinement(processed_image)
            # Clean up refiner VRAM before TRELLIS processing
            self.unload_refiner()
            torch.cuda.empty_cache()

        # High-quality image saving - no compression artifacts
        processed_image.save(f"{tmp_dir}/{trial_id}.png", quality=100, subsampling=0)

        return ProcessingResult(trial_id=trial_id, processed_images=processed_image)

    def preprocess_multiple_images(self, images: List[Image.Image], use_refinement: bool = False, tmp_dir: str = "/tmp/Trellis-demo") -> ProcessingResult:
        """
        Preprocess multiple input images for multi-view 3D reconstruction.
        Background removal happens first, then refinement if requested.
        Memory optimization: Process and save images one by one to reduce peak memory usage.

        Args:
            images: List of input images
            use_refinement: Whether to apply SSD-1B refinement after background removal
            tmp_dir: Directory to save processed images

        Returns:
            ProcessingResult with trial_id and list of processed images
        """
        trial_id = str(uuid.uuid4())
        processed_images = []

        # Process images one by one to minimize memory usage
        for i, img in enumerate(images):
            # Background removal first
            processed_img = self.pipeline.preprocess_image(img)

            # Apply refinement after background removal if requested
            if use_refinement:
                print(f"Refining image {i+1}/{len(images)} after background removal...")
                processed_img = self.apply_refinement(processed_img)

            # High-quality image saving for multi-view - no compression artifacts
            processed_img.save(f"{tmp_dir}/{trial_id}_{i}.png", quality=100, subsampling=0)
            processed_images.append(processed_img)

            # Force cleanup of intermediate objects
            del img
            torch.cuda.empty_cache()

        # Clean up refiner after processing all images
        if use_refinement:
            self.unload_refiner()
            torch.cuda.empty_cache()

        return ProcessingResult(trial_id=trial_id, processed_images=processed_images)
