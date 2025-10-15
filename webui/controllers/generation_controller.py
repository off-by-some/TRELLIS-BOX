"""
Generation controller for TRELLIS application.
Coordinates image processing, 3D generation, and export operations.
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from PIL import Image

from library.models import (
    GenerationParams, ExportParams, ProcessingResult,
    GenerationResult, ExportResult, ModelState
)
from library.image_processor import ImageProcessor
from library.model_generator import ModelGenerator
from library.glb_exporter import GLBExporter


class GenerationController:
    """Handles the coordination of generation workflows."""

    def __init__(self, pipeline: Optional[Any] = None, tmp_dir: str = "/tmp/Trellis-demo"):
        """Initialize the generation controller.

        Args:
            pipeline: TRELLIS pipeline instance
            tmp_dir: Directory for temporary files
        """
        self.pipeline = pipeline
        self.tmp_dir = tmp_dir

        # Initialize library components
        self.image_processor = ImageProcessor(pipeline=pipeline, tmp_dir=tmp_dir)
        self.model_generator = ModelGenerator(pipeline=pipeline, tmp_dir=tmp_dir)
        self.glb_exporter = GLBExporter(tmp_dir=tmp_dir)

    def set_pipeline(self, pipeline: Any) -> None:
        """Set the TRELLIS pipeline for all components."""
        self.pipeline = pipeline
        self.image_processor.set_pipeline(pipeline)
        self.model_generator.set_pipeline(pipeline)

    def process_single_image(self, image: Image.Image, use_refinement: bool = False) -> ProcessingResult:
        """
        Process a single image for generation.

        Args:
            image: Input PIL image
            use_refinement: Whether to apply refinement

        Returns:
            ProcessingResult with processed image data
        """
        return self.image_processor.preprocess_single_image(image, use_refinement)

    def process_multiple_images(self, images: List[Image.Image], use_refinement: bool = False) -> ProcessingResult:
        """
        Process multiple images for generation.

        Args:
            images: List of input PIL images
            use_refinement: Whether to apply refinement

        Returns:
            ProcessingResult with processed images data
        """
        return self.image_processor.preprocess_multiple_images(images, use_refinement)

    def generate_from_single_image(
        self,
        trial_id: str,
        params: GenerationParams,
        resize_dims: Tuple[int, int] = (518, 518)
    ) -> GenerationResult:
        """
        Generate 3D model from single processed image.

        Args:
            trial_id: Processing trial ID
            params: Generation parameters
            resize_dims: Image dimensions

        Returns:
            GenerationResult with model and video
        """
        image_path = f"{self.tmp_dir}/{trial_id}.png"
        return self.model_generator.generate_from_single_image(image_path, params, resize_dims)

    def generate_from_multiple_images(
        self,
        trial_id: str,
        num_images: int,
        params: GenerationParams,
        batch_size: int = 2,
        resize_dims: Tuple[int, int] = (518, 518),
        condition_data: Optional[Dict[str, Any]] = None
    ) -> GenerationResult:
        """
        Generate 3D model from multiple processed images.

        Args:
            trial_id: Processing trial ID
            num_images: Number of images
            params: Generation parameters
            batch_size: Processing batch size
            resize_dims: Image dimensions
            condition_data: Pre-computed conditioning data

        Returns:
            GenerationResult with model and video
        """
        return self.model_generator.generate_from_multiple_images(
            trial_id, num_images, params, batch_size, resize_dims, condition_data
        )

    def export_glb(self, model_state: ModelState, params: ExportParams) -> ExportResult:
        """
        Export model state to GLB file.

        Args:
            model_state: Generated model state
            params: Export parameters

        Returns:
            ExportResult with GLB path
        """
        return self.glb_exporter.export(model_state, params)

    def get_conditioning_data(self, images: List[Image.Image], resize_dims: Tuple[int, int] = (518, 518)) -> Dict[str, Any]:
        """
        Get conditioning data for multi-view images.

        Args:
            images: List of PIL images
            resize_dims: Target resize dimensions

        Returns:
            Conditioning data dictionary
        """
        cond = self.pipeline.get_cond(images, target_size=resize_dims)
        contradiction = self.pipeline.analyze_contradiction(cond)
        return {
            'cond': cond,
            'contradiction': contradiction,
            'multi_view': True
        }
