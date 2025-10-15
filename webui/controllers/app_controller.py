"""
Main application controller for TRELLIS.
Coordinates all business logic and manages application state.
"""

import os
import torch
from typing import Optional, Dict, Any, Tuple, List
from PIL import Image

from .memory_controller import MemoryController
from .generation_controller import GenerationController
from library.models import GenerationParams, ExportParams
from webui.initialize_pipeline import load_pipeline
from library.memory_utils import reduce_memory_usage


class AppController:
    """Main application controller coordinating all operations."""

    MAX_SEED = 2147483647  # np.iinfo(np.int32).max

    def __init__(self):
        """Initialize the application controller."""
        self.tmp_dir = os.environ.get("TRELLIS_OUTPUT_DIR", "/tmp/Trellis-demo")
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.memory_controller = MemoryController(self.tmp_dir)
        self.generation_controller = GenerationController(tmp_dir=self.tmp_dir)

        self.pipeline: Optional[Any] = None

    def initialize_pipeline(self) -> Any:
        """
        Initialize or load the TRELLIS pipeline.

        Returns:
            The loaded pipeline
        """
        if self.pipeline is None:
            # Clean CUDA memory before loading
            if torch.cuda.is_available():
                print("Clearing CUDA memory before pipeline initialization...")
                self.memory_controller.reduce_memory()

            self.pipeline = load_pipeline()
            self.generation_controller.set_pipeline(self.pipeline)

        return self.pipeline

    def get_pipeline(self) -> Optional[Any]:
        """Get the current pipeline instance."""
        return self.pipeline

    def check_gpu(self) -> Optional[str]:
        """
        Check for GPU availability and return GPU info.

        Returns:
            GPU info string or None if GPU check fails

        Raises:
            SystemExit: If no CUDA GPU is available
        """
        if not torch.cuda.is_available():
            raise SystemExit("CUDA GPU not detected. TRELLIS requires a CUDA-compatible GPU to run.")

        # Get GPU info
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            return f"{gpu_name} ({gpu_count} GPU{'s' if gpu_count > 1 else ''})"
        else:
            raise SystemExit("CUDA available but no GPUs accessible")

    def periodic_cleanup(self) -> None:
        """Perform periodic memory cleanup."""
        self.memory_controller.periodic_cleanup()

    def create_generation_params(
        self,
        seed: int,
        randomize_seed: bool,
        ss_guidance_strength: float,
        ss_sampling_steps: int,
        slat_guidance_strength: float,
        slat_sampling_steps: int
    ) -> GenerationParams:
        """Create generation parameters object."""
        return GenerationParams(
            seed=seed,
            randomize_seed=randomize_seed,
            ss_guidance_strength=ss_guidance_strength,
            ss_sampling_steps=ss_sampling_steps,
            slat_guidance_strength=slat_guidance_strength,
            slat_sampling_steps=slat_sampling_steps
        )

    def create_export_params(
        self,
        mesh_simplify: float,
        texture_size: int,
        fill_holes_resolution: int = 1024,
        fill_holes_num_views: int = 1000
    ) -> ExportParams:
        """Create export parameters object."""
        return ExportParams(
            mesh_simplify=mesh_simplify,
            texture_size=texture_size,
            fill_holes_resolution=fill_holes_resolution,
            fill_holes_num_views=fill_holes_num_views
        )

    def process_image(self, image: Image.Image, use_refinement: bool = False):
        """
        Process a single image.

        Args:
            image: Input PIL image
            use_refinement: Whether to apply refinement

        Returns:
            ProcessingResult
        """
        return self.generation_controller.process_single_image(image, use_refinement)

    def process_images(self, images: List[Image.Image], use_refinement: bool = False):
        """
        Process multiple images.

        Args:
            images: List of input PIL images
            use_refinement: Whether to apply refinement

        Returns:
            ProcessingResult
        """
        return self.generation_controller.process_multiple_images(images, use_refinement)

    def generate_single(self, trial_id: str, params: GenerationParams, resize_dims: Tuple[int, int] = (518, 518)):
        """
        Generate 3D model from single image.

        Args:
            trial_id: Processing trial ID
            params: Generation parameters
            resize_dims: Image dimensions

        Returns:
            GenerationResult
        """
        self.memory_controller.reduce_memory()
        return self.generation_controller.generate_from_single_image(trial_id, params, resize_dims)

    def generate_multiple(
        self,
        trial_id: str,
        num_images: int,
        params: GenerationParams,
        batch_size: int = 2,
        resize_dims: Tuple[int, int] = (518, 518),
        condition_data: Optional[Dict[str, Any]] = None
    ):
        """
        Generate 3D model from multiple images.

        Args:
            trial_id: Processing trial ID
            num_images: Number of images
            params: Generation parameters
            batch_size: Processing batch size
            resize_dims: Image dimensions
            condition_data: Pre-computed conditioning data

        Returns:
            GenerationResult
        """
        self.memory_controller.reduce_memory()
        return self.generation_controller.generate_from_multiple_images(
            trial_id, num_images, params, batch_size, resize_dims, condition_data
        )

    def export_model(self, model_state, params: ExportParams):
        """
        Export model to GLB.

        Args:
            model_state: Generated model state
            params: Export parameters

        Returns:
            ExportResult
        """
        self.memory_controller.reduce_memory()
        return self.generation_controller.export_glb(model_state, params)

    def get_conditioning(self, images: List[Image.Image], resize_dims: Tuple[int, int] = (518, 518)) -> Dict[str, Any]:
        """
        Get conditioning data for multi-view images.

        Args:
            images: List of PIL images
            resize_dims: Target resize dimensions

        Returns:
            Conditioning data
        """
        return self.generation_controller.get_conditioning_data(images, resize_dims)
