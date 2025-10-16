"""
3D model generation functionality for TRELLIS.
Pure business logic with no UI or state management dependencies.
"""

import uuid
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional
from easydict import EasyDict as edict

from trellis.representations import Gaussian
from trellis.utils import render_utils
from library.models import GenerationParams, ModelState, GenerationResult
from library.video_renderer import VideoRenderer


class ModelGenerator:
    """Handles 3D model generation from images."""

    def __init__(self, tmp_dir: str = "/tmp/Trellis-demo"):
        """Initialize the model generator.

        Args:
            tmp_dir: Directory for temporary files
        """
        self.tmp_dir = tmp_dir

    def pack_model_state(self, gs: Gaussian, mesh: Any, trial_id: str) -> ModelState:
        """
        Pack model state while keeping tensors on GPU to avoid unnecessary CPU transfers.
        Memory optimization: Only move to CPU when actually needed for serialization.

        Args:
            gs: Gaussian splat representation
            mesh: Mesh extraction result
            trial_id: Trial identifier

        Returns:
            Packed ModelState
        """
        return ModelState(
            gaussian_data={
                **gs.init_params,
                '_xyz': gs._xyz,  # Keep on GPU
                '_features_dc': gs._features_dc,  # Keep on GPU
                '_scaling': gs._scaling,  # Keep on GPU
                '_rotation': gs._rotation,  # Keep on GPU
                '_opacity': gs._opacity,  # Keep on GPU
            },
            mesh_data={
                'vertices': mesh.vertices,  # Keep on GPU
                'faces': mesh.faces,  # Keep on GPU
            },
            trial_id=trial_id
        )

    def unpack_model_state(self, state: ModelState) -> Tuple[Gaussian, edict, str]:
        """
        Unpack model state efficiently - tensors are already on GPU from pack_state.
        Memory optimization: Avoid redundant tensor creation and device transfers.

        Args:
            state: Packed model state

        Returns:
            Tuple of (Gaussian, mesh, trial_id)
        """
        gs = Gaussian(
            aabb=state.gaussian_data['aabb'],
            sh_degree=state.gaussian_data['sh_degree'],
            mininum_kernel_size=state.gaussian_data['mininum_kernel_size'],
            scaling_bias=state.gaussian_data['scaling_bias'],
            opacity_bias=state.gaussian_data['opacity_bias'],
            scaling_activation=state.gaussian_data['scaling_activation'],
        )

        # Tensors are already on GPU from pack_state - direct assignment
        gs._xyz = state.gaussian_data['_xyz']
        gs._features_dc = state.gaussian_data['_features_dc']
        gs._scaling = state.gaussian_data['_scaling']
        gs._rotation = state.gaussian_data['_rotation']
        gs._opacity = state.gaussian_data['_opacity']

        mesh = edict(
            vertices=state.mesh_data['vertices'],  # Already on GPU
            faces=state.mesh_data['faces'],  # Already on GPU
        )

        return gs, mesh, state.trial_id

    def generate_from_single_image(
        self,
        pipeline: Any,
        image_path: str,
        params: GenerationParams,
        resize_dims: Tuple[int, int] = (518, 518)
    ) -> GenerationResult:
        """
        Convert a single image to a 3D model.

        Args:
            image_path: Path to the processed image
            params: Generation parameters
            resize_dims: Image resize dimensions

        Returns:
            GenerationResult with model state and video path
        """
        seed = params.seed
        if params.randomize_seed:
            seed = np.random.randint(0, np.iinfo(np.int32).max)

        with Image.open(image_path) as image:
            # Memory optimization: Ensure clean state before inference
            torch.cuda.empty_cache()

            with torch.inference_mode():
                # Critical memory optimization before heavy computation
                torch.cuda.empty_cache()

                outputs = self.pipeline.run(
                    image,
                    seed=seed,
                    formats=["gaussian", "mesh"],
                    preprocess_image=False,
                    target_size=resize_dims,
                    sparse_structure_sampler_params={
                        "steps": params.ss_sampling_steps,
                        "cfg_strength": params.ss_guidance_strength,
                    },
                    slat_sampler_params={
                        "steps": params.slat_sampling_steps,
                        "cfg_strength": params.slat_guidance_strength,
                    },
                )

        # Extract gaussian and mesh for video rendering
        gaussian = outputs['gaussian'][0]
        mesh = outputs['mesh'][0]

        # Clean up outputs immediately after extraction
        del outputs
        torch.cuda.empty_cache()

        # Render videos with reduced frames for memory efficiency
        video_color = render_utils.render_video(gaussian, num_frames=60)['color']
        video_normal = render_utils.render_video(mesh, num_frames=60)['normal']

        # Generate new trial ID for video output
        video_trial_id = str(uuid.uuid4())
        video_path = f"{self.tmp_dir}/{video_trial_id}.mp4"

        # Create video using streaming approach to minimize memory usage
        VideoRenderer.create_side_by_side_video(video_color, video_normal, video_path, fps=15, quality=8)

        # Clean up video arrays immediately after streaming creation
        del video_color, video_normal
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        # Pack state for GLB extraction - keeps tensors on GPU
        state = self.pack_model_state(gaussian, mesh, video_trial_id)

        # Early cleanup - gaussian and mesh are now in state
        del gaussian, mesh
        torch.cuda.empty_cache()

        return GenerationResult(model_state=state, video_path=video_path)

    def generate_from_multiple_images(
        self,
        pipeline: Any,
        trial_id: str,
        num_images: int,
        params: GenerationParams,
        batch_size: int = 2,
        resize_dims: Tuple[int, int] = (518, 518),
        condition_data: Optional[Dict[str, Any]] = None
    ) -> GenerationResult:
        """
        Convert multiple images to a 3D model using multi-view conditioning.

        Args:
            trial_id: Trial identifier for processed images
            num_images: Number of images in this batch
            params: Generation parameters
            batch_size: Number of images to process simultaneously
            resize_dims: Image resize dimensions
            condition_data: Pre-computed conditioning data

        Returns:
            GenerationResult with model state and video path
        """
        torch.cuda.empty_cache()

        seed = params.seed
        if params.randomize_seed:
            seed = np.random.randint(0, np.iinfo(np.int32).max)

        # Adaptive batch sizing based on available memory
        original_batch_size = batch_size
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            free_memory_gb = free_memory / (1024**3)

            # Adaptive batch sizing for memory efficiency
            if free_memory_gb < 8.0:
                batch_size = min(batch_size, 1)  # Very limited memory
                if batch_size != original_batch_size:
                    print(f"Low memory detected ({free_memory_gb:.1f}GB free). Reducing batch size to {batch_size}.")
            elif free_memory_gb < 12.0:
                batch_size = min(batch_size, 2)  # Limited memory
                if batch_size != original_batch_size:
                    print(f"Moderate memory detected ({free_memory_gb:.1f}GB free). Adjusting batch size to {batch_size}.")

        # Ensure batch size doesn't exceed number of images
        batch_size = min(batch_size, num_images)

        # Load all images for multi-view conditioning
        print(f"Loading {num_images} images for multi-view conditioning...")
        images = []
        for j in range(num_images):
            image = Image.open(f"{self.tmp_dir}/{trial_id}_{j}.png")
            images.append(image)

        # Get multi-view conditioning by passing all images at once
        cond = self.pipeline.get_cond(images, target_size=resize_dims)

        # Analyze contradiction for multi-view inputs
        contradiction = self.pipeline.analyze_contradiction(cond)

        # Clean up images immediately
        del images
        torch.cuda.empty_cache()

        torch.manual_seed(seed)

        with torch.inference_mode():
            coords = self.pipeline.sample_sparse_structure(cond, 1, {
                "steps": params.ss_sampling_steps,
                "cfg_strength": params.ss_guidance_strength,
            })

            slat = self.pipeline.sample_slat(cond, coords, {
                "steps": params.slat_sampling_steps,
                "cfg_strength": params.slat_guidance_strength,
            })

            # Clean up conditioning data immediately after sampling
            del cond, coords
            torch.cuda.empty_cache()

            # Critical memory cleanup before mesh decoding
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Decode SLAT without inference mode (rendering needs autograd compatibility)
            outputs = self.pipeline.decode_slat(slat, ["gaussian", "mesh"])
            del slat

        # Extract gaussian and mesh for video rendering
        gaussian = outputs['gaussian'][0]
        mesh = outputs['mesh'][0]

        # Clean up outputs immediately after extraction
        del outputs
        torch.cuda.empty_cache()

        # Render videos with reduced frames for memory efficiency
        video_color = render_utils.render_video(gaussian, num_frames=60)['color']
        video_normal = render_utils.render_video(mesh, num_frames=60)['normal']

        # Generate new trial ID for video output
        video_trial_id = str(uuid.uuid4())
        video_path = f"{self.tmp_dir}/{video_trial_id}.mp4"

        # Create video using streaming approach to minimize memory usage
        VideoRenderer.create_side_by_side_video(video_color, video_normal, video_path, fps=15, quality=8)

        # Clean up video arrays immediately after streaming creation
        del video_color, video_normal
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        # Pack state for GLB extraction - keeps tensors on GPU
        state = self.pack_model_state(gaussian, mesh, video_trial_id)

        # Early cleanup - gaussian and mesh are now in state
        del gaussian, mesh
        torch.cuda.empty_cache()

        return GenerationResult(model_state=state, video_path=video_path)
