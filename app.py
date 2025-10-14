import streamlit as st

import time
import os
import warnings
from webui.loading_screen import show_loading_screen, finalize_loading
from webui.initialize_pipeline import load_pipeline, reduce_memory_usage
from webui.ui_components import show_video_preview, show_3d_model_viewer, show_example_gallery
from webui.image_preview import image_preview

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*deprecated.*")
warnings.filterwarnings("ignore", message=".*xFormers is available.*")
warnings.filterwarnings("ignore", message=".*torch.library.impl_abstract.*renamed.*")
warnings.filterwarnings("ignore", message=".*torch.library.register_fake.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['SPCONV_ALGO'] = 'native'
# Memory optimizations for Trellis workloads
# Use backend_alloc:cudaMallocAsync for better memory management with PyTorch 2.0+
# Avoid expandable_segments due to compatibility issues with certain operations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,garbage_collection_threshold:0.8,roundup_power2_divisions:16'
# CUDA optimizations
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Non-blocking launches

# Suppress common library warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# Suppress transformers cache warnings
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

from typing import Optional, List, Tuple, Dict, Any
import torch
import numpy as np
import imageio
import uuid
import gc
from dataclasses import dataclass
from easydict import EasyDict as edict
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.pipelines.image_refiner import ImageRefiner
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.environ.get("TRELLIS_OUTPUT_DIR", "/tmp/Trellis-demo")

os.makedirs(TMP_DIR, exist_ok=True)


# ============================================================================
# Data Classes for Type Safety
# ============================================================================

@dataclass
class GenerationParams:
    """Parameters for 3D model generation."""
    seed: int
    randomize_seed: bool
    ss_guidance_strength: float
    ss_sampling_steps: int
    slat_guidance_strength: float
    slat_sampling_steps: int


@dataclass
class ExportParams:
    """Parameters for GLB export."""
    mesh_simplify: float
    texture_size: int


@dataclass
class ModelState:
    """State of a generated 3D model."""
    gaussian_data: Dict[str, Any]
    mesh_data: Dict[str, Any]
    trial_id: str


# ============================================================================
# TODO 1: State Manager - Handles all session state management
# ============================================================================

class StateManager:
    """Manages Streamlit session state with type safety."""
    
    # State keys
    PIPELINE = 'pipeline'
    REFINER = 'refiner'
    UPLOADED_IMAGE = 'uploaded_image'
    PROCESSED_PREVIEW = 'processed_preview'
    GENERATED_VIDEO = 'generated_video'
    GENERATED_GLB = 'generated_glb'
    GENERATED_STATE = 'generated_state'
    CLEANUP_COUNTER = 'cleanup_counter'
    IS_GENERATING = 'is_generating'
    
    @staticmethod
    def initialize() -> None:
        """Initialize all required session state variables."""
        defaults = {
            StateManager.PIPELINE: None,
            StateManager.REFINER: None,
            StateManager.UPLOADED_IMAGE: None,
            StateManager.PROCESSED_PREVIEW: None,
            StateManager.GENERATED_VIDEO: None,
            StateManager.GENERATED_GLB: None,
            StateManager.GENERATED_STATE: None,
            StateManager.CLEANUP_COUNTER: 0,
            StateManager.IS_GENERATING: False,
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def get_pipeline() -> Optional[TrellisImageTo3DPipeline]:
        """Get the pipeline from session state."""
        return st.session_state.get(StateManager.PIPELINE)
    
    @staticmethod
    def set_pipeline(pipeline: TrellisImageTo3DPipeline) -> None:
        """Set the pipeline in session state."""
        st.session_state[StateManager.PIPELINE] = pipeline
    
    @staticmethod
    def get_refiner() -> Optional[ImageRefiner]:
        """Get the refiner from session state."""
        return st.session_state.get(StateManager.REFINER)
    
    @staticmethod
    def set_refiner(refiner: Optional[ImageRefiner]) -> None:
        """Set the refiner in session state."""
        st.session_state[StateManager.REFINER] = refiner
    
    @staticmethod
    def get_uploaded_image() -> Optional[Image.Image]:
        """Get the uploaded image."""
        return st.session_state.get(StateManager.UPLOADED_IMAGE)
    
    @staticmethod
    def set_uploaded_image(image: Optional[Image.Image]) -> None:
        """Set the uploaded image."""
        st.session_state[StateManager.UPLOADED_IMAGE] = image
    
    @staticmethod
    def get_generated_video() -> Optional[str]:
        """Get the generated video path."""
        return st.session_state.get(StateManager.GENERATED_VIDEO)
    
    @staticmethod
    def set_generated_video(video_path: Optional[str]) -> None:
        """Set the generated video path."""
        st.session_state[StateManager.GENERATED_VIDEO] = video_path
    
    @staticmethod
    def get_generated_glb() -> Optional[str]:
        """Get the generated GLB path."""
        return st.session_state.get(StateManager.GENERATED_GLB)
    
    @staticmethod
    def set_generated_glb(glb_path: Optional[str]) -> None:
        """Set the generated GLB path."""
        st.session_state[StateManager.GENERATED_GLB] = glb_path
    
    @staticmethod
    def get_generated_state() -> Optional[Dict[str, Any]]:
        """Get the generated model state."""
        return st.session_state.get(StateManager.GENERATED_STATE)
    
    @staticmethod
    def set_generated_state(state: Optional[Dict[str, Any]]) -> None:
        """Set the generated model state."""
        st.session_state[StateManager.GENERATED_STATE] = state
    
    @staticmethod
    def clear_generated_content() -> None:
        """Clear all generated content from session state."""
        keys_to_clear = [
            StateManager.GENERATED_VIDEO,
            StateManager.GENERATED_GLB,
            StateManager.GENERATED_STATE,
            StateManager.UPLOADED_IMAGE,
            StateManager.PROCESSED_PREVIEW,
        ]
        for key in keys_to_clear:
            st.session_state[key] = None
    
    @staticmethod
    def increment_cleanup_counter() -> int:
        """Increment and return the cleanup counter."""
        counter = st.session_state.get(StateManager.CLEANUP_COUNTER, 0)
        counter += 1
        if counter > 1000:
            counter = 0
        st.session_state[StateManager.CLEANUP_COUNTER] = counter
        return counter
    
    @staticmethod
    def is_generating() -> bool:
        """Check if a generation is currently in progress."""
        return st.session_state.get(StateManager.IS_GENERATING, False)
    
    @staticmethod
    def set_generating(generating: bool) -> None:
        """Set the generation state."""
        st.session_state[StateManager.IS_GENERATING] = generating


# ============================================================================
# TODO 2: Memory Manager - Handles memory cleanup and optimization
# ============================================================================

class MemoryManager:
    """Manages memory cleanup and optimization."""
    
    @staticmethod
    def cleanup_session_state(clear_all: bool = False) -> None:
        """
        Clean up large objects from session state to prevent memory leaks.
        
        Args:
            clear_all: If True, clear all generated content. If False, only clear old cached data.
        """
        if clear_all:
            StateManager.clear_generated_content()
        
        # Always clean up image preview states that can accumulate
        preview_keys = [k for k in st.session_state.keys() if k.startswith('_image_preview_')]
        for key in preview_keys:
            if isinstance(st.session_state[key], dict):
                state = st.session_state[key]
                # Reset render count periodically to prevent integer overflow
                if state.get('render_count', 0) > 1000:
                    state['render_count'] = 0
        
        # Cleanup pipeline resources
        pipeline = StateManager.get_pipeline()
        if pipeline is not None:
            try:
                pipeline.cleanup()
            except Exception as e:
                print(f"Error during pipeline cleanup: {e}")
        
        # Cleanup refiner if loaded
        refiner = StateManager.get_refiner()
        if refiner is not None:
            try:
                refiner.unload()
            except Exception as e:
                print(f"Error during refiner cleanup: {e}")
        
        # Force garbage collection and CUDA cleanup
        reduce_memory_usage()
    
    @staticmethod
    def periodic_cleanup() -> None:
        """
        Perform periodic cleanup to prevent memory accumulation.
        Should be called regularly (e.g., every few generations).
        """
        counter = StateManager.increment_cleanup_counter()
        
        # Perform lightweight cleanup every 5 interactions
        if counter % 5 == 0:
            MemoryManager.cleanup_session_state(clear_all=False)
        
        # Perform more aggressive cleanup every 20 interactions
        if counter % 20 == 0:
            print(f"Performing periodic cleanup (interaction #{counter})")
            
            # Clear old temp files
            from webui.initialize_pipeline import cleanup_temp_files
            cleanup_temp_files(max_age_hours=1)
    
    @staticmethod
    def reduce_memory() -> None:
        """Force garbage collection and CUDA cache cleanup."""
        reduce_memory_usage()


# ============================================================================
# TODO 3: Image Processor - Handles image preprocessing and refinement
# ============================================================================

class ImageProcessor:
    """Handles image preprocessing and refinement operations."""
    
    def __init__(self):
        """Initialize the image processor."""
        pass
    
    @staticmethod
    def apply_refinement(image: Image.Image) -> Image.Image:
        """
        Apply SSD-1B (Segmind Stable Diffusion) refinement to improve input image quality.
        Loads refiner lazily and unloads after use to conserve VRAM.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Refined PIL Image
        """
        refiner = StateManager.get_refiner()
        
        if refiner is None:
            try:
                print("Loading SSD-1B Refiner (Segmind Stable Diffusion)...")
                refiner = ImageRefiner(device="cuda", use_fp16=True)
                StateManager.set_refiner(refiner)
            except Exception as e:
                print(f"Failed to load refiner: {e}, skipping refinement")
                return image
        
        # Apply refinement
        if refiner is not None:
            try:
                refined_image = refiner.refine(
                    image,
                    strength=0.3,  # Subtle refinement to preserve original
                    guidance_scale=7.5,
                    num_inference_steps=20,
                    prompt="high quality, detailed, sharp, clean",
                    negative_prompt="blurry, low quality, distorted, artifacts"
                )
                return refined_image
            except Exception as e:
                print(f"Refinement failed: {e}, using original image")
                return image
        
        return image
    
    @staticmethod
    def preprocess_single_image(image: Image.Image, use_refinement: bool = False) -> Tuple[str, Image.Image]:
        """
        Preprocess a single input image with memory-efficient operations.
        
        Args:
            image: The input image
            use_refinement: Whether to apply SSD-1B refinement
            
        Returns:
            Tuple of (trial_id, processed_image)
        """
        trial_id = str(uuid.uuid4())
        
        # Apply refinement if requested
        if use_refinement:
            print("Applying image refinement...")
            image = ImageProcessor.apply_refinement(image)
            # Clean up refiner VRAM before TRELLIS processing
            refiner = StateManager.get_refiner()
            if refiner is not None:
                refiner.unload()
            torch.cuda.empty_cache()
        
        # Memory-efficient preprocessing with no gradients
        pipeline = StateManager.get_pipeline()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                processed_image = pipeline.preprocess_image(image)
        
        # High-quality image saving - no compression artifacts
        processed_image.save(f"{TMP_DIR}/{trial_id}.png", quality=100, subsampling=0)
        
        return trial_id, processed_image
    
    @staticmethod
    def preprocess_multiple_images(images: List[Image.Image], use_refinement: bool = False) -> Tuple[str, List[Image.Image]]:
        """
        Preprocess multiple input images for multi-view 3D reconstruction.
        Memory optimization: Process and save images one by one to reduce peak memory usage.
        
        Args:
            images: List of input images
            use_refinement: Whether to apply SSD-1B refinement
            
        Returns:
            Tuple of (trial_id, processed_images)
        """
        trial_id = str(uuid.uuid4())
        processed_images = []
        
        # Process images one by one to minimize memory usage
        pipeline = StateManager.get_pipeline()
        for i, img in enumerate(images):
            # Apply refinement if requested
            if use_refinement:
                print(f"Refining image {i+1}/{len(images)}...")
                img = ImageProcessor.apply_refinement(img)
            
            processed_img = pipeline.preprocess_image(img)
            # High-quality image saving for multi-view - no compression artifacts
            processed_img.save(f"{TMP_DIR}/{trial_id}_{i}.png", quality=100, subsampling=0)
            processed_images.append(processed_img)
            
            # Force cleanup of intermediate objects
            del img
            torch.cuda.empty_cache()
        
        # Clean up refiner after processing all images
        if use_refinement:
            refiner = StateManager.get_refiner()
            if refiner is not None:
                refiner.unload()
                torch.cuda.empty_cache()
        
        return trial_id, processed_images


# ============================================================================
# TODO 4: Video Renderer - Handles video creation
# ============================================================================

class VideoRenderer:
    """Handles video rendering operations."""
    
    @staticmethod
    def _convert_frame_to_uint8(frame: Any) -> np.ndarray:
        """
        Convert frame to uint8 numpy array, handling both tensors and arrays.
        
        Args:
            frame: Frame as tensor or numpy array
            
        Returns:
            uint8 numpy array
        """
        if hasattr(frame, 'cpu'):
            # Tensor: (C, H, W) -> (H, W, C), scale to [0, 255]
            frame = frame.detach().cpu().numpy().transpose(1, 2, 0) * 255
            return np.clip(frame, 0, 255).astype(np.uint8)
        else:
            # Already numpy: handle NaN/Inf and ensure uint8
            frame = np.nan_to_num(frame, nan=0, posinf=255, neginf=0)
            return np.asarray(frame, dtype=np.uint8)
    
    @staticmethod
    def create_side_by_side_video(
        color_frames: List[Any],
        normal_frames: List[Any],
        output_path: str,
        fps: int = 15,
        quality: int = 8
    ) -> None:
        """
        Create side-by-side video of color and normal renderings.
        
        Args:
            color_frames: List of color rendered frames
            normal_frames: List of normal map frames
            output_path: Output video file path
            fps: Frames per second
            quality: Video quality (1-10, higher is better)
        """
        # Process and combine all frames
        combined_frames = [
            np.concatenate([
                VideoRenderer._convert_frame_to_uint8(color),
                VideoRenderer._convert_frame_to_uint8(normal)
            ], axis=1)
            for color, normal in zip(color_frames, normal_frames)
        ]
        
        imageio.mimsave(output_path, combined_frames, fps=fps, quality=quality)
        del combined_frames


# ============================================================================
# TODO 5: Model Generator - Handles 3D model generation logic
# ============================================================================

class ModelGenerator:
    """Handles 3D model generation from images."""
    
    @staticmethod
    def pack_model_state(gs: Gaussian, mesh: MeshExtractResult, trial_id: str) -> Dict[str, Any]:
        """
        Pack model state while keeping tensors on GPU to avoid unnecessary CPU transfers.
        Memory optimization: Only move to CPU when actually needed for serialization.
        
        Args:
            gs: Gaussian splat representation
            mesh: Mesh extraction result
            trial_id: Trial identifier
            
        Returns:
            Dictionary containing packed model state
        """
        return {
            'gaussian': {
                **gs.init_params,
                '_xyz': gs._xyz,  # Keep on GPU
                '_features_dc': gs._features_dc,  # Keep on GPU
                '_scaling': gs._scaling,  # Keep on GPU
                '_rotation': gs._rotation,  # Keep on GPU
                '_opacity': gs._opacity,  # Keep on GPU
            },
            'mesh': {
                'vertices': mesh.vertices,  # Keep on GPU
                'faces': mesh.faces,  # Keep on GPU
            },
            'trial_id': trial_id,
        }
    
    @staticmethod
    def unpack_model_state(state: Dict[str, Any]) -> Tuple[Gaussian, edict, str]:
        """
        Unpack model state efficiently - tensors are already on GPU from pack_state.
        Memory optimization: Avoid redundant tensor creation and device transfers.
        
        Args:
            state: Packed model state dictionary
            
        Returns:
            Tuple of (Gaussian, mesh, trial_id)
        """
        gs = Gaussian(
            aabb=state['gaussian']['aabb'],
            sh_degree=state['gaussian']['sh_degree'],
            mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
            scaling_bias=state['gaussian']['scaling_bias'],
            opacity_bias=state['gaussian']['opacity_bias'],
            scaling_activation=state['gaussian']['scaling_activation'],
        )
        
        # Tensors are already on GPU from pack_state - direct assignment
        gs._xyz = state['gaussian']['_xyz']
        gs._features_dc = state['gaussian']['_features_dc']
        gs._scaling = state['gaussian']['_scaling']
        gs._rotation = state['gaussian']['_rotation']
        gs._opacity = state['gaussian']['_opacity']
        
        mesh = edict(
            vertices=state['mesh']['vertices'],  # Already on GPU
            faces=state['mesh']['faces'],  # Already on GPU
        )
        
        return gs, mesh, state['trial_id']
    
    @staticmethod
    def generate_from_single_image(
        trial_id: str,
        params: GenerationParams
    ) -> Tuple[Dict[str, Any], str]:
        """
        Convert a single image to a 3D model.
        
        Args:
            trial_id: The uuid of the trial
            params: Generation parameters
            
        Returns:
            Tuple of (model_state, video_path)
        """
        MemoryManager.reduce_memory()
        
        seed = params.seed
        if params.randomize_seed:
            seed = np.random.randint(0, MAX_SEED)
        
        pipeline = StateManager.get_pipeline()
        
        with Image.open(f"{TMP_DIR}/{trial_id}.png") as image:
            # Memory optimization: Ensure clean state before inference
            MemoryManager.reduce_memory()
            
            with torch.inference_mode():
                # Critical memory optimization before heavy computation
                MemoryManager.reduce_memory()
                
                outputs = pipeline.run(
                    image,
                    seed=seed,
                    formats=["gaussian", "mesh"],
                    preprocess_image=False,
                    target_size=(st.session_state.get("resize_width", 518), st.session_state.get("resize_height", 518)),
                    sparse_structure_sampler_params={
                        "steps": params.ss_sampling_steps,
                        "cfg_strength": params.ss_guidance_strength,
                    },
                    slat_sampler_params={
                        "steps": params.slat_sampling_steps,
                        "cfg_strength": params.slat_guidance_strength,
                    },
                )
            
            # Clean up immediately after inference
            del image
            MemoryManager.reduce_memory()
        
        # Extract gaussian and mesh for video rendering
        gaussian = outputs['gaussian'][0]
        mesh = outputs['mesh'][0]
        
        # Clean up outputs immediately after extraction
        del outputs
        MemoryManager.reduce_memory()
        
        # Render videos with reduced frames for memory efficiency
        video_color = render_utils.render_video(gaussian, num_frames=60)['color']
        video_normal = render_utils.render_video(mesh, num_frames=60)['normal']
        
        # Generate new trial ID for video output
        video_trial_id = str(uuid.uuid4())
        video_path = f"{TMP_DIR}/{video_trial_id}.mp4"
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
        # Create video using streaming approach to minimize memory usage
        VideoRenderer.create_side_by_side_video(video_color, video_normal, video_path, fps=15, quality=8)
        
        # Clean up video arrays immediately after streaming creation
        del video_color, video_normal
        torch.cuda.empty_cache()
        gc.collect()
        
        # Pack state for GLB extraction - keeps tensors on GPU
        state = ModelGenerator.pack_model_state(gaussian, mesh, video_trial_id)
        
        # Early cleanup - gaussian and mesh are now in state
        del gaussian, mesh
        MemoryManager.reduce_memory()
        
        return state, video_path
    
    @staticmethod
    def generate_from_multiple_images(
        trial_id: str,
        num_images: int,
        batch_size: int,
        params: GenerationParams
    ) -> Tuple[Dict[str, Any], str]:
        """
        Convert multiple images to a 3D model using multi-view conditioning.
        
        Args:
            trial_id: The uuid of the trial
            num_images: Number of images in this batch
            batch_size: Number of images to process simultaneously
            params: Generation parameters
            
        Returns:
            Tuple of (model_state, video_path)
        """
        MemoryManager.reduce_memory()
        
        seed = params.seed
        if params.randomize_seed:
            seed = np.random.randint(0, MAX_SEED)
        
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
            image = Image.open(f"{TMP_DIR}/{trial_id}_{j}.png")
            images.append(image)
        
        # Get multi-view conditioning by passing all images at once
        pipeline = StateManager.get_pipeline()
        with torch.inference_mode():
            cond = pipeline.get_cond(images, target_size=(st.session_state.get("resize_width", 518), st.session_state.get("resize_height", 518)))

            # Analyze contradiction for multi-view inputs
            contradiction = pipeline.analyze_contradiction(cond)

        # Display contradiction score for multi-view inputs
        if cond.get('multi_view', False):
            if contradiction < 1.0:
                st.success(f"Multi-view consistency: Good ({contradiction:.2f})")
            elif contradiction < 3.0:
                st.warning(f"Multi-view consistency: Moderate ({contradiction:.2f})")
            else:
                st.error(f"Multi-view consistency: Poor ({contradiction:.2f})")
        
        # Clean up images immediately
        del images
        MemoryManager.reduce_memory()
        
        torch.manual_seed(seed)
        
        with torch.inference_mode():
            coords = pipeline.sample_sparse_structure(cond, 1, {
                "steps": params.ss_sampling_steps,
                "cfg_strength": params.ss_guidance_strength,
            })
            
            slat = pipeline.sample_slat(cond, coords, {
                "steps": params.slat_sampling_steps,
                "cfg_strength": params.slat_guidance_strength,
            })
            
            # Clean up conditioning data immediately after sampling
            del cond, coords
            MemoryManager.reduce_memory()
            
            # Critical memory cleanup before mesh decoding
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Decode SLAT without inference mode (rendering needs autograd compatibility)
            outputs = pipeline.decode_slat(slat, ["gaussian", "mesh"])
            del slat
        
        # Extract gaussian and mesh for video rendering
        gaussian = outputs['gaussian'][0]
        mesh = outputs['mesh'][0]
        
        # Clean up outputs immediately after extraction
        del outputs
        MemoryManager.reduce_memory()
        
        # Render videos with reduced frames for memory efficiency
        video_color = render_utils.render_video(gaussian, num_frames=60)['color']
        video_normal = render_utils.render_video(mesh, num_frames=60)['normal']
        
        # Generate new trial ID for video output
        video_trial_id = str(uuid.uuid4())
        video_path = f"{TMP_DIR}/{video_trial_id}.mp4"
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
        # Create video using streaming approach to minimize memory usage
        VideoRenderer.create_side_by_side_video(video_color, video_normal, video_path, fps=15, quality=8)
        
        # Clean up video arrays immediately after streaming creation
        del video_color, video_normal
        torch.cuda.empty_cache()
        gc.collect()
        
        # Pack state for GLB extraction - keeps tensors on GPU
        state = ModelGenerator.pack_model_state(gaussian, mesh, video_trial_id)
        
        # Early cleanup - gaussian and mesh are now in state
        del gaussian, mesh
        MemoryManager.reduce_memory()
        
        return state, video_path


# ============================================================================
# TODO 6: GLB Exporter - Handles GLB file extraction
# ============================================================================

class GLBExporter:
    """Handles GLB file export operations."""
    
    @staticmethod
    def extract(state: Dict[str, Any], params: ExportParams) -> Tuple[str, str]:
        """
        Extract a GLB file from the 3D model.
        
        Args:
            state: The state of the generated 3D model
            params: Export parameters
            
        Returns:
            Tuple of (glb_path, glb_path) for consistency
        """
        # Aggressive memory cleanup before GLB extraction
        MemoryManager.reduce_memory()
        
        # Unpack state into gaussian splats and mesh
        gs, mesh, trial_id = ModelGenerator.unpack_model_state(state)
        
        # Clear state dict immediately as we've unpacked it
        del state
        MemoryManager.reduce_memory()
        
        # Generate GLB (postprocessing may need autograd compatibility)
        glb = postprocessing_utils.to_glb(
            gs, mesh,
            simplify=params.mesh_simplify,
            texture_size=params.texture_size,
            verbose=False
        )
        
        # Save GLB file
        glb_path = f"{TMP_DIR}/{trial_id}.glb"
        glb.export(glb_path)
        
        # Clean up large objects immediately
        del gs, mesh, glb
        torch.cuda.empty_cache()
        gc.collect()
        
        return glb_path, glb_path


# ============================================================================
# TODO 7: Single Image UI - Handles single image interface
# ============================================================================

class SingleImageUI:
    """Handles the single image generation UI."""
    
    @staticmethod
    def render() -> None:
        """Render the single image generation interface."""
        st.header("Single Image Generation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            SingleImageUI._render_input_column()
        
        with col2:
            SingleImageUI._render_output_column()
        
        # Examples section
        SingleImageUI._render_examples()
    
    @staticmethod
    def _render_input_column() -> None:
        """Render the input column."""
        st.subheader("Input")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg"],
            key="single_image",
            label_visibility="visible"
        )
        
        # Handle uploaded image
        if uploaded_file is not None:
            new_image = Image.open(uploaded_file)
            current_image = StateManager.get_uploaded_image()
            
            if current_image is None or current_image != new_image:
                StateManager.set_uploaded_image(new_image)
                # Reset all generated content when new image is uploaded
                st.session_state.processed_preview = None
                StateManager.set_generated_video(None)
                StateManager.set_generated_glb(None)
                StateManager.set_generated_state(None)
                # Force cleanup when switching images
                MemoryManager.cleanup_session_state(clear_all=False)
                st.rerun()
        
        # Show uploaded image
        uploaded_image = StateManager.get_uploaded_image()
        if uploaded_image is not None:
            st.markdown("**Uploaded Image:**")
            st.image(uploaded_image, use_container_width=True)
            
            # Auto-process and show background-removed preview
            pipeline = StateManager.get_pipeline()
            if pipeline is not None:
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    processed_image = pipeline.preprocess_image(uploaded_image)
                
                st.markdown("**Background Removed (Auto):**")
                st.image(processed_image, use_container_width=True)
                
                # Store preview but clean up old reference first
                if 'processed_preview' in st.session_state and st.session_state.processed_preview is not None:
                    old_preview = st.session_state.processed_preview
                    del old_preview
                st.session_state.processed_preview = processed_image
            else:
                st.info("Background removal preview will be shown after pipeline loads")
                st.session_state.processed_preview = None
    
    @staticmethod
    def _render_generation_panel(
        uploaded_data: Any,
        is_multi_image: bool,
        video_key: str,
        glb_key: str,
        download_key: str,
        generate_key: str,
        seed_key: str,
        randomize_key: str,
        refinement_key: str,
        ss_strength_key: str,
        ss_steps_key: str,
        slat_strength_key: str,
        slat_steps_key: str,
        simplify_key: str,
        texture_key: str,
        batch_size_key: str = None,
        trial_id: str = None
    ) -> None:
        """
        Render complete generation panel with settings, button, and output preview.
        This is a fully reusable component for both single and multi-image tabs.

        Args:
            uploaded_data: Uploaded image or list of images
            is_multi_image: True for multi-image, False for single-image
            trial_id: Unique identifier for this trial (used for UI keys)
            video_key: Unique key for video clear button
            glb_key: Unique key for GLB clear button
            download_key: Unique key for download button
            generate_key: Unique key for generate button
            seed_key: Unique key for seed slider
            randomize_key: Unique key for randomize checkbox
            refinement_key: Unique key for refinement checkbox
            ss_strength_key: Unique key for sparse structure guidance strength
            ss_steps_key: Unique key for sparse structure sampling steps
            slat_strength_key: Unique key for slat guidance strength
            slat_steps_key: Unique key for slat sampling steps
            simplify_key: Unique key for mesh simplify slider
            texture_key: Unique key for texture size slider
            batch_size_key: Unique key for batch size slider (multi-image only)
        """
        # Generate trial_id if not provided
        if trial_id is None:
            import uuid
            trial_id = str(uuid.uuid4())

        # Check if we have valid input
        has_input = False
        if is_multi_image:
            has_input = uploaded_data and len(uploaded_data) >= 2
        else:
            has_input = uploaded_data is not None
        
        # Only show generation settings if we have input
        if has_input:
            # Generation Settings
            with st.expander("Generation Settings", expanded=True):
                seed = st.slider("Seed", 0, MAX_SEED, 0, 1, key=seed_key)
                randomize_seed = st.checkbox("Randomize Seed", value=True, key=randomize_key)
                use_refinement = st.checkbox(
                    "Image Refinement (SSD-1B)",
                    value=False,
                    help="Enhance input quality with SSD-1B - 50% less VRAM than SDXL (adds ~5-7s" + (" per image)" if is_multi_image else ")"),
                    key=refinement_key
                )

                # Resize dimensions for conditioning model
                st.markdown("**Resize Dimensions**")
                col1, col2 = st.columns(2)
                with col1:
                    resize_width = st.number_input(
                        "Width",
                        min_value=256,
                        max_value=1024,
                        value=518,
                        step=64,
                        help="Width to resize images to for conditioning model",
                        key=f"resize_width_{trial_id}"
                    )
                    # Store in session state for use in generation
                    st.session_state["resize_width"] = resize_width
                with col2:
                    resize_height = st.number_input(
                        "Height",
                        min_value=256,
                        max_value=1024,
                        value=518,
                        step=64,
                        help="Height to resize images to for conditioning model",
                        key=f"resize_height_{trial_id}"
                    )
                    # Store in session state for use in generation
                    st.session_state["resize_height"] = resize_height

                # Batch size for multi-image only
                if is_multi_image and batch_size_key:
                    batch_size = st.slider(
                        "Batch Size", 1, 4, 2, 1,
                        help="Number of images processed at once (lower = less memory)",
                        key=batch_size_key
                    )
                
                st.markdown("**Stage 1: Sparse Structure Generation**")
                ss_col1, ss_col2 = st.columns(2)
                with ss_col1:
                    ss_guidance_strength = st.slider("Guidance Strength", 0.0, 10.0, 7.5, 0.1, key=ss_strength_key)
                with ss_col2:
                    ss_sampling_steps = st.slider("Sampling Steps", 1, 50, 12, 1, key=ss_steps_key)
                
                st.markdown("**Stage 2: Structured Latent Generation**")
                slat_col1, slat_col2 = st.columns(2)
                with slat_col1:
                    slat_guidance_strength = st.slider("Guidance Strength", 0.0, 10.0, 3.0, 0.1, key=slat_strength_key)
                with slat_col2:
                    slat_sampling_steps = st.slider("Sampling Steps", 1, 50, 12, 1, key=slat_steps_key)
            
            # GLB Export Settings
            with st.expander("GLB Export Settings", expanded=False):
                mesh_simplify = st.slider("Simplify", 0.9, 0.98, 0.95, 0.01, key=simplify_key)
                texture_size = st.slider("Texture Size", 512, 2048, 1024, 512, key=texture_key)
            
            # Generate/Regenerate button
            is_generating = StateManager.is_generating()
            has_generated = StateManager.get_generated_video() is not None
            
            if is_multi_image:
                button_label = "🔄 Regenerate 3D Model" if has_generated else "Generate 3D Model from Multiple Views"
            else:
                button_label = "🔄 Regenerate 3D Model" if has_generated else "Generate 3D Model"
            
            button_disabled = is_generating or (is_multi_image and len(uploaded_data) < 2)
            
            if st.button(button_label, type="primary", key=generate_key, use_container_width=True, disabled=button_disabled):
                try:
                    StateManager.set_generating(True)
                    
                    # Clear processed preview on regeneration to respect current refinement setting
                    if has_generated and not is_multi_image:
                        st.session_state.processed_preview = None
                    
                    if is_multi_image:
                        # Multi-image generation
                        with st.spinner("Processing multiple images..."):
                            images = [Image.open(f) for f in uploaded_data]
                            
                            if use_refinement:
                                st.info("Applying image refinement to all images...")
                                images = [ImageProcessor.apply_refinement(img) for img in images]
                            
                            trial_id, processed_images = ImageProcessor.preprocess_multiple_images(
                                images,
                                use_refinement
                            )
                            
                            params = GenerationParams(
                                seed=seed if not randomize_seed else np.random.randint(0, MAX_SEED),
                                randomize_seed=randomize_seed,
                                ss_guidance_strength=ss_guidance_strength,
                                ss_sampling_steps=ss_sampling_steps,
                                slat_guidance_strength=slat_guidance_strength,
                                slat_sampling_steps=slat_sampling_steps
                            )
                            
                            state, video_path = ModelGenerator.generate_from_multiple_images(
                                trial_id,
                                len(processed_images),
                                batch_size,
                                params
                            )
                    else:
                        # Single-image generation
                        with st.spinner("Generating 3D model..."):
                            if st.session_state.get('processed_preview') is not None:
                                processed_image = st.session_state.processed_preview
                                trial_id = str(uuid.uuid4())
                                processed_image.save(f"{TMP_DIR}/{trial_id}.png", quality=100, subsampling=0)
                            else:
                                trial_id, processed_image = ImageProcessor.preprocess_single_image(
                                    uploaded_data,
                                    use_refinement
                                )
                            
                            params = GenerationParams(
                                seed=seed if not randomize_seed else np.random.randint(0, MAX_SEED),
                                randomize_seed=randomize_seed,
                                ss_guidance_strength=ss_guidance_strength,
                                ss_sampling_steps=ss_sampling_steps,
                                slat_guidance_strength=slat_guidance_strength,
                                slat_sampling_steps=slat_sampling_steps
                            )
                            
                            state, video_path = ModelGenerator.generate_from_single_image(trial_id, params)
                    
                    # Exit spinner context before setting state and rerunning
                    StateManager.set_generated_glb(None)
                    StateManager.set_generated_video(video_path)
                    StateManager.set_generated_state(state)
                    
                    StateManager.set_generating(False)
                    st.rerun()
                except Exception as e:
                    StateManager.set_generating(False)
                    st.error(f"❌ Generation failed: {str(e)}")
                    st.warning("Try reducing image size or restarting the application if memory errors persist.")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
        
        # Output preview (always shown)
        SingleImageUI._render_output_preview(video_key, glb_key, download_key)
    
    @staticmethod
    def _render_output_preview(video_key: str, glb_key: str, download_key: str) -> None:
        """
        Render output preview section (video + GLB viewer).
        This is a shared component used by both single and multi-image tabs.
        
        Args:
            video_key: Unique key for video clear button
            glb_key: Unique key for GLB clear button
            download_key: Unique key for download button
        """
        # Video preview
        with st.container():
            # Show progress if generating
            is_generating = StateManager.is_generating()
            clear_video = show_video_preview(
                StateManager.get_generated_video(),
                show_clear=True,
                clear_key=video_key,
                show_progress=is_generating,
                progress_text="Generating 3D model..." if is_generating else None
            )
            if clear_video == "clear":
                StateManager.set_generated_video(None)
                StateManager.set_generated_glb(None)
                StateManager.set_generated_state(None)
                MemoryManager.cleanup_session_state(clear_all=False)
                st.rerun()
        
        # 3D model viewer (shown after video or GLB is available)
        generated_video = StateManager.get_generated_video()
        generated_glb = StateManager.get_generated_glb()
        generated_state = StateManager.get_generated_state()
        
        if generated_glb:
            st.success("✅ 3D Model Ready!")
            
            clear_glb = show_3d_model_viewer(
                generated_glb,
                show_clear=True,
                clear_key=glb_key
            )
            if clear_glb == "clear":
                StateManager.set_generated_glb(None)
                StateManager.set_generated_state(None)
                MemoryManager.cleanup_session_state(clear_all=False)
                st.rerun()
            
            # Download button
            with open(generated_glb, "rb") as file:
                st.download_button(
                    label="📥 Download GLB",
                    data=file,
                    file_name="generated_model.glb",
                    mime="model/gltf-binary",
                    type="primary",
                    key=download_key
                )
        else:
            # Show placeholder when no GLB
            show_3d_model_viewer(None)
        
        # Auto-extract GLB after video is shown (runs on next render after video appears)
        if generated_video and not generated_glb and generated_state:
            with st.spinner("Extracting GLB..."):
                # Determine which tab we're in based on the video_key
                if video_key == "single_video":
                    mesh_simplify = st.session_state.get('simplify_single', 0.95)
                    texture_size = st.session_state.get('texture_single', 1024)
                else:  # multi_video
                    mesh_simplify = st.session_state.get('simplify_multi', 0.95)
                    texture_size = st.session_state.get('texture_multi', 1024)
                
                export_params = ExportParams(
                    mesh_simplify=mesh_simplify,
                    texture_size=texture_size
                )
                
                glb_path, _ = GLBExporter.extract(generated_state, export_params)
                StateManager.set_generated_glb(glb_path)
                st.success("✅ 3D model complete!")
                st.rerun()
        
    
    @staticmethod
    def _render_output_column() -> None:
        """Render the output column."""
        st.subheader("Output")
        
        uploaded_image = StateManager.get_uploaded_image()
        
        # Use unified generation panel component
        SingleImageUI._render_generation_panel(
            uploaded_data=uploaded_image,
            is_multi_image=False,
            video_key="single_video",
            glb_key="single_glb",
            download_key="download_single",
            generate_key="generate_single",
            seed_key="seed_single",
            randomize_key="randomize_single",
            refinement_key="refinement_single",
            ss_strength_key="ss_strength_single",
            ss_steps_key="ss_steps_single",
            slat_strength_key="slat_strength_single",
            slat_steps_key="slat_steps_single",
            simplify_key="simplify_single",
            texture_key="texture_single",
            trial_id="single"
        )
    
    @staticmethod
    def _render_examples() -> None:
        """Render the examples section."""
        st.subheader("Examples")
        example_images = sorted([
            f'assets/example_image/{img}'
            for img in os.listdir("assets/example_image")
            if img.endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        selected_example = show_example_gallery(example_images, columns=4)
        if selected_example:
            # Load the selected example and clear all state
            example_img = Image.open(selected_example)
            
            # Resize large images to prevent OOM errors
            max_size = 512
            if max(example_img.size) > max_size:
                ratio = max_size / max(example_img.size)
                new_size = tuple(int(dim * ratio) for dim in example_img.size)
                example_img = example_img.resize(new_size, Image.Resampling.LANCZOS)
                print(f"Resized example image from {Image.open(selected_example).size} to {example_img.size}")
            
            StateManager.set_uploaded_image(example_img)
            st.session_state.processed_preview = None
            StateManager.set_generated_video(None)
            StateManager.set_generated_glb(None)
            StateManager.set_generated_state(None)
            # Force cleanup when loading example
            MemoryManager.cleanup_session_state(clear_all=False)
            st.rerun()


# ============================================================================
# TODO 8: Multi Image UI - Handles multi-image interface
# ============================================================================

class MultiImageUI:
    """Handles the multi-image generation UI."""
    
    @staticmethod
    def render() -> None:
        """Render the multi-image generation interface."""
        st.header("Multi-Image Generation")
        st.markdown("Upload 2-4 images from different viewpoints for improved 3D reconstruction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            MultiImageUI._render_input_column()
        
        with col2:
            MultiImageUI._render_output_column()
    
    @staticmethod
    def _render_input_column() -> None:
        """Render the input column."""
        st.subheader("Input")
        
        multi_uploaded_files = st.file_uploader(
            "Upload Images (2-4 images)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="multi_images",
            label_visibility="visible"
        )
        
        if multi_uploaded_files:
            if len(multi_uploaded_files) < 2:
                st.warning("Please upload at least 2 images")
            elif len(multi_uploaded_files) > 4:
                st.warning("Maximum 4 images allowed. Using first 4.")
                multi_uploaded_files = multi_uploaded_files[:4]
            
            # Display uploaded images (same style as single-image)
            if len(multi_uploaded_files) >= 2:
                st.markdown("**Uploaded Images:**")
                for i, uploaded_file in enumerate(multi_uploaded_files):
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"Image {i+1}", use_container_width=True)
                
                # Auto-process and show background-removed previews
                pipeline = StateManager.get_pipeline()
                if pipeline is not None:
                    st.markdown("**Background Removed (Auto):**")
                    with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        for i, uploaded_file in enumerate(multi_uploaded_files):
                            image = Image.open(uploaded_file)
                            processed_image = pipeline.preprocess_image(image)
                            st.image(processed_image, caption=f"Processed {i+1}", use_container_width=True)
                else:
                    st.info("Background removal preview will be shown after pipeline loads")
    
    @staticmethod
    def _render_output_column() -> None:
        """Render the output column."""
        st.subheader("Output")
        
        # Get uploaded files from the input column
        multi_uploaded_files = st.session_state.get("multi_images")
        
        # Use unified generation panel component
        SingleImageUI._render_generation_panel(
            uploaded_data=multi_uploaded_files,
            is_multi_image=True,
            video_key="multi_video",
            glb_key="multi_glb",
            download_key="download_multi",
            generate_key="generate_multi",
            seed_key="seed_multi",
            randomize_key="randomize_multi",
            refinement_key="refinement_multi",
            ss_strength_key="ss_strength_multi",
            ss_steps_key="ss_steps_multi",
            slat_strength_key="slat_strength_multi",
            slat_steps_key="slat_steps_multi",
            simplify_key="simplify_multi",
            texture_key="texture_multi",
            batch_size_key="batch_size_multi",
            trial_id="multi"
        )


# ============================================================================
# TODO 9: Trellis App - Main orchestrator class
# ============================================================================

class TrellisApp:
    """Main application orchestrator."""
    
    def __init__(self):
        """Initialize the Trellis application."""
        self._configure_page()
        self._check_gpu()
        self._initialize_pipeline()
    
    @staticmethod
    def _configure_page() -> None:
        """Configure Streamlit page settings and styles."""
        st.set_page_config(
            page_title="TRELLIS 3D Generator",
            page_icon="🎨",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Remove default Streamlit padding and margins
        st.markdown("""
            <style>
            /* Remove top padding */
            .block-container {
                padding-top: 1rem !important;
                padding-bottom: 0rem !important;
            }
            
            /* Remove default header space */
            header {
                background-color: transparent !important;
            }
            
            /* Reduce spacing between elements */
            .element-container {
                margin-bottom: 0.5rem !important;
            }
            
            /* Make the app use full viewport */
            .main .block-container {
                max-width: 100%;
                padding-left: 2rem;
                padding-right: 2rem;
            }
            </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def _check_gpu() -> Optional[str]:
        """
        Check for GPU availability and return GPU info.
        
        Returns:
            GPU info string or None if GPU check fails
        """
        if not torch.cuda.is_available():
            st.error("CUDA GPU not detected")
            st.error("TRELLIS requires a CUDA-compatible GPU to run.")
            st.info("If you're running this in Docker, ensure:")
            st.code("• Docker has GPU access (--gpus all)")
            st.code("• NVIDIA Container Toolkit is installed")
            st.code("• Run: nvidia-smi (on host) to verify GPU")
            st.stop()
        
        # Get GPU info
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            return f"{gpu_name} ({gpu_count} GPU{'s' if gpu_count > 1 else ''})"
        else:
            st.error("CUDA available but no GPUs accessible")
            st.stop()
    
    def _initialize_pipeline(self) -> None:
        """Initialize or load the TRELLIS pipeline."""
        # Initialize session state
        StateManager.initialize()

        pipeline = StateManager.get_pipeline()

        if pipeline is None:
            # Check if pipeline is already cached before showing loading screen
            from webui.initialize_pipeline import _PIPELINE_SINGLETON
            
            if _PIPELINE_SINGLETON is not None:
                # Pipeline already cached, show simple spinner
                st.markdown("""
                <style>
                .spinner-container {
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    gap: 1.5rem;
                }
                .spinner-text {
                    font-size: 1.2rem;
                    color: #667eea;
                    font-weight: 500;
                }
                </style>
                <div class="spinner-container">
                    <div class="spinner-text">Loading TRELLIS...</div>
                </div>
                """, unsafe_allow_html=True)
                
                print("Using cached pipeline from previous session")
                pipeline = _PIPELINE_SINGLETON
                StateManager.set_pipeline(pipeline)
                st.rerun()
            
            # Pipeline not cached, show full loading screen
            # Clean CUDA memory before loading
            if torch.cuda.is_available():
                print("Clearing CUDA memory before pipeline initialization...")
                MemoryManager.reduce_memory()

            # Get GPU info
            gpu_info = self._check_gpu()

            # Show loading screen
            from webui.loading_screen import capture_output
            progress_bar, status_text, console_output, start_time = show_loading_screen(gpu_info)

            # Load the pipeline with captured output
            status_text.text("Loading TRELLIS pipeline...")
            progress_bar.progress(10)

            try:
                with capture_output(console_output):
                    pipeline = load_pipeline()

                StateManager.set_pipeline(pipeline)

                # Complete loading UI
                finalize_loading(progress_bar, status_text, pipeline)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Handle OOM error gracefully
                    progress_bar.progress(0)
                    status_text.text("Error: Out of GPU memory")
                    st.error("❌ Out of GPU Memory")
                    st.error(str(e))
                    st.warning("**Solution:** Please restart the application to clear GPU memory.")
                    st.code("docker-compose restart", language="bash")
                    st.info("If the problem persists, there may be other processes using GPU memory. Check with `nvidia-smi`")
                    st.stop()
                else:
                    raise
    
    def run(self) -> None:
        """Run the main application."""
        # Perform periodic cleanup
        MemoryManager.periodic_cleanup()
        
        # Display title and description
        st.title("Image to 3D Asset with TRELLIS")
        st.markdown("""
        * **Single Image**: Upload one image for standard 3D generation
        * **Multi-Image**: Upload 2-4 images from different views for enhanced 3D reconstruction
        * If images have alpha channels, they'll be used as masks. Otherwise, we use `rembg` to remove backgrounds.
        """)
        
        # Create tabs
        tab1, tab2 = st.tabs(["Single Image", "Multi-Image"])
        
        with tab1:
            SingleImageUI.render()
        
        with tab2:
            MultiImageUI.render()


# ============================================================================
# TODO 10: Entry Point
# ============================================================================

if __name__ == "__main__":
    app = TrellisApp()
    app.run()
