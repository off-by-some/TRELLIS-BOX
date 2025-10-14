import streamlit as st
import streamlit.components.v1 as components

import os
import warnings
from webui.loading_screen import show_loading_screen, finalize_loading
from webui.initialize_pipeline import load_pipeline, reduce_memory_usage
from webui.ui_components import show_video_preview, show_3d_model_viewer, show_example_gallery
from webui.image_preview import image_preview
import base64

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*deprecated.*")
warnings.filterwarnings("ignore", message=".*xFormers is available.*")
warnings.filterwarnings("ignore", message=".*torch.library.impl_abstract.*renamed.*")
warnings.filterwarnings("ignore", message=".*torch.library.register_fake.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['SPCONV_ALGO'] = 'native'
# Memory optimizations for Trellis workloads - conservative but effective settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,garbage_collection_threshold:0.8'
# CUDA optimizations
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Non-blocking launches

# Suppress common library warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# Suppress transformers cache warnings
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

from typing import *
import torch
import numpy as np
import imageio
import uuid
import gc
import tempfile
from easydict import EasyDict as edict
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.pipelines.image_refiner import ImageRefiner
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.environ.get("TRELLIS_OUTPUT_DIR", "/tmp/Trellis-demo")

os.makedirs(TMP_DIR, exist_ok=True)

# Global refiner instance (loaded lazily)
refiner = None


def create_video_streaming(color_frames, normal_frames, output_path, fps=15, quality=8):
    """
    Create side-by-side video of color and normal renderings.

    Args:
        color_frames: List of color rendered frames
        normal_frames: List of normal map frames
        output_path: Output video file path
        fps: Frames per second
        quality: Video quality (1-10, higher is better)
    """
    def to_uint8_numpy(frame):
        """Convert frame to uint8 numpy array, handling both tensors and arrays."""
        if hasattr(frame, 'cpu'):
            # Tensor: (C, H, W) -> (H, W, C), scale to [0, 255]
            frame = frame.detach().cpu().numpy().transpose(1, 2, 0) * 255
            return np.clip(frame, 0, 255).astype(np.uint8)
        else:
            # Already numpy: handle NaN/Inf and ensure uint8
            frame = np.nan_to_num(frame, nan=0, posinf=255, neginf=0)
            return np.asarray(frame, dtype=np.uint8)
    
    # Process and combine all frames
    combined_frames = [
        np.concatenate([to_uint8_numpy(color), to_uint8_numpy(normal)], axis=1)
        for color, normal in zip(color_frames, normal_frames)
    ]
    
    imageio.mimsave(output_path, combined_frames, fps=fps, quality=quality)
    del combined_frames


def apply_image_refinement(image: Image.Image) -> Image.Image:
    """
    Apply Stable Diffusion XL refinement to improve input image quality.
    Loads refiner lazily and unloads after use to conserve VRAM.
    
    Args:
        image: Input PIL Image
        
    Returns:
        Refined PIL Image
    """
    global refiner
    
    # Load refiner if not already loaded
    if refiner is None:
        try:
            print("Loading Stable Diffusion XL Refiner...")
            refiner = ImageRefiner(device="cuda", use_fp16=True)
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
    else:
        return image


def preprocess_image(image: Image.Image, use_refinement: bool = False) -> Tuple[str, Image.Image]:
    """
    Preprocess the input image with memory-efficient operations.

    Args:
        image (Image.Image): The input image.

    Returns:
        str: uuid of the trial.
        Image.Image: The preprocessed image.
    """
    trial_id = str(uuid.uuid4())
    
    # Apply refinement if requested
    if use_refinement:
        print("Applying image refinement...")
        image = apply_image_refinement(image)
        # Clean up refiner VRAM before TRELLIS processing
        if refiner is not None:
            refiner.unload()
        torch.cuda.empty_cache()

    # Memory-efficient preprocessing with no gradients
    pipeline = st.session_state.pipeline
    with torch.no_grad():
        # Use autocast for mixed precision preprocessing if beneficial
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            processed_image = pipeline.preprocess_image(image)

    # High-quality image saving - no compression artifacts
    processed_image.save(f"{TMP_DIR}/{trial_id}.png", quality=100, subsampling=0)

    return trial_id, processed_image


def preprocess_images(images: List[Image.Image], use_refinement: bool = False) -> Tuple[str, List[Image.Image]]:
    """
    Preprocess multiple input images for multi-view 3D reconstruction.
    Memory optimization: Process and save images one by one to reduce peak memory usage.

    Args:
        images (List[Image.Image]): List of input images.

    Returns:
        str: uuid of the trial.
        List[Image.Image]: The preprocessed images.
    """
    trial_id = str(uuid.uuid4())
    processed_images = []

    # Process images one by one to minimize memory usage
    pipeline = st.session_state.pipeline
    for i, img in enumerate(images):
        # Apply refinement if requested
        if use_refinement:
            print(f"Refining image {i+1}/{len(images)}...")
            img = apply_image_refinement(img)
        
        processed_img = pipeline.preprocess_image(img)
        # High-quality image saving for multi-view - no compression artifacts
        processed_img.save(f"{TMP_DIR}/{trial_id}_{i}.png", quality=100, subsampling=0)
        processed_images.append(processed_img)

        # Force cleanup of intermediate objects
        del img
        torch.cuda.empty_cache()
    
    # Clean up refiner after processing all images
    if use_refinement and refiner is not None:
        refiner.unload()
        torch.cuda.empty_cache()

    return trial_id, processed_images


def pack_state(gs: Gaussian, mesh: MeshExtractResult, trial_id: str) -> dict:
    """
    Pack state while keeping tensors on GPU to avoid unnecessary CPU transfers.
    Memory optimization: Only move to CPU when actually needed for serialization.
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
    
    
def unpack_state(state: dict) -> Tuple[Gaussian, edict, str]:
    """
    Unpack state efficiently - tensors are already on GPU from pack_state.
    Memory optimization: Avoid redundant tensor creation and device transfers.
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


def image_to_3d(trial_id: str, seed: int, randomize_seed: bool, ss_guidance_strength: float, ss_sampling_steps: int, slat_guidance_strength: float, slat_sampling_steps: int) -> Tuple[dict, str]:
    """
    Convert a single image to a 3D model.

    Args:
        trial_id (str): The uuid of the trial.
        seed (int): The random seed.
        randomize_seed (bool): Whether to randomize the seed.
        ss_guidance_strength (float): The guidance strength for sparse structure generation.
        ss_sampling_steps (int): The number of sampling steps for sparse structure generation.
        slat_guidance_strength (float): The guidance strength for structured latent generation.
        slat_sampling_steps (int): The number of sampling steps for structured latent generation.

    Returns:
        dict: The information of the generated 3D model.
        str: The path to the video of the 3D model.
    """
    reduce_memory_usage()
    
    if randomize_seed:
        seed = np.random.randint(0, MAX_SEED)

    with Image.open(f"{TMP_DIR}/{trial_id}.png") as image:
        # Memory optimization: Ensure clean state before inference
        reduce_memory_usage()

        pipeline = st.session_state.pipeline
        with torch.inference_mode():
            # Critical memory optimization before heavy computation
            reduce_memory_usage()

            outputs = pipeline.run(
                image,
                seed=seed,
                formats=["gaussian", "mesh"],
                preprocess_image=False,
                sparse_structure_sampler_params={
                    "steps": ss_sampling_steps,
                    "cfg_strength": ss_guidance_strength,
                },
                slat_sampler_params={
                    "steps": slat_sampling_steps,
                    "cfg_strength": slat_guidance_strength,
                },
            )

        # Clean up immediately after inference
        del image
        reduce_memory_usage()

    # Extract gaussian and mesh for video rendering
    # Memory optimization: Extract and immediately clean up outputs
    gaussian = outputs['gaussian'][0]
    mesh = outputs['mesh'][0]

    # Clean up outputs immediately after extraction
    del outputs
    reduce_memory_usage()

    # Render videos with reduced frames for memory efficiency
    video_color = render_utils.render_video(gaussian, num_frames=60)['color']  # Reduced frames
    video_normal = render_utils.render_video(mesh, num_frames=60)['normal']   # Reduced frames

    # Generate new trial ID for video output
    video_trial_id = str(uuid.uuid4())
    video_path = f"{TMP_DIR}/{video_trial_id}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # Create video using streaming approach to minimize memory usage
    create_video_streaming(video_color, video_normal, video_path, fps=15, quality=8)

    # Clean up video arrays immediately after streaming creation
    del video_color, video_normal
    torch.cuda.empty_cache()
    gc.collect()

    # Pack state for GLB extraction - keeps tensors on GPU
    state = pack_state(gaussian, mesh, video_trial_id)

    # Early cleanup - gaussian and mesh are now in state
    del gaussian, mesh
    reduce_memory_usage()
    
    return state, video_path


def images_to_3d(trial_id: str, num_images: int, batch_size: int, seed: int, randomize_seed: bool, ss_guidance_strength: float, ss_sampling_steps: int, slat_guidance_strength: float, slat_sampling_steps: int) -> Tuple[dict, str]:
    """
    Convert multiple images to a 3D model using multi-view conditioning.

    Args:
        trial_id (str): The uuid of the trial.
        num_images (int): Number of images in this batch.
        batch_size (int): Number of images to process simultaneously.
        seed (int): The random seed.
        randomize_seed (bool): Whether to randomize the seed.
        ss_guidance_strength (float): The guidance strength for sparse structure generation.
        ss_sampling_steps (int): The number of sampling steps for sparse structure generation.
        slat_guidance_strength (float): The guidance strength for structured latent generation.
        slat_sampling_steps (int): The number of sampling steps for structured latent generation.

    Returns:
        dict: The information of the generated 3D model.
        str: The path to the video of the 3D model.
    """
    reduce_memory_usage()
    
    if randomize_seed:
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
        # For good memory (>=12GB), keep the user-specified batch_size

    # Ensure batch size doesn't exceed number of images
    batch_size = min(batch_size, num_images)

    # Load all images for multi-view conditioning
    # The pipeline natively supports multi-view input by processing all images together
    print(f"Loading {num_images} images for multi-view conditioning...")
    images = []
    for j in range(num_images):
        image = Image.open(f"{TMP_DIR}/{trial_id}_{j}.png")
        images.append(image)
    
    # Get multi-view conditioning by passing all images at once
    # This preserves spatial relationships between different viewpoints
    pipeline = st.session_state.pipeline
    with torch.inference_mode():
        cond = pipeline.get_cond(images)
    
    # Clean up images immediately
    del images
    reduce_memory_usage()

    reduce_memory_usage()

    torch.manual_seed(seed)

    with torch.inference_mode():
        coords = pipeline.sample_sparse_structure(cond, 1, {
            "steps": ss_sampling_steps,
            "cfg_strength": ss_guidance_strength,
        })

        slat = pipeline.sample_slat(cond, coords, {
            "steps": slat_sampling_steps,
            "cfg_strength": slat_guidance_strength,
        })

        # Clean up conditioning data immediately after sampling
        del cond, coords
        reduce_memory_usage()

        # Critical memory cleanup before mesh decoding (where the error occurred)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Decode SLAT without inference mode (rendering needs autograd compatibility)
        outputs = pipeline.decode_slat(slat, ["gaussian", "mesh"])
        del slat

    # Extract gaussian and mesh for video rendering
    # Memory optimization: Extract and immediately clean up outputs
    gaussian = outputs['gaussian'][0]
    mesh = outputs['mesh'][0]

    # Clean up outputs immediately after extraction
    del outputs
    reduce_memory_usage()

    # Render videos with reduced frames for memory efficiency
    video_color = render_utils.render_video(gaussian, num_frames=60)['color']  # Reduced frames
    video_normal = render_utils.render_video(mesh, num_frames=60)['normal']   # Reduced frames

    # Generate new trial ID for video output
    video_trial_id = str(uuid.uuid4())
    video_path = f"{TMP_DIR}/{video_trial_id}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # Create video using streaming approach to minimize memory usage
    create_video_streaming(video_color, video_normal, video_path, fps=15, quality=8)

    # Clean up video arrays immediately after streaming creation
    del video_color, video_normal
    torch.cuda.empty_cache()
    gc.collect()

    # Pack state for GLB extraction - keeps tensors on GPU
    state = pack_state(gaussian, mesh, video_trial_id)

    # Early cleanup - gaussian and mesh are now in state
    del gaussian, mesh
    reduce_memory_usage()
    
    return state, video_path


def extract_glb(state: dict, mesh_simplify: float, texture_size: int) -> Tuple[str, str]:
    """
    Extract a GLB file from the 3D model.

    Args:
        state (dict): The state of the generated 3D model.
        mesh_simplify (float): The mesh simplification factor.
        texture_size (int): The texture resolution.

    Returns:
        str: The path to the extracted GLB file.
    """
    # Aggressive memory cleanup before GLB extraction
    reduce_memory_usage()
    
    # Unpack state into gaussian splats and mesh
    gs, mesh, trial_id = unpack_state(state)
    
    # Clear state dict immediately as we've unpacked it
    del state
    reduce_memory_usage()

    # Generate GLB (postprocessing may need autograd compatibility)
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)

    # Save GLB file
    glb_path = f"{TMP_DIR}/{trial_id}.glb"
    glb.export(glb_path)

    # Clean up large objects immediately
    del gs, mesh, glb
    torch.cuda.empty_cache()
    gc.collect()

    return glb_path, glb_path


# Streamlit doesn't need button activation/deactivation functions


def main():
    st.title("Image to 3D Asset with TRELLIS")
    st.markdown("""
    * **Single Image**: Upload one image for standard 3D generation
    * **Multi-Image**: Upload 2-4 images from different views for enhanced 3D reconstruction
    * If images have alpha channels, they'll be used as masks. Otherwise, we use `rembg` to remove backgrounds.
    """)

    # Initialize session state
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'refiner' not in st.session_state:
        st.session_state.refiner = None
    if 'generated_video' not in st.session_state:
        st.session_state.generated_video = None
    if 'generated_glb' not in st.session_state:
        st.session_state.generated_glb = None
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    
    # Create tabs
    tab1, tab2 = st.tabs(["Single Image", "Multi-Image"])

    with tab1:
        st.header("Single Image Generation")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input")
            
            # File uploader without file listing
            uploaded_file = st.file_uploader(
                "Upload Image", 
                type=["png", "jpg", "jpeg"], 
                key="single_image",
                label_visibility="visible"
            )
            
            # Show original image with clear button using new reactive component
            if image_preview(
                st.session_state.uploaded_image,
                component_id="single_original",
                title="📷 Original Image",
                show_clear=True,
                show_info=True
            ):
                # Clear action triggered
                st.session_state.uploaded_image = None
                st.session_state.processed_preview = None
                st.session_state.generated_video = None
                st.session_state.generated_glb = None
                st.session_state.generated_state = None
                st.rerun()
            
            # Handle uploaded image
            if uploaded_file is not None and st.session_state.uploaded_image is None:
                st.session_state.uploaded_image = Image.open(uploaded_file)

            # Auto-remove background and show processed preview
            if st.session_state.uploaded_image is not None:
                if 'pipeline' in st.session_state and st.session_state.pipeline is not None:
                    pipeline = st.session_state.pipeline
                    with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        processed_image = pipeline.preprocess_image(st.session_state.uploaded_image)
                    
                    # Show processed image with new reactive component
                    image_preview(
                        processed_image,
                        component_id="single_processed",
                        title="🎨 Background Removed (Auto)",
                        show_clear=False,
                        show_info=True
                    )
                    st.session_state.processed_preview = processed_image
                else:
                    st.info("Background removal preview will be shown after pipeline loads")
                    st.session_state.processed_preview = None

        with col2:
            st.subheader("Output")
            
            # Only show generation settings if an image is uploaded
            if st.session_state.uploaded_image is not None:
                with st.expander("Generation Settings", expanded=True):
                    seed_single = st.slider("Seed", 0, MAX_SEED, 0, 1, key="seed_single")
                    randomize_seed_single = st.checkbox("Randomize Seed", value=True, key="randomize_single")
                    use_refinement_single = st.checkbox(
                        "Image Refinement (SD-XL)",
                        value=False,
                        help="Enhance input quality with Stable Diffusion XL (adds ~10s, uses extra VRAM)",
                        key="refinement_single"
                    )

                    st.markdown("**Stage 1: Sparse Structure Generation**")
                    ss_col1, ss_col2 = st.columns(2)
                    with ss_col1:
                        ss_guidance_strength_single = st.slider("Guidance Strength", 0.0, 10.0, 7.5, 0.1, key="ss_strength_single")
                    with ss_col2:
                        ss_sampling_steps_single = st.slider("Sampling Steps", 1, 50, 12, 1, key="ss_steps_single")

                    st.markdown("**Stage 2: Structured Latent Generation**")
                    slat_col1, slat_col2 = st.columns(2)
                    with slat_col1:
                        slat_guidance_strength_single = st.slider("Guidance Strength", 0.0, 10.0, 3.0, 0.1, key="slat_strength_single")
                    with slat_col2:
                        slat_sampling_steps_single = st.slider("Sampling Steps", 1, 50, 12, 1, key="slat_steps_single")

                # GLB Export Settings (shown before generation)
                with st.expander("GLB Export Settings", expanded=False):
                    mesh_simplify_single = st.slider("Simplify", 0.9, 0.98, 0.95, 0.01, key="simplify_single")
                    texture_size_single = st.slider("Texture Size", 512, 2048, 1024, 512, key="texture_single")

                if st.button("Generate 3D Model", type="primary", key="generate_single", use_container_width=True):
                    with st.spinner("Generating 3D model..."):
                        # Use the auto-processed image from preview, or process it if preview failed
                        if st.session_state.get('processed_preview') is not None:
                            processed_image = st.session_state.processed_preview
                            trial_id = str(uuid.uuid4())
                            # Save the processed image
                            processed_image.save(f"{TMP_DIR}/{trial_id}.png", quality=100, subsampling=0)
                        else:
                            # Fallback: process the image normally
                            image = st.session_state.uploaded_image
                            # Preprocess image
                            trial_id, processed_image = preprocess_image(image, use_refinement_single)

                        # Generate 3D model
                        state, video_path = image_to_3d(
                            trial_id,
                            seed_single if not randomize_seed_single else np.random.randint(0, MAX_SEED),
                            randomize_seed_single,
                            ss_guidance_strength_single,
                            ss_sampling_steps_single,
                            slat_guidance_strength_single,
                            slat_sampling_steps_single
                        )

                        st.session_state.generated_video = video_path
                        st.session_state.generated_state = state
                        st.session_state.processed_image = processed_image
                        st.rerun()  # Rerun to show video before GLB extraction
            
            # Video preview with clear button
            with st.container():
                clear_video = show_video_preview(
                    st.session_state.generated_video,
                    show_clear=True,
                    clear_key="single_video"
                )
                if clear_video == "clear":
                    st.session_state.generated_video = None
                    st.rerun()
            
            # Auto-extract GLB after video is shown
            if st.session_state.generated_video and not st.session_state.generated_glb and st.session_state.generated_state:
                with st.spinner("Extracting GLB..."):
                    glb_path, _ = extract_glb(st.session_state.generated_state, mesh_simplify_single, texture_size_single)
                    st.session_state.generated_glb = glb_path
                    st.success("✅ 3D model complete!")
                    st.rerun()

            # 3D model viewer with clear button
            if st.session_state.generated_glb:
                st.success("✅ 3D Model Ready!")
                
                clear_glb = show_3d_model_viewer(
                    st.session_state.generated_glb,
                    show_clear=True,
                    clear_key="single_glb"
                )
                if clear_glb == "clear":
                    st.session_state.generated_glb = None
                    st.session_state.generated_state = None
                    st.rerun()

                # Download button
                with open(st.session_state.generated_glb, "rb") as file:
                    st.download_button(
                        label="📥 Download GLB",
                        data=file,
                        file_name="generated_model.glb",
                        mime="model/gltf-binary",
                        type="primary"
                    )
            else:
                # Show placeholder when no GLB
                show_3d_model_viewer(None)

        # Examples
        st.subheader("Examples")
        example_images = sorted([f'assets/example_image/{img}' for img in os.listdir("assets/example_image") if img.endswith(('.png', '.jpg', '.jpeg'))])
        
        selected_example = show_example_gallery(example_images, columns=4)
        if selected_example:
            # Load the selected example and clear all state
            st.session_state.uploaded_image = Image.open(selected_example)
            st.session_state.processed_preview = None
            st.session_state.generated_video = None
            st.session_state.generated_glb = None
            st.session_state.generated_state = None
            st.rerun()
        
    with tab2:
        st.header("Multi-Image Generation")
        st.markdown("Upload 2-4 images from different viewpoints for improved 3D reconstruction")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input")
            multi_uploaded_files = st.file_uploader(
                "Upload Images (2-4 images)",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key="multi_images"
            )

            if multi_uploaded_files:
                if len(multi_uploaded_files) < 2:
                    st.warning("Please upload at least 2 images")
                elif len(multi_uploaded_files) > 4:
                    st.warning("Maximum 4 images allowed. Using first 4.")
                    multi_uploaded_files = multi_uploaded_files[:4]

                # Display uploaded images
                if len(multi_uploaded_files) >= 2:
                    st.markdown("**Uploaded Images:**")
                    image_cols = st.columns(min(len(multi_uploaded_files), 4))
                    for i, uploaded_file in enumerate(multi_uploaded_files):
                        with image_cols[i]:
                            image = Image.open(uploaded_file)
                            st.image(image, caption=f"Image {i+1}", width=150)

            with st.expander("Generation Settings", expanded=False):
                seed_multi = st.slider("Seed", 0, MAX_SEED, 0, 1, key="seed_multi")
                randomize_seed_multi = st.checkbox("Randomize Seed", value=True, key="randomize_multi")
                use_refinement_multi = st.checkbox(
                    "Image Refinement (SD-XL)",
                    value=False,
                    help="Enhance input quality with Stable Diffusion XL (adds ~10s per image)",
                    key="refinement_multi"
                )
                batch_size_multi = st.slider("Batch Size", 1, 4, 2, 1,
                    help="Number of images processed at once (lower = less memory)", key="batch_size_multi")

                st.markdown("**Stage 1: Sparse Structure Generation**")
                ss_col1, ss_col2 = st.columns(2)
                with ss_col1:
                    ss_guidance_strength_multi = st.slider("Guidance Strength", 0.0, 10.0, 7.5, 0.1, key="ss_strength_multi")
                with ss_col2:
                    ss_sampling_steps_multi = st.slider("Sampling Steps", 1, 50, 12, 1, key="ss_steps_multi")

                st.markdown("**Stage 2: Structured Latent Generation**")
                slat_col1, slat_col2 = st.columns(2)
                with slat_col1:
                    slat_guidance_strength_multi = st.slider("Guidance Strength", 0.0, 10.0, 3.0, 0.1, key="slat_strength_multi")
                with slat_col2:
                    slat_sampling_steps_multi = st.slider("Sampling Steps", 1, 50, 12, 1, key="slat_steps_multi")

            # GLB export settings (shown before generation)
            with st.expander("GLB Export Settings", expanded=False):
                mesh_simplify_multi = st.slider("Simplify", 0.9, 0.98, 0.95, 0.01, key="simplify_multi")
                texture_size_multi = st.slider("Texture Size", 512, 2048, 1024, 512, key="texture_multi")

            if st.button("Generate 3D Model from Multiple Views", type="primary",
                        disabled=len(multi_uploaded_files or []) < 2, key="generate_multi"):
                if multi_uploaded_files and len(multi_uploaded_files) >= 2:
                    with st.spinner("Processing multiple images..."):
                        # Process uploaded images
                        images = [Image.open(f) for f in multi_uploaded_files]

                        # Apply refinement if requested
                        if use_refinement_multi:
                            st.info("Applying image refinement to all images...")
                            images = [apply_image_refinement(img) for img in images]

                        # Preprocess images
                        trial_id, processed_images = preprocess_images(images, use_refinement_multi)

                        # Generate 3D model from multiple images
                        state, video_path = images_to_3d(
                            trial_id,
                            len(processed_images),
                            batch_size_multi,
                            seed_multi if not randomize_seed_multi else np.random.randint(0, MAX_SEED),
                            randomize_seed_multi,
                            ss_guidance_strength_multi,
                            ss_sampling_steps_multi,
                            slat_guidance_strength_multi,
                            slat_sampling_steps_multi
                        )

                        st.session_state.generated_video = video_path
                        st.session_state.generated_state = state

                        st.success("Multi-view 3D model video generated! Extracting GLB...")
                        
                        # Auto-extract GLB after video generation
                        glb_path, _ = extract_glb(state, mesh_simplify_multi, texture_size_multi)
                        st.session_state.generated_glb = glb_path
                        st.success("✅ Multi-view 3D model complete!")
                        st.rerun()

        with col2:
            st.subheader("Output")

            # Video preview with clear button
            with st.container():
                clear_video = show_video_preview(
                    st.session_state.generated_video,
                    show_clear=True,
                    clear_key="multi_video"
                )
                if clear_video == "clear":
                    st.session_state.generated_video = None
                    st.rerun()

            # 3D model viewer with clear button
            if st.session_state.generated_glb:
                st.success("✅ Multi-View 3D Model Ready!")
                
                clear_glb = show_3d_model_viewer(
                    st.session_state.generated_glb,
                    show_clear=True,
                    clear_key="multi_glb"
                )
                if clear_glb == "clear":
                    st.session_state.generated_glb = None
                    st.session_state.generated_state = None
                    st.rerun()

                # Download button
                with open(st.session_state.generated_glb, "rb") as file:
                    st.download_button(
                        label="📥 Download GLB",
                        data=file,
                        file_name="multi_view_model.glb",
                        mime="model/gltf-binary",
                        type="primary",
                        key="download_multi"
                    )
            else:
                # Show placeholder when no GLB
                show_3d_model_viewer(None)

    

# Launch the Streamlit app
if __name__ == "__main__":
    import time

    # Configure the page
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

    # Simple GPU check
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
        gpu_info = f"{gpu_name} ({gpu_count} GPU{'s' if gpu_count > 1 else ''})"
    else:
        st.error("CUDA available but no GPUs accessible")
        st.stop()

    # Check if pipeline is already loaded
    if 'pipeline' not in st.session_state or st.session_state.pipeline is None:
        # Show loading screen with console output and GPU info
        from webui.loading_screen import capture_output
        progress_bar, status_text, console_output, start_time = show_loading_screen(gpu_info)

        # Load the pipeline with captured output
        status_text.text("Loading TRELLIS pipeline...")
        progress_bar.progress(10)
        
        with capture_output(console_output):
            pipeline = load_pipeline()
        
        st.session_state.pipeline = pipeline

        # Complete loading UI
        finalize_loading(progress_bar, status_text, pipeline)

    else:
        # Pipeline is loaded, show main interface
        main()
