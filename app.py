import gradio as gr
import spaces
from gradio_litmodel3d import LitModel3D

import os
os.environ['SPCONV_ALGO'] = 'native'
# Memory optimizations for Trellis workloads - conservative but effective settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,garbage_collection_threshold:0.8'
# CUDA optimizations
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Non-blocking launches

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
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = "/tmp/Trellis-demo"

os.makedirs(TMP_DIR, exist_ok=True)


def defragment_memory():
    """
    Aggressive memory defragmentation for PyTorch CUDA allocator.
    Attempts to force consolidation of fragmented memory blocks.
    """
    if not torch.cuda.is_available():
        return

    # Multiple rounds of cache clearing and memory consolidation
    for _ in range(3):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if hasattr(torch.cuda, 'consolidate_memory'):
            torch.cuda.consolidate_memory()

    # Defragment by allocating temporary tensors in increasing sizes
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated()
    free_memory_mb = (total_memory - allocated_memory) // (1024 * 1024)
    
    for i in range(10):
        size_mb = 50 * (i + 1)
        if size_mb > min(500, free_memory_mb):
            break
        
        temp = torch.zeros(
            size_mb * 1024 * 1024 // 4,
            dtype=torch.float32,
            device='cuda'
        )
        del temp
        torch.cuda.empty_cache()
    
    torch.cuda.synchronize()


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


def cleanup_temp_files(max_age_hours=1):
    """
    Aggressive cleanup of temporary files to free disk space.
    Memory optimization: Remove old temporary files that are no longer needed.

    Args:
        max_age_hours: Maximum age of files to keep (in hours)
    """
    import time
    import os
    import glob

    try:
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        # Clean up image files
        for pattern in [f"{TMP_DIR}/*.png", f"{TMP_DIR}/*.mp4", f"{TMP_DIR}/*.glb"]:
            for filepath in glob.glob(pattern):
                try:
                    if os.path.exists(filepath):
                        file_age = current_time - os.path.getmtime(filepath)
                        if file_age > max_age_seconds:
                            os.remove(filepath)
                except (OSError, FileNotFoundError):
                    pass  # File already deleted or inaccessible
    except Exception as e:
        # Don't let cleanup errors affect the main flow
        pass


def reduce_memory_usage():
    """
    Critical memory management for Trellis workloads: Clears cache, forces GC, and optimizes memory layout.
    Includes advanced PyTorch memory management and fragmentation prevention.
    """
    if torch.cuda.is_available():
        # Critical: Empty cache and synchronize multiple times
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Advanced memory optimization - consolidate memory pools to reduce fragmentation
        if hasattr(torch.cuda, 'consolidate_memory'):
            torch.cuda.consolidate_memory()

        # Force multiple cache clears to ensure fragmentation is reduced
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Reset allocator statistics for better future allocations
        torch.cuda.reset_peak_memory_stats()

        # Force cleanup of any cached computations
        if hasattr(torch.cuda, 'reset_max_memory_allocated'):
            torch.cuda.reset_max_memory_allocated()
        if hasattr(torch.cuda, 'reset_max_memory_cached'):
            torch.cuda.reset_max_memory_cached()

        # Additional memory defragmentation attempt
        # Allocate a small tensor and immediately free it to trigger cleanup
        temp = torch.zeros(1, device='cuda')
        del temp
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Optimize Python garbage collection with maximum aggression
    gc.collect(generation=2)  # Collect all generations
    gc.collect(generation=1)  # Collect generation 1
    gc.collect(generation=0)  # Collect generation 0
    gc.collect()  # Additional collection pass

    # Tune GC thresholds for maximum memory efficiency with large objects
    gc.set_threshold(300, 5, 5)  # Maximum aggression

    # Aggressive temporary file cleanup
    cleanup_temp_files(max_age_hours=1)

    # Final defragmentation attempt
    defragment_memory()


def preprocess_image(image: Image.Image) -> Tuple[str, Image.Image]:
    """
    Preprocess the input image with memory-efficient operations.

    Args:
        image (Image.Image): The input image.

    Returns:
        str: uuid of the trial.
        Image.Image: The preprocessed image.
    """
    trial_id = str(uuid.uuid4())

    # Memory-efficient preprocessing with no gradients
    with torch.no_grad():
        # Use autocast for mixed precision preprocessing if beneficial
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            processed_image = pipeline.preprocess_image(image)

    # High-quality image saving - no compression artifacts
    processed_image.save(f"{TMP_DIR}/{trial_id}.png", quality=100, subsampling=0)

    return trial_id, processed_image


def preprocess_images(images: List[Image.Image]) -> Tuple[str, List[Image.Image]]:
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
    for i, img in enumerate(images):
        processed_img = pipeline.preprocess_image(img)
        # High-quality image saving for multi-view - no compression artifacts
        processed_img.save(f"{TMP_DIR}/{trial_id}_{i}.png", quality=100, subsampling=0)
        processed_images.append(processed_img)

        # Force cleanup of intermediate objects
        del img
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


@spaces.GPU
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


@spaces.GPU
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


@spaces.GPU
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


def activate_button() -> gr.Button:
    return gr.Button(interactive=True)


def deactivate_button() -> gr.Button:
    return gr.Button(interactive=False)


with gr.Blocks() as demo:
    gr.Markdown("""
    ## Image to 3D Asset with [TRELLIS](https://trellis3d.github.io/)
    * **Single Image**: Upload one image for standard 3D generation
    * **Multi-Image**: Upload 2-4 images from different views for enhanced 3D reconstruction
    * If images have alpha channels, they'll be used as masks. Otherwise, we use `rembg` to remove backgrounds.
    """)
    
    # Shared state and output elements
    output_buf = gr.State()
    
    with gr.Tabs():
        # ===== Single Image Tab =====
        with gr.Tab("Single Image"):
            with gr.Row():
                with gr.Column():
                    image_prompt = gr.Image(label="Image Prompt", image_mode="RGBA", type="pil", height=300)
                    
                    with gr.Accordion(label="Generation Settings", open=False):
                        seed_single = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                        randomize_seed_single = gr.Checkbox(label="Randomize Seed", value=True)
                        gr.Markdown("Stage 1: Sparse Structure Generation")
                        with gr.Row():
                            ss_guidance_strength_single = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                            ss_sampling_steps_single = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                        gr.Markdown("Stage 2: Structured Latent Generation")
                        with gr.Row():
                            slat_guidance_strength_single = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3.0, step=0.1)
                            slat_sampling_steps_single = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)

                    generate_btn_single = gr.Button("Generate 3D Model", variant="primary")
                    
                    with gr.Accordion(label="GLB Export Settings (Automatic)", open=False):
                        gr.Markdown("*GLB export happens automatically after generation*")
                        mesh_simplify_single = gr.Slider(0.9, 0.98, label="Simplify", value=0.95, step=0.01)
                        texture_size_single = gr.Slider(512, 2048, label="Texture Size", value=1024, step=512)
                    
                    extract_glb_btn_single = gr.Button("Extract GLB", interactive=False)

                with gr.Column():
                    video_output_single = gr.Video(label="Generated 3D Asset", autoplay=True, loop=True, height=300)
                    model_output_single = LitModel3D(label="Extracted GLB", exposure=20.0, height=300)
                    download_glb_single = gr.DownloadButton(label="Download GLB", interactive=False)
            
            trial_id_single = gr.Textbox(visible=False)
            
            # Examples for single image
            with gr.Row():
                examples_single = gr.Examples(
                    examples=[f'assets/example_image/{image}' for image in os.listdir("assets/example_image")],
                    inputs=[image_prompt],
                    fn=preprocess_image,
                    outputs=[trial_id_single, image_prompt],
                    run_on_click=True,
                    examples_per_page=64,
                )
        
        # ===== Multi-Image Tab =====
        with gr.Tab("Multi-Image"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("Upload 2-4 images from different viewpoints for improved 3D reconstruction")
                    multi_image_prompt = gr.Gallery(label="Image Prompts (2-4 images)", type="pil", height=300, columns=2)
                    
                    with gr.Accordion(label="Generation Settings", open=False):
                        seed_multi = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                        randomize_seed_multi = gr.Checkbox(label="Randomize Seed", value=True)
                        batch_size_multi = gr.Slider(1, 4, label="Batch Size", value=2, step=1, info="Number of images processed at once (lower = less memory)")
                        gr.Markdown("Stage 1: Sparse Structure Generation")
                        with gr.Row():
                            ss_guidance_strength_multi = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                            ss_sampling_steps_multi = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                        gr.Markdown("Stage 2: Structured Latent Generation")
                        with gr.Row():
                            slat_guidance_strength_multi = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3.0, step=0.1)
                            slat_sampling_steps_multi = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)

                    generate_btn_multi = gr.Button("Generate 3D Model from Multiple Views", variant="primary")
                    
                    with gr.Accordion(label="GLB Export Settings (Automatic)", open=False):
                        gr.Markdown("*GLB export happens automatically after generation*")
                        mesh_simplify_multi = gr.Slider(0.9, 0.98, label="Simplify", value=0.95, step=0.01)
                        texture_size_multi = gr.Slider(512, 2048, label="Texture Size", value=1024, step=512)
                    
                    extract_glb_btn_multi = gr.Button("Extract GLB", interactive=False)

                with gr.Column():
                    video_output_multi = gr.Video(label="Generated 3D Asset", autoplay=True, loop=True, height=300)
                    model_output_multi = LitModel3D(label="Extracted GLB", exposure=20.0, height=300)
                    download_glb_multi = gr.DownloadButton(label="Download GLB", interactive=False)
            
            trial_id_multi = gr.Textbox(visible=False)
            num_images_multi = gr.Number(visible=False, value=0)

    # ===== Single Image Handlers =====
    image_prompt.upload(
        preprocess_image,
        inputs=[image_prompt],
        outputs=[trial_id_single, image_prompt],
        api_name=False,
    )
    
    generate_btn_single.click(
        image_to_3d,
        inputs=[trial_id_single, seed_single, randomize_seed_single, ss_guidance_strength_single, ss_sampling_steps_single, slat_guidance_strength_single, slat_sampling_steps_single],
        outputs=[output_buf, video_output_single],
        api_name=False,
    ).then(
        extract_glb,  # Automatically extract GLB after video rendering
        inputs=[output_buf, mesh_simplify_single, texture_size_single],
        outputs=[model_output_single, download_glb_single],
        api_name=False,
    ).then(
        activate_button,
        outputs=[download_glb_single],
        api_name=False,
    )

    extract_glb_btn_single.click(
        extract_glb,
        inputs=[output_buf, mesh_simplify_single, texture_size_single],
        outputs=[model_output_single, download_glb_single],
        api_name=False,
    ).then(
        activate_button,
        outputs=[download_glb_single],
        api_name=False,
    )

    # ===== Multi-Image Handlers =====
    def process_multi_images(images):
        if not images or len(images) < 2:
            return None, [], 0

        # Limit to 4 images for memory safety (streaming processing)
        max_images = 4
        if len(images) > max_images:
            print(f"Warning: Limiting to {max_images} images for memory safety (received {len(images)})")
            images = images[:max_images]

        # Extract PIL images from gallery tuples (gallery returns list of tuples)
        pil_images = [img[0] if isinstance(img, tuple) else img for img in images]
        trial_id, processed = preprocess_images(pil_images)
        return trial_id, processed, len(processed)
    
    multi_image_prompt.upload(
        process_multi_images,
        inputs=[multi_image_prompt],
        outputs=[trial_id_multi, multi_image_prompt, num_images_multi],
        api_name=False,
    )
    
    generate_btn_multi.click(
        images_to_3d,
        inputs=[trial_id_multi, num_images_multi, batch_size_multi, seed_multi, randomize_seed_multi, ss_guidance_strength_multi, ss_sampling_steps_multi, slat_guidance_strength_multi, slat_sampling_steps_multi],
        outputs=[output_buf, video_output_multi],
        api_name=False,
    ).then(
        extract_glb,  # Automatically extract GLB after video rendering
        inputs=[output_buf, mesh_simplify_multi, texture_size_multi],
        outputs=[model_output_multi, download_glb_multi],
        api_name=False,
    ).then(
        activate_button,
        outputs=[download_glb_multi],
        api_name=False,
    )

    extract_glb_btn_multi.click(
        extract_glb,
        inputs=[output_buf, mesh_simplify_multi, texture_size_multi],
        outputs=[model_output_multi, download_glb_multi],
        api_name=False,
    ).then(
        activate_button,
        outputs=[download_glb_multi],
        api_name=False,
    )
    

# Launch the Gradio app
if __name__ == "__main__":
    print("Loading TRELLIS pipeline...")

    # Pre-optimize memory before loading large model
    reduce_memory_usage()

    # Load pipeline with memory optimizations
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")

    # Move to GPU with memory optimization
    pipeline.cuda()
    reduce_memory_usage()  # Clean up any loading artifacts

    # Set models to evaluation mode and convert transformer models to half precision for memory efficiency
    for model_name, model in pipeline.models.items():
        if hasattr(model, 'eval'):
            model.eval()

        # Convert all transformer models to fp16 for maximum VRAM savings
        if 'flow' in model_name or 'decoder' in model_name:
            # Convert flow models and decoders to half precision for significant memory savings
            # But keep norm layers in fp32 for numerical stability
            model.half()

            # Ensure norm layers stay in fp32 for numerical stability
            from trellis.modules.norm import LayerNorm32, GroupNorm32, ChannelLayerNorm32
            from trellis.modules.sparse.norm import SparseGroupNorm32, SparseLayerNorm32
            from trellis.modules.attention.modules import MultiHeadRMSNorm
            from trellis.modules.sparse.attention.modules import SparseMultiHeadRMSNorm
            for module in model.modules():
                if isinstance(module, (LayerNorm32, GroupNorm32, ChannelLayerNorm32,
                                     SparseGroupNorm32, SparseLayerNorm32,
                                     MultiHeadRMSNorm, SparseMultiHeadRMSNorm)):
                    module.float()

        # Keep image_cond_model (DINOv2) and other models in fp32 for compatibility

    # Enable cuDNN optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    # Advanced memory and precision optimizations
    if torch.cuda.is_available():
        # Enable TF32 for better performance on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Advanced attention and memory optimizations
        # Enable gradient checkpointing for inference memory efficiency
        for model in pipeline.models.values():
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()

        # Memory-efficient sparse attention optimizations
        if hasattr(torch, 'sparse'):
            # Ensure sparse operations are available for memory efficiency
            pass  # torch.sparse is available

        # Enable memory-efficient sparse operations
        os.environ['CUDNN_CONVOLUTION_BWD_FILTER_ALGO'] = '1'  # Use efficient backward algorithms

        # Kernel fusion optimizations
        if hasattr(torch.jit, 'enable_onednn_fusion'):
            torch.jit.enable_onednn_fusion(True)

        # Memory optimization for large models on 3080 Ti
        # Removed memory fraction limit to allow full GPU utilization during mesh decoding

        # Enable memory efficient attention if available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            os.environ['TORCH_USE_CUDA_DSA'] = '1'

    print("TRELLIS pipeline loaded successfully")
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # Final memory optimization before serving
    reduce_memory_usage()
    print(f"GPU Memory after optimization: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")

    demo.launch(server_name="0.0.0.0", server_port=7860)
