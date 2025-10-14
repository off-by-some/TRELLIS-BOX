"""
Example: TRELLIS 3D Generation with Image Refinement

This example shows how to use the Stable Diffusion XL Refiner to improve
input image quality before 3D generation, resulting in better 3D assets.
"""

import os
os.environ['SPCONV_ALGO'] = 'native'

import torch
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.pipelines.image_refiner import ImageRefiner
from trellis.utils import render_utils, postprocessing_utils


def main():
    print("Loading TRELLIS pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()
    
    print("Loading image refiner...")
    refiner = ImageRefiner(
        device="cuda",
        use_fp16=True  # Memory efficient for RTX 3080 Ti
    )
    
    # Load input image
    print("Loading input image...")
    input_image = Image.open("assets/example_image/T.png")
    
    # Optional: Refine the input image first
    print("Refining input image with Stable Diffusion XL...")
    refined_image = refiner.refine(
        input_image,
        strength=0.3,  # Subtle refinement to preserve original content
        guidance_scale=7.5,
        num_inference_steps=20,
        prompt="high quality, detailed, sharp, clean",
        negative_prompt="blurry, low quality, distorted, artifacts"
    )
    
    # Save refined image for comparison
    refined_image.save("refined_input.png")
    print("Refined image saved as refined_input.png")
    
    # Unload refiner to free VRAM for TRELLIS
    print("Unloading refiner to free VRAM...")
    refiner.unload()
    del refiner
    torch.cuda.empty_cache()
    
    # Generate 3D from refined image
    print("Generating 3D from refined image...")
    outputs = pipeline.run(
        refined_image,
        seed=1,
        sparse_structure_sampler_params={
            "steps": 12,
            "cfg_strength": 7.5,
        },
        slat_sampler_params={
            "steps": 12,
            "cfg_strength": 3,
        },
    )
    
    # Render outputs
    print("Rendering videos...")
    video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave("output_refined_gs.mp4", video_gs, fps=30)
    
    video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave("output_refined_mesh.mp4", video_mesh, fps=30)
    
    # Export GLB
    print("Exporting GLB...")
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.95,
        texture_size=1024,
    )
    glb.export("output_refined.glb")
    
    print("\n✓ Generation complete!")
    print("  - Refined input: refined_input.png")
    print("  - Gaussian video: output_refined_gs.mp4")
    print("  - Mesh video: output_refined_mesh.mp4")
    print("  - GLB model: output_refined.glb")


def compare_with_without_refinement():
    """
    Generate two 3D models side-by-side: one with refinement, one without.
    This helps visualize the quality improvement from image refinement.
    """
    print("=== Comparison Mode: With vs Without Refinement ===\n")
    
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()
    
    input_image = Image.open("assets/example_image/T.png")
    
    # Generate WITHOUT refinement
    print("1. Generating 3D WITHOUT refinement...")
    outputs_original = pipeline.run(input_image, seed=1)
    
    glb_original = postprocessing_utils.to_glb(
        outputs_original['gaussian'][0],
        outputs_original['mesh'][0],
    )
    glb_original.export("comparison_original.glb")
    print("   Saved: comparison_original.glb")
    
    # Generate WITH refinement
    print("\n2. Generating 3D WITH refinement...")
    refiner = ImageRefiner(device="cuda", use_fp16=True)
    refined_image = refiner.refine(input_image, strength=0.3)
    refined_image.save("comparison_refined_input.png")
    
    refiner.unload()
    del refiner
    torch.cuda.empty_cache()
    
    outputs_refined = pipeline.run(refined_image, seed=1)
    
    glb_refined = postprocessing_utils.to_glb(
        outputs_refined['gaussian'][0],
        outputs_refined['mesh'][0],
    )
    glb_refined.export("comparison_refined.glb")
    print("   Saved: comparison_refined.glb")
    
    print("\n✓ Comparison complete!")
    print("  Compare these files to see the quality difference:")
    print("  - comparison_original.glb (without refinement)")
    print("  - comparison_refined.glb (with refinement)")


if __name__ == "__main__":
    # Run basic example with refinement
    main()
    
    # Uncomment to run comparison
    # compare_with_without_refinement()

