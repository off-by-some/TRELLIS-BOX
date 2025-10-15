# Image Refinement Feature

TRELLIS-BOX includes optional image refinement using Stable Diffusion XL Refiner to enhance input image quality before 3D generation.

## What It Does

The refinement process:
1. Loads Stable Diffusion XL Refiner (1.0) from Hugging Face
2. Applies subtle image-to-image refinement to improve:
   - Texture clarity and sharpness
   - Color consistency
   - Detail preservation
   - Artifact reduction
3. Passes the refined image to TRELLIS for 3D generation

## How to Use

### Web Interface

1. Upload your image(s)
2. Open "Generation Settings" accordion
3. Check "Image Refinement (SD-XL)"
4. Click "Generate 3D Model"

The refinement happens automatically during preprocessing.

### Python API

```python
from trellis.pipelines import TrellisImageTo3DPipeline
from library.image_refiner import ImageRefiner
from PIL import Image

# Load pipeline and refiner
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

refiner = ImageRefiner(device="cuda", use_fp16=True)

# Load and refine image
image = Image.open("input.png")
refined_image = refiner.refine(
    image,
    strength=0.3,  # 0.0-1.0, lower = more faithful to original
    guidance_scale=7.5,
    num_inference_steps=20,
)

# Clean up refiner before TRELLIS
refiner.unload()

# Generate 3D from refined image
outputs = pipeline.run(refined_image, seed=1)
```

## Performance Impact

- **Time**: Adds ~10-15 seconds per image
- **VRAM**: Requires additional ~3-4GB during refinement
- **Quality**: Noticeable improvement for low-quality or noisy inputs

## Hardware Requirements

- **Minimum**: RTX 3080 (10GB VRAM)
- **Recommended**: RTX 3080 Ti (12GB VRAM) or better
- **Optimal**: RTX 4080/4090 (16GB+ VRAM)

The refiner uses FP16 precision and is automatically unloaded after use to free VRAM for TRELLIS.

## When to Use

**Use refinement when:**
- Input images are low resolution
- Images have compression artifacts
- Photos are slightly blurry or noisy
- You want maximum quality output

**Skip refinement when:**
- Input images are already high quality
- You need fastest generation time
- VRAM is limited (<12GB)
- Processing many images in batch

## Technical Details

- **Model**: `stabilityai/stable-diffusion-xl-refiner-1.0`
- **Precision**: FP16 for memory efficiency
- **Optimizations**: Attention slicing, model CPU offload
- **Default strength**: 0.3 (subtle refinement)
- **Inference steps**: 20 (balanced quality/speed)

## Troubleshooting

**Out of memory errors:**
- The refiner is automatically unloaded after use
- If issues persist, disable refinement or reduce batch size

**Refinement changes content too much:**
- This shouldn't happen with default settings (strength=0.3)
- If it does, the model may need different prompts

**Slow performance:**
- Refinement adds ~10s per image
- This is normal for SD-XL on consumer GPUs
- Consider disabling for faster iteration

## Future Enhancements

Potential improvements for future versions:
- 3D Gaussian refinement (post-generation)
- Adjustable refinement strength in UI
- Custom prompts for refinement
- Batch refinement optimization

