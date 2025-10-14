from typing import *
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from torchvision import transforms
from PIL import Image
import rembg
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from ..representations import Gaussian, Strivec, MeshExtractResult


def _bhattacharyya_coefficient(p: torch.Tensor, q: torch.Tensor) -> float:
    """
    Compute Bhattacharyya coefficient between two probability distributions.

    BC(p,q) = sum_i sqrt(p_i * q_i)

    Args:
        p, q: Probability distributions (will be normalized)

    Returns:
        Bhattacharyya coefficient in [0, 1]
    """
    # Normalize to probability distributions
    p = p.flatten()
    q = q.flatten()
    p = p / p.sum()
    q = q / q.sum()

    # Compute BC
    bc = torch.sum(torch.sqrt(p * q)).item()
    return bc


def _compute_contradiction_measure(cond_list: List[torch.Tensor]) -> float:
    """
    Compute contradiction measure K for multi-view conditioning.

    Based on the mathematical theory of contradiction, we treat each conditioning
    view as a different "context" and measure how much they contradict any
    single global frame-independent model.

    K = -log₂(min_{contexts} BC(p_i, p_j))

    Args:
        cond_list: List of conditioning tensors, each shape (1, num_patches, hidden_dim)

    Returns:
        Contradiction measure K in [0, ∞), where 0 = perfect consistency
    """
    if len(cond_list) < 2:
        return 0.0  # Single view has no contradiction

    min_bc = float('inf')

    # Compute pairwise BC between all conditioning views
    for i in range(len(cond_list)):
        for j in range(i + 1, len(cond_list)):
            # Treat conditioning tensors as distributions over patch tokens
            # Normalize each conditioning tensor to sum to 1
            p = F.softmax(cond_list[i].flatten(), dim=0)
            q = F.softmax(cond_list[j].flatten(), dim=0)

            bc = torch.sum(torch.sqrt(p * q)).item()
            min_bc = min(min_bc, bc)

    # Apply weakest link principle: contradiction determined by most inconsistent pair
    if min_bc <= 0:
        return float('inf')  # Complete contradiction

    return -np.log2(min_bc)


class TrellisImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisImageTo3DPipeline, TrellisImageTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess input image: remove background, crop, and resize to 518x518.
        
        Resizes large images before background removal to prevent OOM errors.
        """
        # Resize images where either dimension exceeds 1024 to prevent OOM
        # Only resize if necessary, preserving original size for smaller images
        max_dimension = 728
        width, height = input.size
        
        if width > max_dimension or height > max_dimension:
            # Scale down so the largest dimension becomes max_dimension
            ratio = max_dimension / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            input = input.resize(new_size, Image.Resampling.LANCZOS)
            print(f"Pre-resized input image from {(width, height)} to {input.size} before background removal")
        
        # Check for existing alpha channel
        has_alpha = (
            input.mode == 'RGBA' and 
            not np.all(np.array(input)[:, :, 3] == 255)
        )
        
        if has_alpha:
            output = input
        else:
            # Remove background
            input = input.convert('RGB')
            if not hasattr(self, 'rembg_session') or self.rembg_session is None:
                cache_dir = os.environ.get('U2NET_HOME', os.path.expanduser('~/.u2net'))
                os.makedirs(cache_dir, exist_ok=True)
                os.chmod(cache_dir, 0o755)
                self.rembg_session = rembg.new_session('u2net', cache_dir=cache_dir)
            
            # Remove background - rembg will create proper alpha channel
            # where foreground = opaque (255) and background = transparent (0)
            # This preserves all colors in the foreground, including dark ones
            output = rembg.remove(
                input, 
                session=self.rembg_session,
                only_mask=False,
                post_process_mask=True
            )
        
        # Find object bounding box from alpha channel
        # Only exclude truly transparent pixels (background)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        # Use threshold of 10 to include all foreground pixels, even if partially transparent
        bbox_coords = np.argwhere(alpha > 10)
        
        # Calculate centered crop with 20% padding
        x_min, y_min = np.min(bbox_coords[:, 1]), np.min(bbox_coords[:, 0])
        x_max, y_max = np.max(bbox_coords[:, 1]), np.max(bbox_coords[:, 0])
        center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
        size = int(max(x_max - x_min, y_max - y_min) * 1.2)
        
        # Crop to square bounding box
        bbox = (
            center_x - size // 2,
            center_y - size // 2,
            center_x + size // 2,
            center_y + size // 2
        )
        output = output.crop(bbox)
        
        # Resize to target resolution with high-quality resampling
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        
        # Apply alpha compositing directly on PIL image to avoid quality loss
        # Convert to RGBA if needed, then composite onto white background
        if output.mode != 'RGBA':
            output = output.convert('RGBA')

        # Create black background
        background = Image.new('RGBA', output.size, (0, 0, 0, 255))

        # Composite the image onto black background
        composited = Image.alpha_composite(background, output)

        # Convert to RGB (remove alpha channel)
        return composited.convert('RGB')

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image: Single image or list of images for multi-view conditioning.

        Returns:
            dict: Conditioning dictionary with 'cond', 'neg_cond', 'multi_view', and 'contradiction' keys.
                Single image: cond shape (1, num_patches, hidden_dim), contradiction = 0.0
                Multi-view: cond is list of shapes [(1, num_patches, hidden_dim), ...]
                    to be cycled through during sampling steps.
                    contradiction = K measure of multi-view consistency (0.0 = perfect consistency)
        """
        cond = self.encode_image(image)

        # Handle single vs multiple images
        if isinstance(image, list) and len(image) > 1:
            # Multi-view: keep as separate conditioning tensors
            # These will be cycled through during the sampling loop
            # Each image gets its own conditioning tensor
            if cond.ndim == 2:
                cond = cond.unsqueeze(1)  # (num_images, num_patches, hidden_dim) -> add batch dim
            # Split into list of individual image conditionings
            cond_list = [cond[i:i+1] for i in range(cond.shape[0])]
            neg_cond_list = [torch.zeros_like(c) for c in cond_list]

            # Compute contradiction measure for multi-view consistency assessment
            contradiction = _compute_contradiction_measure(cond_list)

            return {
                'cond': cond_list,
                'neg_cond': neg_cond_list,
                'multi_view': True,
                'contradiction': contradiction
            }
        else:
            # Single image
            if cond.ndim == 2:
                cond = cond.unsqueeze(0)
            neg_cond = torch.zeros_like(cond)
            return {
                'cond': cond,
                'neg_cond': neg_cond,
                'multi_view': False,
                'contradiction': 0.0
            }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse 3D structure coordinates using flow matching.

        Args:
            cond: Conditioning dictionary with 'cond', 'neg_cond', and 'multi_view' keys
            num_samples: Number of samples to generate
            sampler_params: Additional sampler parameters (steps, cfg_strength, etc.)

        Returns:
            Sparse voxel coordinates as int tensor of shape (N, 4) where N is 
            the number of occupied voxels. Format: [batch_idx, z, y, x]
        """
        flow_model = self.models['sparse_structure_flow_model']
        
        # Initialize noise with same dtype as model for memory efficiency
        noise = torch.randn(
            num_samples, 
            flow_model.in_channels, 
            flow_model.resolution, 
            flow_model.resolution, 
            flow_model.resolution,
            dtype=flow_model.dtype,
            device=self.device
        )
        
        # Extract multi_view flag
        multi_view = cond.get('multi_view', False)
        
        # Ensure conditioning matches model dtype
        if multi_view:
            # For multi-view, cond and neg_cond are lists of tensors
            cond_typed = {
                'cond': [v.to(dtype=flow_model.dtype) for v in cond['cond']],
                'neg_cond': [v.to(dtype=flow_model.dtype) for v in cond['neg_cond']],
                'multi_view': True
            }
        else:
            # Single view: standard dict of tensors
            cond_typed = {k: v.to(dtype=flow_model.dtype) for k, v in cond.items() if k != 'multi_view'}
        
        params = {**self.sparse_structure_sampler_params, **sampler_params}
        
        # Sample sparse structure latent
        z_s = self.sparse_structure_sampler.sample(
            flow_model, noise, **cond_typed, **params, verbose=True
        ).samples
        
        # Decode to binary occupancy grid and extract coordinates
        occupancy = self.models['sparse_structure_decoder'](z_s)
        coords = torch.argwhere(occupancy.float() > 0)[:, [0, 2, 3, 4]].int()
        
        # Free memory
        del z_s, occupancy
        torch.cuda.empty_cache()
        
        return coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode structured latent to 3D representations.

        Args:
            slat: Structured latent sparse tensor
            formats: List of output formats to generate

        Returns:
            Dictionary with requested 3D representations (mesh, gaussian, radiance_field)
            All outputs are converted to fp32 for downstream compatibility
        """
        outputs = {}
        
        if 'mesh' in formats:
            # Aggressive memory cleanup before mesh decoding (most memory-intensive operation)
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Aggressive memory cleanup before mesh decoding (most memory-intensive operation)
            if torch.cuda.is_available():
                # Multiple cache clears to ensure memory is freed
                for _ in range(3):
                    torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Print memory stats for debugging
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"Before mesh decode - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
            # Force garbage collection before heavy operation
            gc.collect()
            
            mesh = self.models['slat_decoder_mesh'](slat)
            outputs['mesh'] = self._convert_to_fp32(mesh)
            
            # Cleanup after mesh decoding
            torch.cuda.empty_cache()
            gc.collect()
        
        if 'gaussian' in formats:
            gaussian = self.models['slat_decoder_gs'](slat)
            outputs['gaussian'] = self._convert_to_fp32(gaussian)
        
        if 'radiance_field' in formats:
            radiance_field = self.models['slat_decoder_rf'](slat)
            outputs['radiance_field'] = self._convert_to_fp32(radiance_field)
        
        return outputs

    def _convert_to_fp32(self, obj):
        """
        Convert model outputs from fp16 to fp32 for downstream compatibility.
        
        Handles tensors, lists, and custom objects with tensor attributes.
        All tensors are detached from the computation graph for inference.
        """
        if isinstance(obj, torch.Tensor):
            return obj.float().detach()
        
        if isinstance(obj, list):
            return [self._convert_to_fp32(item) for item in obj]
        
        if hasattr(obj, '__dict__'):
            # Convert tensor attributes in custom objects (e.g., MeshExtractResult, Gaussian)
            for attr_name, attr_value in obj.__dict__.items():
                if isinstance(attr_value, torch.Tensor):
                    setattr(obj, attr_name, attr_value.float().detach())
                elif isinstance(attr_value, list):
                    setattr(obj, attr_name, [self._convert_to_fp32(item) for item in attr_value])
            return obj
        
        return obj

    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent (SLAT) features using flow matching.

        Args:
            cond: Conditioning dictionary with 'cond', 'neg_cond', and 'multi_view' keys
            coords: Sparse voxel coordinates from sample_sparse_structure
            sampler_params: Additional sampler parameters

        Returns:
            Sparse tensor with latent features at given coordinates
        """
        flow_model = self.models['slat_flow_model']
        
        # Initialize sparse noise tensor
        noise = sp.SparseTensor(
            feats=torch.randn(
                coords.shape[0], 
                flow_model.in_channels,
                dtype=flow_model.dtype,
                device=self.device
            ),
            coords=coords
        )
        
        # Extract multi_view flag
        multi_view = cond.get('multi_view', False)
        
        # Ensure conditioning matches model dtype
        if multi_view:
            # For multi-view, cond and neg_cond are lists of tensors
            cond_typed = {
                'cond': [v.to(dtype=flow_model.dtype) for v in cond['cond']],
                'neg_cond': [v.to(dtype=flow_model.dtype) for v in cond['neg_cond']],
                'multi_view': True
            }
        else:
            # Single view: standard dict of tensors
            cond_typed = {k: v.to(dtype=flow_model.dtype) for k, v in cond.items() if k != 'multi_view'}
        
        params = {**self.slat_sampler_params, **sampler_params}
        
        # Sample latent features
        slat = self.slat_sampler.sample(
            flow_model, noise, **cond_typed, **params, verbose=True
        ).samples
        
        # Denormalize latent features
        std = torch.tensor(
            self.slat_normalization['std'], 
            device=slat.device, 
            dtype=slat.dtype
        )[None]
        mean = torch.tensor(
            self.slat_normalization['mean'],
            device=slat.device,
            dtype=slat.dtype
        )[None]
        slat = slat * std + mean
        
        torch.cuda.empty_cache()
        return slat

    @torch.no_grad()
    def run(
        self,
        image: Union[Image.Image, List[Image.Image]],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image: Single image or list of images for multi-view 3D generation.
                Multi-view: provide 2-4 images from different angles for improved results.
            num_samples: Number of samples to generate.
            seed: Random seed for reproducibility.
            sparse_structure_sampler_params: Additional parameters for sparse structure sampler.
            slat_sampler_params: Additional parameters for structured latent sampler.
            formats: Output formats to generate (['mesh', 'gaussian', 'radiance_field']).
            preprocess_image: Whether to preprocess images (background removal, cropping).

        Returns:
            Dictionary with requested 3D representations (mesh, gaussian, radiance_field).
        """
        # Handle both single image and list of images
        if isinstance(image, list):
            if preprocess_image:
                image = [self.preprocess_image(img) for img in image]
            cond = self.get_cond(image)
        else:
            if preprocess_image:
                image = self.preprocess_image(image)
            cond = self.get_cond([image])
        
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        
        # Critical: Clean up conditioning and coords before decoding to free memory
        del cond, coords
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        return self.decode_slat(slat, formats)
    
    def cleanup(self):
        """
        Cleanup resources to prevent memory leaks.
        Call this periodically or when shutting down.
        """
        # Close rembg session if it exists
        if hasattr(self, 'rembg_session') and self.rembg_session is not None:
            try:
                # rembg sessions don't have an explicit close, but we can delete the reference
                del self.rembg_session
                self.rembg_session = None
            except Exception as e:
                print(f"Error cleaning up rembg session: {e}")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
