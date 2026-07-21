from typing import *
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from ..representations import Mesh, MeshWithVoxel
from ..utils.device import cleanup_memory, env_flag


class Trellis2ImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis2 image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        shape_slat_sampler (samplers.Sampler): The sampler for the structured latent.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        sparse_structure_sampler_params (dict): The parameters for the sparse structure sampler.
        shape_slat_sampler_params (dict): The parameters for the structured latent sampler.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model (Callable): The image conditioning model.
        rembg_model (Callable): The model for removing background.
        low_vram (bool): Whether to use low-VRAM mode.
    """
    model_names_to_load = [
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
        'shape_slat_decoder',
        'tex_slat_flow_model_512',
        'tex_slat_flow_model_1024',
        'tex_slat_decoder',
    ]
    lazy_model_names = model_names_to_load
    lazy_model_loading_env = "TRELLIS2_LAZY_LOAD_MODELS"
    lazy_model_loading_default = True
    unload_models_after_run_env = "TRELLIS2_UNLOAD_MODELS_AFTER_RUN"
    unload_models_after_run_default = True

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model: Callable = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
        default_pipeline_type: str = '1024_cascade',
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self.low_vram = low_vram
        self.flow_block_offload = env_flag("TRELLIS2_FLOW_BLOCK_OFFLOAD", True)
        self.default_pipeline_type = default_pipeline_type
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Trellis2ImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
        pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

        pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        pipeline.shape_slat_normalization = args['shape_slat_normalization']
        pipeline.tex_slat_normalization = args['tex_slat_normalization']

        image_cond_model_config = image_feature_extractor.resolve_image_cond_model_config(args['image_cond_model'])
        pipeline.image_cond_model = getattr(image_feature_extractor, image_cond_model_config['name'])(**image_cond_model_config['args'])
        rembg_args = args.get('rembg_model', {}).get('args', {})
        pipeline.rembg_model = rembg.create_rembg_model(
            rembg_args.get('model_name', 'ZhengPeng7/BiRefNet')
        )
        
        pipeline.low_vram = args.get('low_vram', True)
        pipeline.flow_block_offload = env_flag("TRELLIS2_FLOW_BLOCK_OFFLOAD", True)
        pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
        pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        pipeline._device = 'cpu'

        return pipeline

    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)
            self.image_cond_model.to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)

    def _use_flow_block_offload(self, model) -> bool:
        return (
            self.low_vram
            and self.flow_block_offload
            and torch.device(self.device).type == "cuda"
            and isinstance(model, nn.Module)
            and hasattr(model, "prepare_inference_block_offload")
        )

    def _prepare_sampling_model(self, model) -> None:
        if not self.low_vram:
            return
        if isinstance(model, nn.Module):
            if self._use_flow_block_offload(model):
                model.prepare_inference_block_offload(self.device)
            else:
                if hasattr(model, "disable_inference_block_offload"):
                    model.disable_inference_block_offload()
                model.to(self.device)
        else:
            for submodel in model:
                self._prepare_sampling_model(submodel)

    def _release_sampling_model(self, model) -> None:
        if not self.low_vram:
            return
        if isinstance(model, nn.Module):
            model.cpu()
        else:
            for submodel in model:
                self._release_sampling_model(submodel)
        cleanup_memory(self.device)

    @staticmethod
    def _inference_sampler_params(defaults: dict, overrides: dict) -> dict:
        return {"store_intermediates": False, **(defaults or {}), **(overrides or {})}

    @staticmethod
    def _cond_to_device(cond: dict, device: torch.device) -> dict:
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in cond.items()
        }

    @classmethod
    def required_model_names_for_pipeline_type(cls, pipeline_type: str) -> List[str]:
        common = [
            'sparse_structure_flow_model',
            'sparse_structure_decoder',
            'shape_slat_decoder',
            'tex_slat_decoder',
        ]
        if pipeline_type == '512':
            return common + [
                'shape_slat_flow_model_512',
                'tex_slat_flow_model_512',
            ]
        if pipeline_type == '1024':
            return common + [
                'shape_slat_flow_model_1024',
                'tex_slat_flow_model_1024',
            ]
        if pipeline_type in {'1024_cascade', '1536_cascade'}:
            return common + [
                'shape_slat_flow_model_512',
                'shape_slat_flow_model_1024',
                'tex_slat_flow_model_1024',
            ]
        raise ValueError(f"Invalid pipeline type: {pipeline_type}")

    def unload_run_models(self, names: Iterable[str]) -> None:
        if not (
            getattr(self, '_lazy_model_loading', False)
            and env_flag(self.unload_models_after_run_env, self.unload_models_after_run_default)
        ):
            return
        self.unload_models(names)
        cleanup_memory(self.device)

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            if self.rembg_model is None:
                raise ValueError(
                    "Background removal is disabled. Upload an RGBA image "
                    "with a non-opaque alpha channel, or set "
                    "TRELLIS2_REMBG_BACKEND=auto."
                )
            if self.low_vram:
                self.rembg_model.to(self.device)
            output = self.rembg_model(input)
            if self.low_vram:
                self.rembg_model.cpu()
                cleanup_memory(self.device)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        if bbox.size == 0:
            raise ValueError(
                "Background removal produced an empty alpha mask. Try an image "
                "with a clearer foreground, or upload an RGBA image with alpha."
            )
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]], resolution: int, include_neg_cond: bool = True) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        self.image_cond_model.image_size = resolution
        if self.low_vram:
            self.image_cond_model.to(self.device)
        cond = self.image_cond_model(image)
        if self.low_vram:
            self.image_cond_model.cpu()
            cleanup_memory(self.device)
        if not include_neg_cond:
            return {'cond': cond}
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            resolution (int): The resolution of the sparse structure.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample sparse structure latent
        flow_model = self.ensure_model('sparse_structure_flow_model')
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
        sampler_params = self._inference_sampler_params(self.sparse_structure_sampler_params, sampler_params)
        sampler_cond = self._cond_to_device(cond, self.device) if self.low_vram else cond
        if self.low_vram:
            self._prepare_sampling_model(flow_model)
        try:
            z_s = self.sparse_structure_sampler.sample(
                flow_model,
                noise,
                **sampler_cond,
                **sampler_params,
                verbose=True,
                tqdm_desc="Sampling sparse structure",
            ).samples
        finally:
            if self.low_vram:
                del sampler_cond
                self._release_sampling_model(flow_model)
        del noise
        
        # Decode sparse structure latent
        decoder = self.ensure_model('sparse_structure_decoder')
        if self.low_vram:
            decoder.to(self.device)
        try:
            decoded = decoder(z_s)>0
        finally:
            if self.low_vram:
                decoder.cpu()
                cleanup_memory(self.device)
        del z_s
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
        del decoded
        cleanup_memory(self.device)

        return coords

    def sample_shape_slat(
        self,
        cond: dict,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = self._inference_sampler_params(self.shape_slat_sampler_params, sampler_params)
        sampler_cond = self._cond_to_device(cond, self.device) if self.low_vram else cond
        if self.low_vram:
            self._prepare_sampling_model(flow_model)
        try:
            slat = self.shape_slat_sampler.sample(
                flow_model,
                noise,
                **sampler_cond,
                **sampler_params,
                verbose=True,
                tqdm_desc="Sampling shape SLat",
            ).samples
        finally:
            if self.low_vram:
                del sampler_cond
                self._release_sampling_model(flow_model)
        del noise

        std = torch.tensor(self.shape_slat_normalization['std'], dtype=slat.dtype, device=slat.device)[None]
        mean = torch.tensor(self.shape_slat_normalization['mean'], dtype=slat.dtype, device=slat.device)[None]
        slat = slat * std + mean
        cleanup_memory(self.device)
        
        return slat
    
    def sample_shape_slat_cascade(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # LR
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = self._inference_sampler_params(self.shape_slat_sampler_params, sampler_params)
        sampler_cond = self._cond_to_device(lr_cond, self.device) if self.low_vram else lr_cond
        if self.low_vram:
            self._prepare_sampling_model(flow_model_lr)
        try:
            slat = self.shape_slat_sampler.sample(
                flow_model_lr,
                noise,
                **sampler_cond,
                **sampler_params,
                verbose=True,
                tqdm_desc="Sampling shape SLat",
            ).samples
        finally:
            if self.low_vram:
                del sampler_cond
                self._release_sampling_model(flow_model_lr)
        del noise
        std = torch.tensor(self.shape_slat_normalization['std'], dtype=slat.dtype, device=slat.device)[None]
        mean = torch.tensor(self.shape_slat_normalization['mean'], dtype=slat.dtype, device=slat.device)[None]
        slat = slat * std + mean
        cleanup_memory(self.device)
        
        # Upsample
        shape_slat_decoder = self.ensure_model('shape_slat_decoder')
        if self.low_vram:
            shape_slat_decoder.to(self.device)
            shape_slat_decoder.low_vram = True
        try:
            hr_coords = shape_slat_decoder.upsample(slat, upsample_times=4)
        finally:
            if self.low_vram:
                shape_slat_decoder.cpu()
                shape_slat_decoder.low_vram = False
                cleanup_memory(self.device)
        hr_resolution = resolution
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens or hr_resolution == 1024:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128
        del hr_coords, quant_coords, slat
        cleanup_memory(self.device)
        
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = self._inference_sampler_params(self.shape_slat_sampler_params, sampler_params)
        sampler_cond = self._cond_to_device(cond, self.device) if self.low_vram else cond
        if self.low_vram:
            self._prepare_sampling_model(flow_model)
        try:
            slat = self.shape_slat_sampler.sample(
                flow_model,
                noise,
                **sampler_cond,
                **sampler_params,
                verbose=True,
                tqdm_desc="Sampling shape SLat",
            ).samples
        finally:
            if self.low_vram:
                del sampler_cond
                self._release_sampling_model(flow_model)
        del noise

        std = torch.tensor(self.shape_slat_normalization['std'], dtype=slat.dtype, device=slat.device)[None]
        mean = torch.tensor(self.shape_slat_normalization['mean'], dtype=slat.dtype, device=slat.device)[None]
        slat = slat * std + mean
        cleanup_memory(self.device)
        
        return slat, hr_resolution

    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            List[Mesh]: The decoded meshes.
            List[SparseTensor]: The decoded substructures.
        """
        shape_slat_decoder = self.ensure_model('shape_slat_decoder')
        shape_slat_decoder.set_resolution(resolution)
        if self.low_vram:
            shape_slat_decoder.to(self.device)
            shape_slat_decoder.low_vram = True
        try:
            ret = shape_slat_decoder(slat, return_subs=True)
        finally:
            if self.low_vram:
                shape_slat_decoder.cpu()
                shape_slat_decoder.low_vram = False
                cleanup_memory(self.device)
        return ret
    
    def sample_tex_slat(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'], dtype=shape_slat.dtype, device=shape_slat.device)[None]
        mean = torch.tensor(self.shape_slat_normalization['mean'], dtype=shape_slat.dtype, device=shape_slat.device)[None]
        shape_slat = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = self._inference_sampler_params(self.tex_slat_sampler_params, sampler_params)
        sampler_cond = self._cond_to_device(cond, self.device) if self.low_vram else cond
        if self.low_vram:
            self._prepare_sampling_model(flow_model)
        try:
            slat = self.tex_slat_sampler.sample(
                flow_model,
                noise,
                concat_cond=shape_slat,
                **sampler_cond,
                **sampler_params,
                verbose=True,
                tqdm_desc="Sampling texture SLat",
            ).samples
        finally:
            if self.low_vram:
                del sampler_cond
                self._release_sampling_model(flow_model)
        del noise, shape_slat

        std = torch.tensor(self.tex_slat_normalization['std'], dtype=slat.dtype, device=slat.device)[None]
        mean = torch.tensor(self.tex_slat_normalization['mean'], dtype=slat.dtype, device=slat.device)[None]
        slat = slat * std + mean
        cleanup_memory(self.device)
        
        return slat

    def decode_tex_slat(
        self,
        slat: SparseTensor,
        subs: List[SparseTensor],
    ) -> SparseTensor:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            SparseTensor: The decoded texture voxels
        """
        tex_slat_decoder = self.ensure_model('tex_slat_decoder')
        if self.low_vram:
            tex_slat_decoder.to(self.device)
        try:
            ret = tex_slat_decoder(slat, guide_subs=subs) * 0.5 + 0.5
        finally:
            if self.low_vram:
                tex_slat_decoder.cpu()
                cleanup_memory(self.device)
        return ret
    
    @torch.inference_mode()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
    ) -> List[MeshWithVoxel]:
        """
        Decode the latent codes.

        Args:
            shape_slat (SparseTensor): The structured latent for shape.
            tex_slat (SparseTensor): The structured latent for texture.
            resolution (int): The resolution of the output.
        """
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        tex_voxels = self.decode_tex_slat(tex_slat, subs)
        out_mesh = []
        for m, v in zip(meshes, tex_voxels):
            m.fill_holes()
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / resolution,
                    coords = v.coords[:, 1:],
                    attrs = v.feats,
                    voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pbr_attr_layout
                )
            )
        del meshes, subs, tex_voxels
        cleanup_memory(self.device)
        return out_mesh
    
    @torch.inference_mode()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            pipeline_type (str): The type of the pipeline. Options: '512', '1024', '1024_cascade', '1536_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.
        """
        # Check pipeline type
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type in {'1024_cascade', '1536_cascade'}:
            print(
                f"[startup] Using {pipeline_type}: loading 512 coarse shape and high-resolution refinement models.",
                flush=True,
            )
        required_model_names = self.required_model_names_for_pipeline_type(pipeline_type)
        self.ensure_models(required_model_names)

        try:
            if preprocess_image:
                image = self.preprocess_image(image)
            torch.manual_seed(seed)
            cond_512 = self.get_cond([image], 512)
            if self.low_vram:
                cond_512 = self._cond_to_device(cond_512, torch.device("cpu"))
                cleanup_memory(self.device)
            cond_1024 = self.get_cond([image], 1024) if pipeline_type != '512' else None
            if self.low_vram and cond_1024 is not None:
                cond_1024 = self._cond_to_device(cond_1024, torch.device("cpu"))
                cleanup_memory(self.device)
            ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
            coords = self.sample_sparse_structure(
                cond_512, ss_res,
                num_samples, sparse_structure_sampler_params
            )
            if pipeline_type == '512':
                shape_slat = self.sample_shape_slat(
                    cond_512, self.ensure_model('shape_slat_flow_model_512'),
                    coords, shape_slat_sampler_params
                )
                tex_slat = self.sample_tex_slat(
                    cond_512, self.ensure_model('tex_slat_flow_model_512'),
                    shape_slat, tex_slat_sampler_params
                )
                res = 512
            elif pipeline_type == '1024':
                shape_slat = self.sample_shape_slat(
                    cond_1024, self.ensure_model('shape_slat_flow_model_1024'),
                    coords, shape_slat_sampler_params
                )
                tex_slat = self.sample_tex_slat(
                    cond_1024, self.ensure_model('tex_slat_flow_model_1024'),
                    shape_slat, tex_slat_sampler_params
                )
                res = 1024
            elif pipeline_type == '1024_cascade':
                shape_slat, res = self.sample_shape_slat_cascade(
                    cond_512, cond_1024,
                    self.ensure_model('shape_slat_flow_model_512'), self.ensure_model('shape_slat_flow_model_1024'),
                    512, 1024,
                    coords, shape_slat_sampler_params,
                    max_num_tokens
                )
                tex_slat = self.sample_tex_slat(
                    cond_1024, self.ensure_model('tex_slat_flow_model_1024'),
                    shape_slat, tex_slat_sampler_params
                )
            elif pipeline_type == '1536_cascade':
                shape_slat, res = self.sample_shape_slat_cascade(
                    cond_512, cond_1024,
                    self.ensure_model('shape_slat_flow_model_512'), self.ensure_model('shape_slat_flow_model_1024'),
                    512, 1536,
                    coords, shape_slat_sampler_params,
                    max_num_tokens
                )
                tex_slat = self.sample_tex_slat(
                    cond_1024, self.ensure_model('tex_slat_flow_model_1024'),
                    shape_slat, tex_slat_sampler_params
                )
            del coords, cond_512
            if cond_1024 is not None:
                del cond_1024
            cleanup_memory(self.device)
            out_mesh = self.decode_latent(shape_slat, tex_slat, res)
            if return_latent:
                return out_mesh, (shape_slat, tex_slat, res)
            else:
                del shape_slat, tex_slat
                cleanup_memory(self.device)
                return out_mesh
        finally:
            self.unload_run_models(required_model_names)
