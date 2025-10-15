"""
GLB export functionality for TRELLIS.
Pure business logic with no UI or state management dependencies.
"""

import torch
import gc
from typing import Optional

from trellis.utils import postprocessing_utils
from library.models import ModelState, ExportParams, ExportResult
from library.model_generator import ModelGenerator


class GLBExporter:
    """Handles GLB file export operations."""

    def __init__(self, tmp_dir: str = "/tmp/Trellis-demo"):
        """Initialize the GLB exporter.

        Args:
            tmp_dir: Directory for temporary files
        """
        self.tmp_dir = tmp_dir
        self.model_generator = ModelGenerator(tmp_dir=tmp_dir)

    def export(self, state: ModelState, params: ExportParams) -> ExportResult:
        """
        Extract a GLB file from the 3D model.

        Args:
            state: The state of the generated 3D model
            params: Export parameters

        Returns:
            ExportResult with GLB path
        """
        # Aggressive memory cleanup before GLB extraction
        torch.cuda.empty_cache()

        # Unpack state into gaussian splats and mesh
        gs, mesh, trial_id = self.model_generator.unpack_model_state(state)

        # Clear state dict immediately as we've unpacked it
        del state
        torch.cuda.empty_cache()

        # Generate GLB (postprocessing may need autograd compatibility)
        glb = postprocessing_utils.to_glb(
            gs, mesh,
            simplify=params.mesh_simplify,
            texture_size=params.texture_size,
            fill_holes_resolution=params.fill_holes_resolution,
            fill_holes_num_views=params.fill_holes_num_views,
            verbose=False
        )

        # Save GLB file
        glb_path = f"{self.tmp_dir}/{trial_id}.glb"
        glb.export(glb_path)

        # Clean up large objects immediately
        del gs, mesh, glb
        torch.cuda.empty_cache()
        gc.collect()

        return ExportResult(glb_path=glb_path, glb_path_duplicate=glb_path)
