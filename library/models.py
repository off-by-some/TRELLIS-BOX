"""
Data models for TRELLIS application.
Pure data structures with no business logic.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


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
    fill_holes_resolution: int = 1024
    fill_holes_num_views: int = 1000


@dataclass
class ModelState:
    """State of a generated 3D model."""
    gaussian_data: Dict[str, Any]
    mesh_data: Dict[str, Any]
    trial_id: str


@dataclass
class ProcessingResult:
    """Result of image processing."""
    trial_id: str
    processed_images: Any  # PIL Images or list of PIL Images


@dataclass
class GenerationResult:
    """Result of 3D model generation."""
    model_state: ModelState
    video_path: str


@dataclass
class ExportResult:
    """Result of GLB export."""
    glb_path: str
    glb_path_duplicate: str  # For compatibility
