"""
Video rendering functionality for TRELLIS.
Pure business logic with no UI or state management dependencies.
"""

import numpy as np
import imageio
from typing import List, Any


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
