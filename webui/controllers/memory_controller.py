"""
Memory management controller for TRELLIS application.
Handles memory cleanup and optimization without UI dependencies.
"""

import torch
import gc
from typing import Optional
import os

from webui.initialize_pipeline import reduce_memory_usage


class MemoryController:
    """Manages memory cleanup and optimization."""

    def __init__(self, tmp_dir: str = "/tmp/Trellis-demo"):
        """Initialize the memory controller.

        Args:
            tmp_dir: Directory for temporary files
        """
        self.tmp_dir = tmp_dir
        self.cleanup_counter = 0

    def cleanup_temp_files(self, max_age_hours: int = 1) -> None:
        """
        Clean up old temporary files.

        Args:
            max_age_hours: Maximum age of files to keep in hours
        """
        try:
            import time
            from webui.initialize_pipeline import cleanup_temp_files
            cleanup_temp_files(max_age_hours=max_age_hours)
        except Exception as e:
            print(f"Warning: Failed to cleanup temp files: {e}")

    def reduce_memory(self) -> None:
        """Force garbage collection and CUDA cache cleanup."""
        reduce_memory_usage()

    def periodic_cleanup(self) -> None:
        """
        Perform periodic cleanup to prevent memory accumulation.
        Should be called regularly (e.g., every few generations).
        """
        self.cleanup_counter += 1
        if self.cleanup_counter > 1000:
            self.cleanup_counter = 0

        # Perform lightweight cleanup every 5 interactions
        if self.cleanup_counter % 5 == 0:
            self.reduce_memory()

        # Perform more aggressive cleanup every 20 interactions
        if self.cleanup_counter % 20 == 0:
            print(f"Performing periodic cleanup (interaction #{self.cleanup_counter})")
            self.cleanup_temp_files(max_age_hours=1)
            self.reduce_memory()
