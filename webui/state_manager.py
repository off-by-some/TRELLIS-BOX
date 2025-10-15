"""
State manager for TRELLIS UI.
Handles Streamlit session state - the only place that touches st.session_state.
"""

import streamlit as st
from typing import Optional, Dict, Any
from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline
from library.image_refiner import ImageRefiner
from library.models import ModelState


class StateManager:
    """Manages Streamlit session state with type safety."""

    # State keys
    PIPELINE = 'pipeline'
    REFINER = 'refiner'
    UPLOADED_IMAGE = 'uploaded_image'
    PROCESSED_PREVIEW = 'processed_preview'
    GENERATED_VIDEO = 'generated_video'
    GENERATED_GLB = 'generated_glb'
    GENERATED_STATE = 'generated_state'
    CLEANUP_COUNTER = 'cleanup_counter'
    IS_GENERATING = 'is_generating'

    @staticmethod
    def initialize() -> None:
        """Initialize all required session state variables."""
        defaults = {
            StateManager.PIPELINE: None,
            StateManager.REFINER: None,
            StateManager.UPLOADED_IMAGE: None,
            StateManager.PROCESSED_PREVIEW: None,
            StateManager.GENERATED_VIDEO: None,
            StateManager.GENERATED_GLB: None,
            StateManager.GENERATED_STATE: None,
            StateManager.CLEANUP_COUNTER: 0,
            StateManager.IS_GENERATING: False,
        }

        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    @staticmethod
    def get_pipeline() -> Optional[TrellisImageTo3DPipeline]:
        """Get the pipeline from session state."""
        return st.session_state.get(StateManager.PIPELINE)

    @staticmethod
    def set_pipeline(pipeline: TrellisImageTo3DPipeline) -> None:
        """Set the pipeline in session state."""
        st.session_state[StateManager.PIPELINE] = pipeline

    @staticmethod
    def get_refiner() -> Optional[ImageRefiner]:
        """Get the refiner from session state."""
        return st.session_state.get(StateManager.REFINER)

    @staticmethod
    def set_refiner(refiner: Optional[ImageRefiner]) -> None:
        """Set the refiner in session state."""
        st.session_state[StateManager.REFINER] = refiner

    @staticmethod
    def get_uploaded_image() -> Optional[Image.Image]:
        """Get the uploaded image."""
        return st.session_state.get(StateManager.UPLOADED_IMAGE)

    @staticmethod
    def set_uploaded_image(image: Optional[Image.Image]) -> None:
        """Set the uploaded image."""
        st.session_state[StateManager.UPLOADED_IMAGE] = image

    @staticmethod
    def get_processed_preview() -> Optional[Image.Image]:
        """Get the processed preview image."""
        return st.session_state.get(StateManager.PROCESSED_PREVIEW)

    @staticmethod
    def set_processed_preview(image: Optional[Image.Image]) -> None:
        """Set the processed preview image."""
        st.session_state[StateManager.PROCESSED_PREVIEW] = image

    @staticmethod
    def get_generated_video() -> Optional[str]:
        """Get the generated video path."""
        return st.session_state.get(StateManager.GENERATED_VIDEO)

    @staticmethod
    def set_generated_video(video_path: Optional[str]) -> None:
        """Set the generated video path."""
        st.session_state[StateManager.GENERATED_VIDEO] = video_path

    @staticmethod
    def get_generated_glb() -> Optional[str]:
        """Get the generated GLB path."""
        return st.session_state.get(StateManager.GENERATED_GLB)

    @staticmethod
    def set_generated_glb(glb_path: Optional[str]) -> None:
        """Set the generated GLB path."""
        st.session_state[StateManager.GENERATED_GLB] = glb_path

    @staticmethod
    def get_generated_state() -> Optional[Dict[str, Any]]:
        """Get the generated model state."""
        return st.session_state.get(StateManager.GENERATED_STATE)

    @staticmethod
    def set_generated_state(state: Optional[Dict[str, Any]]) -> None:
        """Set the generated model state."""
        st.session_state[StateManager.GENERATED_STATE] = state

    @staticmethod
    def clear_generated_content() -> None:
        """Clear all generated content from session state."""
        keys_to_clear = [
            StateManager.GENERATED_VIDEO,
            StateManager.GENERATED_GLB,
            StateManager.GENERATED_STATE,
            StateManager.UPLOADED_IMAGE,
            StateManager.PROCESSED_PREVIEW,
        ]
        for key in keys_to_clear:
            st.session_state[key] = None

    @staticmethod
    def increment_cleanup_counter() -> int:
        """Increment and return the cleanup counter."""
        counter = st.session_state.get(StateManager.CLEANUP_COUNTER, 0)
        counter += 1
        if counter > 1000:
            counter = 0
        st.session_state[StateManager.CLEANUP_COUNTER] = counter
        return counter

    @staticmethod
    def is_generating() -> bool:
        """Check if a generation is currently in progress."""
        return st.session_state.get(StateManager.IS_GENERATING, False)

    @staticmethod
    def set_generating(generating: bool) -> None:
        """Set the generation state."""
        st.session_state[StateManager.IS_GENERATING] = generating

    @staticmethod
    def get_resize_width() -> int:
        """Get resize width setting."""
        return st.session_state.get("resize_width", 518)

    @staticmethod
    def set_resize_width(width: int) -> None:
        """Set resize width setting."""
        st.session_state["resize_width"] = width

    @staticmethod
    def get_resize_height() -> int:
        """Get resize height setting."""
        return st.session_state.get("resize_height", 518)

    @staticmethod
    def set_resize_height(height: int) -> None:
        """Set resize height setting."""
        st.session_state["resize_height"] = height
