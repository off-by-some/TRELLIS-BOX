"""
Main application UI for TRELLIS.
Orchestrates the UI components and manages application flow.
"""

import streamlit as st
import time

from webui.controllers import AppController
from webui.state_manager import StateManager
from webui.single_image_ui import SingleImageUI
from webui.multi_image_ui import MultiImageUI
from webui.loading_screen import show_loading_screen, finalize_loading, capture_output


class TrellisApp:
    """Main application UI orchestrator."""

    def __init__(self, controller: AppController):
        """Initialize the Trellis application UI."""
        self.controller = controller
        self._configure_page()

    @staticmethod
    def _configure_page() -> None:
        """Configure Streamlit page settings and styles."""
        st.set_page_config(
            page_title="TRELLIS 3D Generator",
            page_icon="🎨",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Remove default Streamlit padding and margins
        st.markdown("""
            <style>
            /* Remove top padding */
            .block-container {
                padding-top: 1rem !important;
                padding-bottom: 0rem !important;
            }

            /* Remove default header space */
            header {
                background-color: transparent !important;
            }

            /* Reduce spacing between elements */
            .element-container {
                margin-bottom: 0.5rem !important;
            }

            /* Make the app use full viewport */
            .main .block-container {
                max-width: 100%;
                padding-left: 2rem;
                padding-right: 2rem;
            }
            </style>
        """, unsafe_allow_html=True)

    def _check_gpu(self) -> str:
        """
        Check for GPU availability and return GPU info.

        Returns:
            GPU info string

        Raises:
            SystemExit: If no CUDA GPU is available
        """
        return self.controller.check_gpu()

    def _initialize_pipeline(self) -> None:
        """Initialize or load the TRELLIS pipeline with UI."""
        StateManager.initialize()

        pipeline = StateManager.get_pipeline()

        if pipeline is None:
            # Check if pipeline is already cached
            from webui.initialize_pipeline import _PIPELINE_SINGLETON

            if _PIPELINE_SINGLETON is not None:
                print("Using cached pipeline from previous session")
                pipeline = _PIPELINE_SINGLETON
                StateManager.set_pipeline(pipeline)
                self.controller.pipeline = pipeline
                st.rerun()

            # Pipeline not cached, show full loading screen
            gpu_info = self._check_gpu()

            # Show loading screen
            progress_bar, status_text, console_output, start_time = show_loading_screen(gpu_info)

            status_text.text("Loading TRELLIS pipeline...")
            progress_bar.progress(10)

            try:
                with capture_output(console_output):
                    pipeline = self.controller.initialize_pipeline()

                StateManager.set_pipeline(pipeline)

                # Complete loading UI
                finalize_loading(progress_bar, status_text, pipeline)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    progress_bar.progress(0)
                    status_text.text("Error: Out of GPU memory")
                    st.error("❌ Out of GPU Memory")
                    st.error(str(e))
                    st.warning("**Solution:** Please restart the application to clear GPU memory.")
                    st.code("docker-compose restart", language="bash")
                    st.info("If the problem persists, there may be other processes using GPU memory. Check with `nvidia-smi`")
                    st.stop()
                else:
                    raise

    def run(self) -> None:
        """Run the main application."""
        # Perform periodic cleanup
        self.controller.periodic_cleanup()

        # Initialize pipeline if needed
        self._initialize_pipeline()

        # Display title and description
        st.title("Image to 3D Asset with TRELLIS")
        st.markdown("""
        * **Single Image**: Upload one image for standard 3D generation
        * **Multi-Image**: Upload 2-4 images from different views for enhanced 3D reconstruction
        * If images have alpha channels, they'll be used as masks. Otherwise, we use `rembg` to remove backgrounds.
        """)

        # Create tabs
        tab1, tab2 = st.tabs(["Single Image", "Multi-Image"])

        with tab1:
            SingleImageUI.render(self.controller)

        with tab2:
            MultiImageUI.render(self.controller)
