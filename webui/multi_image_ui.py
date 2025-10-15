"""
Multi-image UI view for TRELLIS application.
Pure view that handles rendering and user interaction.
"""

import streamlit as st
from typing import Any, List, Optional
from PIL import Image

from webui.state_manager import StateManager
from webui.controllers import AppController
from webui.ui_components import show_video_preview, show_3d_model_viewer


class MultiImageUI:
    """Handles the multi-image generation UI."""

    @staticmethod
    def render(controller: AppController) -> None:
        """Render the multi-image generation interface."""
        st.header("Multi-Image Generation")
        st.markdown("Upload 2-4 images from different viewpoints for improved 3D reconstruction")

        col1, col2 = st.columns(2)

        with col1:
            MultiImageUI._render_input_column(controller)

        with col2:
            MultiImageUI._render_output_column(controller)

    @staticmethod
    def _render_input_column(controller: AppController) -> None:
        """Render the input column."""
        st.subheader("Input")

        # File uploader
        multi_uploaded_files = st.file_uploader(
            "Upload Images (2-4 images)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="multi_images",
            label_visibility="visible"
        )

        if multi_uploaded_files:
            if len(multi_uploaded_files) < 2:
                st.warning("Please upload at least 2 images")
            elif len(multi_uploaded_files) > 4:
                st.warning("Maximum 4 images allowed. Using first 4.")
                multi_uploaded_files = multi_uploaded_files[:4]

            if len(multi_uploaded_files) >= 2:
                # Image preprocessing options
                with st.expander("Image Preprocessing Options", expanded=True):
                    col1, col2 = st.columns(2)

                    with col1:
                        use_refinement = st.checkbox(
                            "Apply Image Refinement (SSD-1B)",
                            value=False,
                            help="Enhance input quality with SSD-1B after background removal. Adds ~5-7s per image.",
                            key="refinement_multi_input"
                        )

                    with col2:
                        valid_sizes = [i * 14 for i in range(19, 74)]

                        resize_width = st.selectbox(
                            "Resize Width",
                            options=valid_sizes,
                            index=valid_sizes.index(518) if 518 in valid_sizes else 0,
                            key="resize_width_multi",
                            help="Width to resize images to for conditioning model (must be multiple of 14)",
                            format_func=lambda x: f"{x}px"
                        )
                        StateManager.set_resize_width(resize_width)

                        resize_height = st.selectbox(
                            "Resize Height",
                            options=valid_sizes,
                            index=valid_sizes.index(518) if 518 in valid_sizes else 0,
                            key="resize_height_multi",
                            help="Height to resize images to for conditioning model (must be multiple of 14)",
                            format_func=lambda x: f"{x}px"
                        )
                        StateManager.set_resize_height(resize_height)

                st.markdown("**Uploaded Images:**")
                for i, uploaded_file in enumerate(multi_uploaded_files):
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"Image {i+1}", use_container_width=True)

                # Show processed previews
                pipeline = StateManager.get_pipeline()
                if pipeline is not None:
                    current_width = StateManager.get_resize_width()
                    current_height = StateManager.get_resize_height()

                    preview_label = "**Processed Previews**"
                    if use_refinement:
                        preview_label += " *(with refinement)*"
                    else:
                        preview_label += " *(background removed)*"
                    st.markdown(preview_label)

                    for i, uploaded_file in enumerate(multi_uploaded_files):
                        image = Image.open(uploaded_file)
                        # Note: In a full implementation, you'd want to cache these processed images
                        # For now, just show the originals as placeholders
                        st.image(image, caption=f"Processed {i+1}", use_container_width=True)
                else:
                    st.info("Processed previews will be shown after pipeline loads")

    @staticmethod
    def _render_output_column(controller: AppController) -> None:
        """Render the output column."""
        st.subheader("Output")

        multi_uploaded_files = st.session_state.get("multi_images")
        if multi_uploaded_files is None:
            multi_uploaded_files = st.session_state.get("_preserved_multi_images")

        # Use the same generation panel as single image but adapted for multi-image
        from webui.single_image_ui import SingleImageUI
        SingleImageUI._render_generation_panel(
            controller=controller,
            uploaded_data=multi_uploaded_files,
            is_multi_image=True,
            video_key="multi_video",
            glb_key="multi_glb",
            download_key="download_multi",
            generate_key="generate_multi",
            seed_key="seed_multi",
            randomize_key="randomize_multi",
            ss_strength_key="ss_strength_multi",
            ss_steps_key="ss_steps_multi",
            slat_strength_key="slat_strength_multi",
            slat_steps_key="slat_steps_multi",
            simplify_key="simplify_multi",
            texture_key="texture_multi",
            batch_size_key="batch_size_multi",
            trial_id="multi"
        )
