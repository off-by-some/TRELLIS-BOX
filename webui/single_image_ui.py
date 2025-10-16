"""
Single image UI view for TRELLIS application.
Pure view that handles rendering and user interaction.
"""

import streamlit as st
import os
import time
import uuid
from typing import Optional, List, Any
from PIL import Image

from webui.state_manager import StateManager
from webui.controllers import AppController
from library.models import GenerationParams, ExportParams
from webui.ui_components import show_video_preview, show_3d_model_viewer
from webui.ui_components import show_example_gallery


class SingleImageUI:
    """Handles the single image generation UI."""

    @staticmethod
    def render(controller: AppController) -> None:
        """Render the single image generation interface."""
        st.header("Single Image Generation")

        col1, col2 = st.columns(2)

        with col1:
            SingleImageUI._render_input_column(controller)

        with col2:
            SingleImageUI._render_output_column(controller)

        # Examples section
        SingleImageUI._render_examples(controller)

    @staticmethod
    def _render_input_column(controller: AppController) -> None:
        """Render the input column."""
        st.subheader("Input")

        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg"],
            key="single_image",
            label_visibility="visible"
        )

        # Handle uploaded image
        if uploaded_file is not None:
            new_image = Image.open(uploaded_file)
            current_image = StateManager.uploaded_image

            if current_image is None or current_image != new_image:
                StateManager.set_uploaded_image(new_image)
                StateManager.set_processed_preview(None)
                StateManager.set_generated_video(None)
                StateManager.set_generated_glb(None)
                StateManager.set_generated_state(None)
                st.rerun()

        # Show uploaded image
        uploaded_image = StateManager.get_uploaded_image()
        if uploaded_image is not None:
            # Image preprocessing options
            with st.expander("Image Preprocessing Options", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    use_refinement = st.checkbox(
                        "Apply Image Refinement (SSD-1B)",
                        value=False,
                        help="Enhance input quality with SSD-1B after background removal. Adds ~5-7s processing time.",
                        key="refinement_single_input"
                    )

                with col2:
                    valid_sizes = [i * 14 for i in range(19, 74)]  # 266 to 1022

                    resize_width = st.selectbox(
                        "Resize Width",
                        options=valid_sizes,
                        index=valid_sizes.index(518) if 518 in valid_sizes else 0,
                        key="resize_width_single",
                        help="Width to resize images to for conditioning model (must be multiple of 14)",
                        format_func=lambda x: f"{x}px"
                    )
                    StateManager.set_resize_width(resize_width)

                    resize_height = st.selectbox(
                        "Resize Height",
                        options=valid_sizes,
                        index=valid_sizes.index(518) if 518 in valid_sizes else 0,
                        key="resize_height_single",
                        help="Height to resize images to for conditioning model (must be multiple of 14)",
                        format_func=lambda x: f"{x}px"
                    )
                    StateManager.set_resize_height(resize_height)

            st.markdown("**Uploaded Image:**")
            st.image(uploaded_image, use_container_width=True)

            # Auto-process and show final processed preview
            pipeline = controller.get_pipeline()
            if pipeline is not None and uploaded_image is not None:
                current_width = StateManager.get_resize_width()
                current_height = StateManager.get_resize_height()
                target_size = (current_width, current_height)

                processed_image = StateManager.get_processed_preview()
                needs_regeneration = (
                    processed_image is None or
                    StateManager.processed_preview_size != target_size or
                    StateManager.current_refinement_setting != use_refinement
                )

                if needs_regeneration:
                    with st.spinner("Processing image..."):
                        result = controller.process_image(uploaded_image, use_refinement)
                        processed_image = result.processed_images
                        StateManager.set_processed_preview(processed_image)
                        st.session_state.processed_preview_size = target_size
                        st.session_state.current_refinement_setting = use_refinement

                if processed_image is not None:
                    preview_label = "**Processed Preview**"
                    if use_refinement:
                        preview_label += " *(with refinement)*"
                    else:
                        preview_label += " *(background removed)*"
                    st.markdown(preview_label)
                    st.image(processed_image, use_container_width=True)
            else:
                st.info("Processed preview will be shown after pipeline loads")

    @staticmethod
    def _render_output_column(controller: AppController) -> None:
        """Render the output column."""
        st.subheader("Output")

        uploaded_image = StateManager.get_uploaded_image()
        SingleImageUI._render_generation_panel(
            controller=controller,
            uploaded_data=uploaded_image,
            is_multi_image=False,
            video_key="single_video",
            glb_key="single_glb",
            download_key="download_single",
            generate_key="generate_single",
            seed_key="seed_single",
            randomize_key="randomize_single",
            ss_strength_key="ss_strength_single",
            ss_steps_key="ss_steps_single",
            slat_strength_key="slat_strength_single",
            slat_steps_key="slat_steps_single",
            simplify_key="simplify_single",
            texture_key="texture_single",
            trial_id="single"
        )

    @staticmethod
    def _render_generation_panel(
        controller: AppController,
        uploaded_data: Any,
        is_multi_image: bool,
        video_key: str,
        glb_key: str,
        download_key: str,
        generate_key: str,
        seed_key: str,
        randomize_key: str,
        ss_strength_key: str,
        ss_steps_key: str,
        slat_strength_key: str,
        slat_steps_key: str,
        simplify_key: str,
        texture_key: str,
        batch_size_key: str = None,
        trial_id: str = None
    ) -> None:
        """Render complete generation panel."""
        # Generate trial_id if not provided
        if trial_id is None:
            trial_id = str(uuid.uuid4())

        # Check if we have valid input for generation
        has_valid_input = uploaded_data is not None if not is_multi_image else (uploaded_data and len(uploaded_data) >= 2)

        # Settings
        if has_valid_input:
            st.subheader("⚙️ Generation Settings")

            # Quality presets
            quality_presets = {
                "Fast (Low Quality)": {"ss_strength": 5.0, "ss_steps": 8, "slat_strength": 2.0, "slat_steps": 8},
                "Balanced": {"ss_strength": 7.5, "ss_steps": 12, "slat_strength": 3.0, "slat_steps": 12},
                "High Quality (Recommended)": {"ss_strength": 10.0, "ss_steps": 20, "slat_strength": 4.0, "slat_steps": 20},
                "Maximum Quality (Slow)": {"ss_strength": 12.0, "ss_steps": 30, "slat_strength": 5.0, "slat_steps": 30}
            }

            quality_preset = st.selectbox(
                "Choose Quality Level",
                options=list(quality_presets.keys()),
                index=1,
                key=f"quality_preset_{trial_id}",
                help="Higher quality = better details but slower generation."
            )

            preset_settings = quality_presets[quality_preset]

            seed = st.slider("Seed", 0, controller.MAX_SEED, 0, 1, key=seed_key)
            randomize_seed = st.checkbox("Randomize Seed", value=True, key=randomize_key)

            # Advanced settings
            with st.expander("Advanced Settings", expanded=False):
                st.markdown("**Sparse Structure Generation**")
                ss_strength = st.slider("Guidance Strength", 0.0, 15.0, preset_settings["ss_strength"], 0.1, key=ss_strength_key)
                ss_sampling_steps = st.slider("Sampling Steps", 1, 50, preset_settings["ss_steps"], 1, key=ss_steps_key)

                st.markdown("**Structured Latent Generation**")
                slat_strength = st.slider("Guidance Strength", 0.0, 10.0, preset_settings["slat_strength"], 0.1, key=slat_strength_key)
                slat_sampling_steps = st.slider("Sampling Steps", 1, 50, preset_settings["slat_steps"], 1, key=slat_steps_key)

            # Export settings
            with st.expander("GLB Export Settings", expanded=False):
                mesh_quality_options = {
                    "Low (Fast)": {"simplify": 0.85, "texture": 512},
                    "Medium (Balanced)": {"simplify": 0.90, "texture": 1024},
                    "High (Detailed)": {"simplify": 0.95, "texture": 1024},
                    "Premium (Max Quality)": {"simplify": 0.98, "texture": 2048}
                }

                mesh_quality = st.selectbox(
                    "Mesh Quality",
                    options=list(mesh_quality_options.keys()),
                    index=2,
                    key=f"mesh_quality_{trial_id}"
                )

                quality_settings = mesh_quality_options[mesh_quality]
                mesh_simplify = st.slider("Mesh Simplify Ratio", 0.8, 0.99, quality_settings["simplify"], 0.01, key=simplify_key)

                texture_size_options = [256, 512, 1024, 2048, 4096]
                texture_size = st.selectbox(
                    "Texture Size",
                    options=texture_size_options,
                    index=texture_size_options.index(quality_settings["texture"]),
                    key=texture_key,
                    format_func=lambda x: f"{x}px"
                )

                fill_holes_resolution = st.selectbox(
                    "Hole Fill Resolution",
                    options=[256, 512, 1024, 2048, 4096],
                    index=2,
                    key=f"fill_res_{trial_id}",
                    format_func=lambda x: f"{x}px"
                )

                fill_holes_num_views = st.slider("Hole Fill Views", 100, 4000, 1000, 100, key=f"fill_views_{trial_id}")

            # Generate button
            is_generating = StateManager.is_generating()
            has_generated = StateManager.get_generated_video() is not None

            button_label = "🔄 Regenerate 3D Model" if has_generated else "Generate 3D Model"
            button_disabled = is_generating

            if st.button(button_label, type="primary", key=generate_key, use_container_width=True, disabled=button_disabled):
                try:
                    StateManager.set_generating(True)

                    with st.spinner("Generating 3D model..."):
                        # Get processed preview
                        processed_result = StateManager.get_processed_preview()
                        if processed_result is None:
                            with StateManager.refinement_single_input as refinement_input:
                                processed_result = controller.process_image(uploaded_data, refinement_input)
                            StateManager.set_processed_preview(processed_result.processed_images)

                        # Create generation params
                        params = controller.create_generation_params(
                            seed=seed if not randomize_seed else 0,
                            randomize_seed=randomize_seed,
                            ss_guidance_strength=st.session_state.get(ss_strength_key, preset_settings["ss_strength"]),
                            ss_sampling_steps=st.session_state.get(ss_steps_key, preset_settings["ss_steps"]),
                            slat_guidance_strength=st.session_state.get(slat_strength_key, preset_settings["slat_strength"]),
                            slat_sampling_steps=st.session_state.get(slat_steps_key, preset_settings["slat_steps"])
                        )

                        # Generate
                        result = controller.generate_single(
                            processed_result.trial_id,
                            params,
                            (StateManager.get_resize_width(), StateManager.get_resize_height())
                        )

                        # Export GLB
                        export_params = controller.create_export_params(
                            mesh_simplify=mesh_simplify,
                            texture_size=texture_size,
                            fill_holes_resolution=fill_holes_resolution,
                            fill_holes_num_views=fill_holes_num_views
                        )

                        glb_result = controller.export_model(result.model_state, export_params)

                        # Update state
                        StateManager.set_generated_video(result.video_path)
                        StateManager.set_generated_glb(glb_result.glb_path)
                        StateManager.set_generated_state(result.model_state)

                        st.success("✅ 3D model complete!")
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Generation failed: {str(e)}")
                finally:
                    StateManager.set_generating(False)

        # Output preview
        st.markdown("---")
        SingleImageUI._render_output_preview(video_key, glb_key, download_key)

    @staticmethod
    def _render_output_preview(video_key: str, glb_key: str, download_key: str) -> None:
        """Render output preview section."""
        # Video preview
        with st.container():
            is_generating = StateManager.is_generating()
            clear_video = show_video_preview(
                StateManager.get_generated_video(),
                show_clear=True,
                clear_key=video_key,
                show_progress=is_generating,
                progress_text="Generating 3D model..." if is_generating else None
            )
            if clear_video == "clear":
                StateManager.set_generated_video(None)
                StateManager.set_generated_glb(None)
                StateManager.set_generated_state(None)
                st.rerun()

        # 3D model viewer
        generated_video = StateManager.get_generated_video()
        generated_glb = StateManager.get_generated_glb()
        generated_state = StateManager.get_generated_state()

        if generated_glb and generated_video and generated_state:
            st.success("✅ 3D Model Ready!")

            clear_glb = show_3d_model_viewer(
                generated_glb,
                show_clear=True,
                clear_key=glb_key
            )
            if clear_glb == "clear":
                StateManager.set_generated_glb(None)
                StateManager.set_generated_state(None)
                st.rerun()

            # Download button
            with open(generated_glb, "rb") as file:
                st.download_button(
                    label="📥 Download GLB",
                    data=file,
                    file_name="generated_model.glb",
                    mime="model/gltf-binary",
                    type="primary",
                    key=download_key
                )
        else:
            show_3d_model_viewer(None)

    @staticmethod
    def _render_examples(controller: AppController) -> None:
        """Render the examples section."""
        st.subheader("Examples")
        example_images = sorted([
            f'assets/example_image/{img}'
            for img in os.listdir("assets/example_image")
            if img.endswith(('.png', '.jpg', '.jpeg'))
        ])

        selected_example = show_example_gallery(example_images, columns=4)
        if selected_example:
            example_img = Image.open(selected_example)
            # Resize large images
            max_size = 512
            if max(example_img.size) > max_size:
                ratio = max_size / max(example_img.size)
                new_size = tuple(int(dim * ratio) for dim in example_img.size)
                example_img = example_img.resize(new_size, Image.Resampling.LANCZOS)

            StateManager.set_uploaded_image(example_img)
            StateManager.set_processed_preview(None)
            StateManager.set_generated_video(None)
            StateManager.set_generated_glb(None)
            StateManager.set_generated_state(None)
            st.rerun()
