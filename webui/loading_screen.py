# Loading screen module for TRELLIS 3D Generator

import streamlit as st
import time
import warnings

# Suppress warnings during loading
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*deprecated.*")
warnings.filterwarnings("ignore", message=".*xFormers is available.*")
warnings.filterwarnings("ignore", message=".*torch.library.impl_abstract.*renamed.*")
warnings.filterwarnings("ignore", message=".*torch.library.register_fake.*")


def show_loading_screen():
    """Display the loading screen while initializing TRELLIS pipeline."""
    # Show loading screen with spinner
    st.title("🚀 TRELLIS 3D Generator")
    st.markdown("## Initializing Application...")

    # Create a nice loading interface
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div style="
            text-align: center;
            padding: 2rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            color: white;
            margin: 2rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h2 style="margin-bottom: 1rem;">Initializing TRELLIS</h2>
            <p style="margin-bottom: 1rem; opacity: 0.9;">
                Loading AI models and preparing 3D generation pipeline.<br>
                This one-time setup may take 2-5 minutes.
            </p>
            <div style="font-size: 3rem;">⏳</div>
        </div>
        """, unsafe_allow_html=True)

        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Show initialization steps
        steps = [
            "Setting up PyTorch and CUDA environment...",
            "Downloading TRELLIS model weights...",
            "Initializing image processing pipeline...",
            "Configuring memory optimizations...",
            "Loading Stable Diffusion components...",
            "Preparing 3D reconstruction pipeline..."
        ]

        # Use a spinner for the actual loading
        with st.spinner("Loading AI models... This may take several minutes."):
            start_time = time.time()

            # Update progress for each step
            for i, step in enumerate(steps):
                status_text.text(step)
                progress_bar.progress(int((i + 1) / len(steps) * 80))  # Leave room for final step
                time.sleep(0.5)

            return progress_bar, status_text, start_time


def finalize_loading(progress_bar, status_text, pipeline):
    """Complete the loading UI after pipeline is loaded."""
    progress_bar.progress(100)
    status_text.text("Application ready")
    
    st.success("TRELLIS initialization completed")
    st.balloons()
    
    time.sleep(2)
    st.rerun()
