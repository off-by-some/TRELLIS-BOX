# Loading screen module for TRELLIS 3D Generator

import streamlit as st
import time
import warnings
import sys
from io import StringIO
import contextlib

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
                This may take 2-5 minutes on first run.
            </p>
            <div style="font-size: 3rem;">⏳</div>
        </div>
        """, unsafe_allow_html=True)

        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Console output display
        with st.expander("📋 Loading Details", expanded=False):
            console_output = st.empty()

        start_time = time.time()
        return progress_bar, status_text, console_output, start_time


@contextlib.contextmanager
def capture_output(console_display):
    """Context manager to capture stdout/stderr and display in Streamlit."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    
    class TeeOutput:
        def __init__(self, original, buffer, console_display):
            self.original = original
            self.buffer = buffer
            self.console_display = console_display
            
        def write(self, text):
            self.original.write(text)
            self.buffer.write(text)
            # Update console display with accumulated output
            if self.console_display and text.strip():
                full_output = self.buffer.getvalue()
                # Show last 50 lines
                lines = full_output.split('\n')
                display_text = '\n'.join(lines[-50:])
                self.console_display.code(display_text, language='bash')
            
        def flush(self):
            self.original.flush()
    
    try:
        sys.stdout = TeeOutput(old_stdout, stdout_buffer, console_display)
        sys.stderr = TeeOutput(old_stderr, stderr_buffer, console_display)
        yield stdout_buffer, stderr_buffer
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def finalize_loading(progress_bar, status_text, pipeline):
    """Complete the loading UI after pipeline is loaded."""
    progress_bar.progress(100)
    status_text.text("Application ready")
    
    st.success("TRELLIS initialization completed")
    st.balloons()
    
    time.sleep(2)
    st.rerun()
