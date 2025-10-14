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
    # Hide default streamlit elements for cleaner loading screen
    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    # Console output container (behind the glass overlay)
    console_output = st.empty()
    
    # Glassmorphic overlay banner
    st.markdown("""
    <style>
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .loading-container {
        position: relative;
        width: 100%;
        margin: 2rem auto;
        max-width: 1200px;
    }
    
    .glass-overlay {
        position: relative;
        width: 100%;
        padding: 3rem;
        background: rgba(102, 126, 234, 0.15);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        text-align: center;
        transition: all 0.5s ease;
        cursor: pointer;
        margin-bottom: 2rem;
    }
    
    .glass-overlay:hover {
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        background: rgba(102, 126, 234, 0.05);
    }
    
    .glass-overlay h1 {
        color: #667eea;
        font-size: 3rem;
        margin-bottom: 1rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .glass-overlay h2 {
        color: #764ba2;
        font-size: 1.8rem;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }
    
    .glass-overlay p {
        color: #333;
        font-size: 1.1rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    .loading-icon {
        font-size: 4rem;
        animation: pulse 2s ease-in-out infinite;
        margin: 1rem 0;
    }
    
    .hint-text {
        font-size: 0.9rem;
        color: #666;
        margin-top: 2rem;
        font-style: italic;
    }
    </style>
    
    <div class="loading-container">
        <div class="glass-overlay" onclick="this.style.opacity='0'; this.style.pointerEvents='none';">
            <h1>🚀 TRELLIS</h1>
            <h2>3D Generation Pipeline</h2>
            <p>
                Initializing AI models and CUDA environment<br>
                <strong>First run may take 2-5 minutes</strong>
            </p>
            <div class="loading-icon">⏳</div>
            <p class="hint-text">💡 Click or hover to see live terminal output</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress indicators below the glass overlay
    st.markdown("---")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
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
