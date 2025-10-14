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


def show_loading_screen(gpu_info="Unknown GPU"):
    """Display the loading screen while initializing TRELLIS pipeline."""
    import base64
    from pathlib import Path
    
    # Load and encode the banner image
    banner_path = Path("docs/trellis-docker-image.png")
    if banner_path.exists():
        with open(banner_path, "rb") as f:
            banner_data = base64.b64encode(f.read()).decode()
        banner_img = f'data:image/png;base64,{banner_data}'
    else:
        banner_img = None
    
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
    
    # Glassmorphic overlay banner with GPU info
    st.markdown(f"""
    <style>
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.8; }}
    }}
    
    .loading-container {{
        position: relative;
        width: 100%;
        margin: 2rem auto;
        max-width: 1200px;
        min-height: 400px;
    }}
    
    .glass-overlay {{
        position: fixed;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        width: 90%;
        max-width: 840px;
        padding: 2rem;
        background: rgba(102, 126, 234, 0.15);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        text-align: center;
        transition: all 0.5s ease;
        cursor: pointer;
        z-index: 9999;
    }}
    
    .glass-overlay:hover {{
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        background: rgba(102, 126, 234, 0.05);
    }}
    
    .banner-image {{
        max-width: 280px;
        max-height: 280px;
        width: 100%;
        height: auto;
        margin: 0 auto 1rem auto;
        display: block;
        animation: pulse 2s ease-in-out infinite;
        object-fit: contain;
    }}
    
    .glass-overlay p {{
        color: #333;
        font-size: 0.88rem;
        line-height: 1.6;
        margin-bottom: 0.8rem;
    }}
    
    .loading-icon {{
        font-size: 3.2rem;
        animation: pulse 2s ease-in-out infinite;
        margin: 0.8rem 0;
    }}
    
    .gpu-info {{
        font-size: 0.8rem;
        color: #4a5568;
        margin-top: 1.2rem;
        padding: 0.6rem 1.2rem;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 8px;
        display: inline-block;
        font-weight: 500;
    }}
    
    .gpu-info strong {{
        color: #667eea;
    }}
    </style>
    
    <div class="loading-container">
        <div class="glass-overlay" onclick="this.style.opacity='0'; this.style.pointerEvents='none';">
            {'<img src="' + banner_img + '" class="banner-image" alt="TRELLIS Banner">' if banner_img else '<h1 style="color: #667eea; font-size: 2.4rem; margin-bottom: 0.8rem;">TRELLIS</h1>'}
            <p>
                Initializing AI models and CUDA environment<br>
                <strong>First run may take 2-5 minutes</strong>
            </p>
            <div class="gpu-info">
                <strong>GPU:</strong> {gpu_info}
            </div>
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
    
    # Use lists to accumulate all output
    output_lines = []
    
    class TeeOutput:
        def __init__(self, original, output_lines, console_display):
            self.original = original
            self.output_lines = output_lines
            self.console_display = console_display
            self.current_line = ""
            
        def write(self, text):
            self.original.write(text)
            self.original.flush()
            
            # Accumulate text
            self.current_line += text
            
            # If we have newlines, add complete lines to our list
            if '\n' in self.current_line:
                lines = self.current_line.split('\n')
                # Add all complete lines
                for line in lines[:-1]:
                    if line.strip():  # Only add non-empty lines
                        self.output_lines.append(line)
                # Keep the last incomplete line
                self.current_line = lines[-1]
                
                # Update display with accumulated output
                if self.console_display and self.output_lines:
                    # Show last 50 lines
                    display_text = '\n'.join(self.output_lines[-50:])
                    if self.current_line:
                        display_text += '\n' + self.current_line
                    self.console_display.code(display_text, language='bash')
            
        def flush(self):
            self.original.flush()
    
    try:
        tee_out = TeeOutput(old_stdout, output_lines, console_display)
        tee_err = TeeOutput(old_stderr, output_lines, console_display)
        sys.stdout = tee_out
        sys.stderr = tee_err
        yield output_lines
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
