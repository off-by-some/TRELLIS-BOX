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
    
    /* Main container with flexbox layout */
    .main .block-container {
        display: flex;
        flex-direction: column;
        height: 100vh;
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Header container */
    .main .block-container > div:nth-child(1) {
        flex-shrink: 0;
    }
    
    /* Terminal container (grows to fill space) */
    .main .block-container > div:nth-child(2) {
        flex: 1 1 auto;
        display: flex;
        flex-direction: column;
        min-height: 0;
        background: #0d1117;
        border-radius: 0;
        border: none;
        overflow: hidden;
    }
    
    /* Progress container (stays at bottom) */
    .main .block-container > div:nth-child(3) {
        flex-shrink: 0;
        padding: 1.5rem 2rem;
        background: #161b22;
        border-top: 1px solid #30363d;
    }
    
    /* Terminal content - the streamlit empty element */
    .main .block-container > div:nth-child(2) > div > div:last-child {
        flex: 1 1 auto;
        display: flex;
        flex-direction: column;
        overflow-y: auto;
        padding: 1.5rem 2rem;
        min-height: 0;
    }
    
    /* Style code blocks inside terminal */
    .main .block-container > div:nth-child(2) pre {
        flex: 1 1 auto;
        margin: 0 !important;
        padding: 1.5rem !important;
        background: #0d1117 !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', 'Consolas', monospace !important;
        font-size: 0.875rem !important;
        line-height: 1.6 !important;
        color: #c9d1d9 !important;
        overflow-y: auto !important;
        min-height: 0 !important;
    }
    
    .main .block-container > div:nth-child(2) code {
        color: #c9d1d9 !important;
        background: transparent !important;
    }
    
    /* Custom scrollbar for terminal */
    .main .block-container > div:nth-child(2) pre::-webkit-scrollbar {
        width: 10px;
    }
    
    .main .block-container > div:nth-child(2) pre::-webkit-scrollbar-track {
        background: #0d1117;
    }
    
    .main .block-container > div:nth-child(2) pre::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 5px;
    }
    
    .main .block-container > div:nth-child(2) pre::-webkit-scrollbar-thumb:hover {
        background: #484f58;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Beautiful header banner with GPU info
    st.markdown(f"""
    <style>
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(-20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    @keyframes shimmer {{
        0% {{ background-position: -1000px 0; }}
        100% {{ background-position: 1000px 0; }}
    }}
    
    .loading-header {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem 2.5rem 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: fadeIn 0.6s ease-out;
    }}
    
    .loading-header::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.1),
            transparent
        );
        animation: shimmer 3s infinite;
    }}
    
    .header-content {{
        position: relative;
        z-index: 1;
    }}
    
    .banner-image {{
        max-width: 200px;
        max-height: 200px;
        width: 100%;
        height: auto;
        margin: 0 auto 1.5rem auto;
        display: block;
        filter: drop-shadow(0 10px 30px rgba(0, 0, 0, 0.3));
        object-fit: contain;
    }}
    
    .loading-title {{
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        letter-spacing: -0.5px;
    }}
    
    .loading-subtitle {{
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        font-weight: 400;
        margin: 0 0 1.5rem 0;
        line-height: 1.6;
    }}
    
    .gpu-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1.5rem;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 50px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #ffffff;
        font-size: 0.9rem;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }}
    
    .gpu-icon {{
        width: 20px;
        height: 20px;
        fill: currentColor;
    }}
    </style>
    
    <div class="loading-header">
        <div class="header-content">
            {'<img src="' + banner_img + '" class="banner-image" alt="TRELLIS">' if banner_img else ''}
            <h1 class="loading-title">TRELLIS Pipeline Initialization</h1>
            <p class="loading-subtitle">
                Loading AI models and CUDA environment<br>
                <strong>First run may take 2-5 minutes</strong>
            </p>
            <div class="gpu-badge">
                <svg class="gpu-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path d="M4 6h16v2H4zm0 5h16v2H4zm0 5h16v2H4z"/>
                </svg>
                <span>{gpu_info}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create containers for layout
    terminal_container = st.container()
    progress_container = st.container()
    
    # Terminal output
    with terminal_container:
        console_output = st.empty()
    
    # Progress indicators at the bottom
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    start_time = time.time()
    return progress_bar, status_text, console_output, start_time


@contextlib.contextmanager
def capture_output(console_display):
    """Context manager to capture stdout/stderr and display in Streamlit like a real terminal."""
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
                # Add all complete lines (including empty ones for proper spacing)
                for line in lines[:-1]:
                    self.output_lines.append(line)
                # Keep the last incomplete line
                self.current_line = lines[-1]
                
                # Update display with accumulated output - like a real terminal
                if self.console_display and self.output_lines:
                    # Show last 100 lines (more than before for better context)
                    display_lines = self.output_lines[-100:]
                    
                    # Add current incomplete line if exists
                    if self.current_line.strip():
                        display_lines.append(self.current_line)
                    
                    # Join and display - newest lines are at the bottom
                    display_text = '\n'.join(display_lines)
                    
                    # Display with terminal-like styling using code block
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
