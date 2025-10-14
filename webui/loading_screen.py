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

# CACHE banner image to prevent reloading on every refresh - CRITICAL for memory
_BANNER_CACHE = None

def _load_banner_image():
    """Load and cache the banner image to prevent memory leaks on refresh."""
    global _BANNER_CACHE
    
    if _BANNER_CACHE is not None:
        return _BANNER_CACHE
    
    import base64
    from pathlib import Path
    
    # Load and encode the banner image ONCE
    banner_path = Path("docs/trellis-docker-image.png")
    if banner_path.exists():
        with open(banner_path, "rb") as f:
            banner_data = base64.b64encode(f.read()).decode()
        _BANNER_CACHE = f'data:image/png;base64,{banner_data}'
    else:
        _BANNER_CACHE = None
    
    return _BANNER_CACHE


def show_loading_screen(gpu_info="Unknown GPU"):
    """Display the loading screen while initializing TRELLIS pipeline."""
    # Use cached banner image
    banner_img = _load_banner_image()
    
    # Hide default streamlit elements for cleaner loading screen
    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container with flexbox layout */
    .main .block-container {
        display: flex !important;
        flex-direction: column !important;
        height: 100vh !important;
        padding: 1rem !important;
    }
    
    /* Terminal header bar */
    .terminal-header {
        background: #161b22;
        padding: 0.5rem 1rem;
        border-bottom: 1px solid #30363d;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 0.85rem;
        color: #8b949e;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        flex-shrink: 0;
    }
    
    .terminal-dots {
        display: flex;
        gap: 0.4rem;
    }
    
    .terminal-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }
    
    .terminal-dot.red { background: #ff5f56; }
    .terminal-dot.yellow { background: #ffbd2e; }
    .terminal-dot.green { background: #27c93f; }
    
    /* Style code blocks inside terminal container */
    div[data-testid="stVerticalBlock"]:has(.terminal-header) pre {
        flex: 1 1 auto !important;
        margin: 0 !important;
        padding: 1rem !important;
        background: #0d1117 !important;
        border: none !important;
        border-radius: 0 !important;
        font-family: 'Courier New', 'Consolas', monospace !important;
        font-size: 0.85rem !important;
        line-height: 1.5 !important;
        color: #58a6ff !important;
        overflow-y: auto !important;
        min-height: 0 !important;
    }
    
    div[data-testid="stVerticalBlock"]:has(.terminal-header) code {
        color: #58a6ff !important;
        background: transparent !important;
    }
    
    /* Custom scrollbar for terminal */
    div[data-testid="stVerticalBlock"]:has(.terminal-header) pre::-webkit-scrollbar {
        width: 8px;
    }
    
    div[data-testid="stVerticalBlock"]:has(.terminal-header) pre::-webkit-scrollbar-track {
        background: #161b22;
    }
    
    div[data-testid="stVerticalBlock"]:has(.terminal-header) pre::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 4px;
    }
    
    div[data-testid="stVerticalBlock"]:has(.terminal-header) pre::-webkit-scrollbar-thumb:hover {
        background: #484f58;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Glassmorphic overlay banner with GPU info
    st.markdown(f"""
    <style>
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.8; }}
    }}
    
    .glass-overlay {{
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
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
    """, unsafe_allow_html=True)
    
    # Add custom styling to identify containers
    st.markdown("""
    <style>
    /* Target all direct children of block-container for proper layout */
    .main .block-container > .element-container {
        display: contents;
    }
    
    /* Terminal container styling */
    div[data-testid="stVerticalBlock"]:has(.terminal-header) {
        flex: 1 1 auto !important;
        display: flex !important;
        flex-direction: column !important;
        min-height: 0 !important;
        margin-bottom: 1rem !important;
        background: #0d1117 !important;
        border-radius: 8px !important;
        border: 1px solid #30363d !important;
        overflow: hidden !important;
    }
    
    /* Progress container styling */
    div[data-testid="stVerticalBlock"]:has(.stProgress) {
        flex-shrink: 0 !important;
        padding: 1rem !important;
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create containers for layout
    terminal_container = st.container()
    progress_container = st.container()
    
    # Terminal with header
    with terminal_container:
        st.markdown("""
        <div class="terminal-header">
            <div class="terminal-dots">
                <div class="terminal-dot red"></div>
                <div class="terminal-dot yellow"></div>
                <div class="terminal-dot green"></div>
            </div>
            <span>TRELLIS Pipeline Initialization</span>
        </div>
        """, unsafe_allow_html=True)
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
    
    # Use lists to accumulate output with BOUNDED SIZE to prevent memory leaks
    output_lines = []
    MAX_LINES = 100  # Hard limit to prevent unbounded growth
    
    class TeeOutput:
        def __init__(self, original, output_lines, console_display):
            self.original = original
            self.output_lines = output_lines
            self.console_display = console_display
            self.current_line = ""
            self.update_counter = 0  # Throttle display updates
            
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
                    # CRITICAL FIX: Keep only last MAX_LINES to prevent memory leak
                    if len(self.output_lines) > MAX_LINES:
                        self.output_lines.pop(0)
                
                # Keep the last incomplete line
                self.current_line = lines[-1]
                
                # THROTTLE updates: Only update display every 5 writes to reduce Streamlit element creation
                self.update_counter += 1
                if self.console_display and self.output_lines and self.update_counter % 5 == 0:
                    # Show last 100 lines (but list is already bounded)
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
        # CRITICAL: Final display update and cleanup
        if console_display and output_lines:
            display_text = '\n'.join(output_lines[-100:])
            console_display.code(display_text, language='bash')
        
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        # Clear references to help garbage collection
        output_lines.clear()
        del tee_out
        del tee_err


def finalize_loading(progress_bar, status_text, pipeline):
    """Complete the loading UI after pipeline is loaded."""
    # Handle None values for cleanup scenarios
    if progress_bar is not None:
        progress_bar.progress(100)
    if status_text is not None:
        status_text.text("Application ready")
    
    st.success("TRELLIS initialization completed")
    st.balloons()
    
    time.sleep(2)
    st.rerun()
