"""UI components for TRELLIS 3D Generator."""

import streamlit as st
import warnings
from PIL import Image, ImageDraw, ImageFont
import io

# Suppress warnings in UI components
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*deprecated.*")
warnings.filterwarnings("ignore", message=".*xFormers is available.*")
warnings.filterwarnings("ignore", message=".*torch.library.impl_abstract.*renamed.*")
warnings.filterwarnings("ignore", message=".*torch.library.register_fake.*")


def create_placeholder_image(width=512, height=512, text="No Image"):
    """Create a placeholder square image."""
    img = Image.new('RGB', (width, height), color='#E8E8E8')
    draw = ImageDraw.Draw(img)
    
    # Draw a border
    draw.rectangle([(10, 10), (width-10, height-10)], outline='#CCCCCC', width=3)
    
    # Draw an icon (simple camera shape)
    icon_size = width // 4
    icon_x = (width - icon_size) // 2
    icon_y = (height - icon_size) // 2 - 20
    
    # Camera body
    draw.rectangle(
        [(icon_x, icon_y + icon_size//4), (icon_x + icon_size, icon_y + icon_size)],
        fill='#CCCCCC'
    )
    # Camera lens
    lens_center_x = icon_x + icon_size // 2
    lens_center_y = icon_y + icon_size * 5 // 8
    lens_radius = icon_size // 4
    draw.ellipse(
        [(lens_center_x - lens_radius, lens_center_y - lens_radius),
         (lens_center_x + lens_radius, lens_center_y + lens_radius)],
        fill='#999999'
    )
    
    # Add text
    text_y = icon_y + icon_size + 20
    # Use default font
    draw.text((width//2, text_y), text, fill='#999999', anchor='mt')
    
    return img


def show_image_preview(image, title="Image", expanded=True, show_clear=False, clear_key=None):
    """Display an image preview with a nice UI."""
    
    # Create a container with title and clear button
    col_title, col_clear = st.columns([5, 1])
    with col_title:
        st.markdown(f"**{title}**")
    with col_clear:
        if show_clear and clear_key and image is not None:
            # Use a unique button key and return True when clicked
            if st.button("🗑️", key=f"clear_img_{clear_key}", help="Clear image", use_container_width=True):
                return "clear"
    
    # Show image or placeholder
    if image is not None:
        st.image(image, use_container_width=True)
    else:
        placeholder = create_placeholder_image()
        st.image(placeholder, use_container_width=True)
    
    return None


def create_placeholder_video():
    """Create a placeholder for video."""
    img = Image.new('RGB', (512, 256), color='#E8E8E8')
    draw = ImageDraw.Draw(img)
    
    # Draw a border
    draw.rectangle([(10, 10), (502, 246)], outline='#CCCCCC', width=3)
    
    # Draw play button icon
    center_x, center_y = 256, 128
    triangle = [
        (center_x - 30, center_y - 40),
        (center_x - 30, center_y + 40),
        (center_x + 40, center_y)
    ]
    draw.polygon(triangle, fill='#CCCCCC')
    
    # Add text
    draw.text((center_x, center_y + 70), "No Video", fill='#999999', anchor='mt')
    
    return img


def show_video_preview(video_path, show_clear=False, clear_key=None, show_progress=False, progress_text=None):
    """Display video preview with information."""
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("### 🎬 3D Model Preview")
    with col2:
        clear_clicked = False
        if show_clear and clear_key and video_path:
            if st.button("🗑️", key=f"clear_video_{clear_key}", help="Clear video", use_container_width=True):
                clear_clicked = True
    
    if video_path:
        # Use a unique container to avoid duplicate element IDs
        # The key is to ensure each video element has a unique context
        import hashlib
        
        # Generate a stable unique identifier based on clear_key
        video_hash = hashlib.md5(f"{clear_key}".encode()).hexdigest()[:8]
        
        # Create a unique key for this video in session state
        video_state_key = f"_video_render_{clear_key}"
        if video_state_key not in st.session_state:
            st.session_state[video_state_key] = 0
        
        # Use a fragment-like approach with unique markdown separators
        st.markdown(f'<div class="video-section" data-video-id="{video_hash}"></div>', unsafe_allow_html=True)
        
        # Read video as bytes to ensure unique content signature
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
        
        st.video(video_bytes, loop=True, autoplay=True)
        
        with st.expander("ℹ️ Video Info", expanded=False):
            st.info("This video shows color rendering (left) and normal map (right) of your 3D model rotating.")
    else:
        # Show placeholder with optional progress bar
        placeholder = create_placeholder_video()
        st.image(placeholder, use_container_width=True)
        
        if show_progress and progress_text:
            st.progress(0.5, text=progress_text)
    
    return "clear" if clear_clicked else None


def create_placeholder_glb():
    """Create a placeholder for GLB viewer."""
    img = Image.new('RGB', (512, 512), color='#E8E8E8')
    draw = ImageDraw.Draw(img)
    
    # Draw a border
    draw.rectangle([(10, 10), (502, 502)], outline='#CCCCCC', width=3)
    
    # Draw a 3D cube icon
    center_x, center_y = 256, 230
    size = 80
    
    # Back face
    draw.polygon([
        (center_x - size//2, center_y - size//4),
        (center_x + size//2, center_y - size//4),
        (center_x + size//2, center_y + size//2),
        (center_x - size//2, center_y + size//2)
    ], fill='#BBBBBB', outline='#999999', width=2)
    
    # Top face
    draw.polygon([
        (center_x - size//2, center_y - size//4),
        (center_x, center_y - size//2),
        (center_x + size, center_y - size//2),
        (center_x + size//2, center_y - size//4)
    ], fill='#CCCCCC', outline='#999999', width=2)
    
    # Right face
    draw.polygon([
        (center_x + size//2, center_y - size//4),
        (center_x + size, center_y - size//2),
        (center_x + size, center_y + size//4),
        (center_x + size//2, center_y + size//2)
    ], fill='#AAAAAA', outline='#999999', width=2)
    
    # Add text
    draw.text((center_x + size//4, center_y + size + 30), "No 3D Model", fill='#999999', anchor='mt')
    
    return img


def show_3d_model_viewer(glb_path, show_clear=False, clear_key=None, show_progress=False, progress_text=None):
    """Display interactive 3D model viewer."""
    import streamlit.components.v1 as components
    import base64
    
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("### 🎯 Interactive 3D Viewer")
    with col2:
        clear_clicked = False
        if show_clear and clear_key and glb_path:
            if st.button("🗑️", key=f"clear_glb_{clear_key}", help="Clear 3D model", use_container_width=True):
                clear_clicked = True
    
    if glb_path:
        try:
            with open(glb_path, 'rb') as f:
                glb_data = base64.b64encode(f.read()).decode()
            
            glb_html = f"""
            <div style="width: 100%; height: 500px; border-radius: 10px; overflow: hidden;">
                <model-viewer src="data:model/gltf-binary;base64,{glb_data}"
                             camera-controls 
                             auto-rotate
                             shadow-intensity="1"
                             style="width: 100%; height: 100%; background-color: #f0f0f0;">
                </model-viewer>
                <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
            </div>
            """
            components.html(glb_html, height=520)
            st.caption("🖱️ Click and drag to rotate • Scroll to zoom • Right-click and drag to pan")
            return "clear" if clear_clicked else None
        except Exception as e:
            st.warning("3D viewer not available. Download the GLB file to view in external 3D software.")
            return None
    else:
        # Show placeholder with optional progress bar
        placeholder = create_placeholder_glb()
        st.image(placeholder, use_container_width=True)
        
        if show_progress and progress_text:
            st.progress(0.5, text=progress_text)
        
        return None


def show_generation_progress(stage, current_step, total_steps):
    """Show generation progress."""
    progress = current_step / total_steps
    st.progress(progress, text=f"{stage}: Step {current_step}/{total_steps}")


def show_example_gallery(example_paths, columns=4, on_click_callback=None):
    """
    Display a gallery of example images as square thumbnails.
    
    Args:
        example_paths: List of paths to example images
        columns: Number of columns in the gallery
        on_click_callback: Function to call when an image is clicked (receives image path)
    
    Returns:
        Selected image path if clicked, None otherwise
    """
    import os
    
    cols = st.columns(columns)
    selected_path = None
    
    for i, img_path in enumerate(example_paths):
        with cols[i % columns]:
            try:
                img = Image.open(img_path)
                
                # Make square thumbnail
                size = min(img.size)
                left = (img.width - size) // 2
                top = (img.height - size) // 2
                img_square = img.crop((left, top, left + size, top + size))
                img_square.thumbnail((200, 200), Image.Resampling.LANCZOS)
                
                # Display image
                st.image(img_square, use_container_width=True)
                
                # Button to load this example
                filename = os.path.basename(img_path).replace('.png', '').replace('_', ' ').title()
                if st.button(f"Load", key=f"example_btn_{i}", use_container_width=True):
                    selected_path = img_path
                    
            except Exception as e:
                st.error(f"Error loading {img_path}: {e}")
    
    return selected_path

