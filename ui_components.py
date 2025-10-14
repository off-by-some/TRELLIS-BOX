"""UI components for TRELLIS 3D Generator."""

import streamlit as st


def show_image_preview(image, title="Image", expanded=True):
    """Display an image preview with a nice UI."""
    with st.expander(title, expanded=expanded):
        st.image(image, use_container_width=True)


def show_video_preview(video_path):
    """Display video preview with information."""
    st.markdown("### 🎬 3D Model Preview")
    st.video(video_path, loop=True)
    
    with st.expander("ℹ️ Video Info", expanded=False):
        st.info("This video shows color rendering (left) and normal map (right) of your 3D model rotating.")


def show_3d_model_viewer(glb_path):
    """Display interactive 3D model viewer."""
    import streamlit.components.v1 as components
    import base64
    
    st.markdown("### 🎯 Interactive 3D Viewer")
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
        return True
    except Exception as e:
        st.warning("3D viewer not available. Download the GLB file to view in external 3D software.")
        return False


def show_generation_progress(stage, current_step, total_steps):
    """Show generation progress."""
    progress = current_step / total_steps
    st.progress(progress, text=f"{stage}: Step {current_step}/{total_steps}")

