"""
Image Preview Component - A reactive component that manages its own lifecycle
and updates when props change.

This component follows a React-like pattern where it maintains internal state
and reactively updates when props change.
"""

import streamlit as st
from PIL import Image
import hashlib
from typing import Optional, Callable, Union
from io import BytesIO


def _get_image_hash(image: Optional[Image.Image]) -> str:
    """Generate a hash for an image to detect changes."""
    if image is None:
        return "none"
    
    # Convert image to bytes and hash it
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return hashlib.md5(img_byte_arr).hexdigest()


def _create_placeholder_image(width: int = 400, height: int = 400) -> Image.Image:
    """Create a placeholder image when no image is provided."""
    from PIL import ImageDraw, ImageFont
    
    img = Image.new('RGB', (width, height), color=(240, 242, 246))
    draw = ImageDraw.Draw(img)
    
    # Draw border
    border_color = (200, 204, 214)
    draw.rectangle([10, 10, width-10, height-10], outline=border_color, width=3)
    
    # Draw icon (camera emoji placeholder)
    icon_size = 80
    icon_y = height // 2 - icon_size
    try:
        # Try to load a font, fall back to default if not available
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", icon_size)
    except:
        font = ImageFont.load_default()
    
    # Draw text
    text = "📷"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = (width - text_width) // 2
    text_y = icon_y
    
    draw.text((text_x, text_y), text, fill=(102, 126, 234), font=font)
    
    # Draw subtitle
    subtitle = "No image"
    subtitle_y = text_y + text_height + 20
    try:
        subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        subtitle_font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    subtitle_width = bbox[2] - bbox[0]
    subtitle_x = (width - subtitle_width) // 2
    
    draw.text((subtitle_x, subtitle_y), subtitle, fill=(118, 75, 162), font=subtitle_font)
    
    return img


class ImagePreview:
    """
    A reactive image preview component that manages its own lifecycle.
    
    This component:
    - Detects when the image prop changes and updates accordingly
    - Manages its own internal state
    - Provides callbacks for user interactions (clear, click, etc.)
    - Shows placeholder when no image is provided
    - Handles clear functionality with a trash button
    
    Usage:
        preview = ImagePreview(
            component_id="my_image",
            title="📷 My Image",
            show_clear=True
        )
        
        cleared = preview.render(image=my_image)
        if cleared:
            # Handle clear action
            my_image = None
    """
    
    def __init__(
        self,
        component_id: str,
        title: str = "Image Preview",
        show_clear: bool = False,
        show_info: bool = False,
        placeholder_text: str = "No image",
        on_clear: Optional[Callable] = None,
        use_container_width: bool = True,
        expanded: bool = True
    ):
        """
        Initialize the ImagePreview component.
        
        Args:
            component_id: Unique identifier for this component instance
            title: Title to display above the image
            show_clear: Whether to show the clear button
            show_info: Whether to show image information expander
            placeholder_text: Text to show in placeholder
            on_clear: Optional callback function when image is cleared
            use_container_width: Whether image should use full container width
            expanded: Whether the component should be expanded by default
        """
        self.component_id = component_id
        self.title = title
        self.show_clear = show_clear
        self.show_info = show_info
        self.placeholder_text = placeholder_text
        self.on_clear = on_clear
        self.use_container_width = use_container_width
        self.expanded = expanded
        
        # Initialize component state in session state if not exists
        state_key = f"_image_preview_{component_id}"
        if state_key not in st.session_state:
            st.session_state[state_key] = {
                "current_image_hash": None,
                "render_count": 0,
                "last_clear_time": None
            }
        
        self.state = st.session_state[state_key]
    
    def _detect_image_change(self, image: Optional[Image.Image]) -> bool:
        """Detect if the image prop has changed."""
        new_hash = _get_image_hash(image)
        changed = new_hash != self.state["current_image_hash"]
        
        if changed:
            self.state["current_image_hash"] = new_hash
            self.state["render_count"] += 1
        
        return changed
    
    def render(self, image: Optional[Image.Image]) -> bool:
        """
        Render the image preview component.
        
        Args:
            image: The image to display (None for placeholder)
        
        Returns:
            bool: True if the clear button was clicked, False otherwise
        """
        # Detect if image changed
        image_changed = self._detect_image_change(image)
        
        # Create header with title and clear button
        col_title, col_clear = st.columns([5, 1])
        
        with col_title:
            st.markdown(f"**{self.title}**")
        
        cleared = False
        with col_clear:
            if self.show_clear and image is not None:
                if st.button(
                    "🗑️",
                    key=f"{self.component_id}_clear_{self.state['render_count']}",
                    help="Clear image",
                    use_container_width=True
                ):
                    cleared = True
                    self.state["last_clear_time"] = st.session_state.get("_last_interaction_time", 0)
                    
                    # Call the on_clear callback if provided
                    if self.on_clear:
                        self.on_clear()
        
        # Display image or placeholder
        if image is not None:
            st.image(image, use_container_width=self.use_container_width)
            
            # Show image info if requested
            if self.show_info:
                with st.expander("ℹ️ Image Info", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Width", f"{image.width}px")
                        st.metric("Format", image.format or "Unknown")
                    with col2:
                        st.metric("Height", f"{image.height}px")
                        st.metric("Mode", image.mode)
        else:
            # Show placeholder
            placeholder = _create_placeholder_image()
            st.image(placeholder, use_container_width=self.use_container_width)
        
        return cleared
    
    def get_state(self) -> dict:
        """Get the current internal state of the component."""
        return self.state.copy()
    
    def reset(self):
        """Reset the component state."""
        self.state["current_image_hash"] = None
        self.state["render_count"] = 0
        self.state["last_clear_time"] = None


# Convenience function for simple use cases
def image_preview(
    image: Optional[Image.Image],
    component_id: str,
    title: str = "Image Preview",
    show_clear: bool = False,
    show_info: bool = False,
    on_clear: Optional[Callable] = None
) -> bool:
    """
    Functional wrapper for ImagePreview component.
    
    This is a simpler way to use the component for basic cases.
    
    Args:
        image: The image to display
        component_id: Unique identifier for this component
        title: Title to display
        show_clear: Whether to show clear button
        show_info: Whether to show image info
        on_clear: Callback when cleared
    
    Returns:
        bool: True if cleared
    
    Example:
        if image_preview(my_image, "preview1", show_clear=True):
            my_image = None
            st.rerun()
    """
    preview = ImagePreview(
        component_id=component_id,
        title=title,
        show_clear=show_clear,
        show_info=show_info,
        on_clear=on_clear
    )
    
    return preview.render(image)

