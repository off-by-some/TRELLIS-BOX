"""
TRELLIS 3D Generator - Main Application Entry Point
Clean layered architecture with separation of concerns.
"""

import warnings
import os

# Application components
from webui.controllers import AppController
from webui.app_ui import TrellisApp

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*deprecated.*")
warnings.filterwarnings("ignore", message=".*xFormers is available.*")
warnings.filterwarnings("ignore", message=".*torch.library.impl_abstract.*renamed.*")
warnings.filterwarnings("ignore", message=".*torch.library.register_fake.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Environment setup
os.environ['SPCONV_ALGO'] = 'native'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,garbage_collection_threshold:0.8,roundup_power2_divisions:16'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

def main():
    """Main application entry point."""
    # Create the application controller
    controller = AppController()

    # Create and run the UI
    app = TrellisApp(controller)
    app.run()


if __name__ == "__main__":
    main()
