"""
src/__init__.py
================
Makes the src/ directory a proper Python package.
இந்த file src/ folder-ஐ Python package-ஆக மாற்றும்.
"""

# Expose key functions for convenient import from src package
# Example: from src import predict_image

from src.main import predict_image
from src.utils import print_banner, check_dataset, Timer

__all__ = [
    "predict_image",
    "print_banner",
    "check_dataset",
    "Timer",
]
