"""Simple GUI debugger for inspecting image matching results."""

from tkinter import *
from .image_debugger_controller import UILocatorController


class ImageDebugger:
    """Wrapper that launches the image debugger GUI."""

    def __init__(self, image_horizon_instance, minimize=True):
        """Create the debugger and start the main UI loop."""
        app = UILocatorController(image_horizon_instance, minimize=minimize)
        app.main()
