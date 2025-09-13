#!/usr/bin/env python3
"""Utility script to launch the ImageHorizonLibrary Image Debugger.

This script is meant for development and debugging purposes. It creates an
``ImageHorizonLibrary`` instance and opens the interactive Image Debugger GUI.
The reference folder can be provided via command line; otherwise a default
folder inside the repository is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open the ImageHorizonLibrary Image Debugger"
    )
    parser.add_argument(
        "--reference-folder",
        type=Path,
        default=Path("tests/utest/reference_images"),
        help="Folder containing reference images (default: tests/utest/reference_images)",
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Minimize the debugger window while taking a screenshot",
    )

    args = parser.parse_args()

    from ImageHorizonLibrary import ImageHorizonLibrary

    lib = ImageHorizonLibrary(reference_folder=str(args.reference_folder))
    lib.debug_image(
        minimize=args.minimize, dialog_default_dir=str(args.reference_folder)
    )


if __name__ == "__main__":
    main()
