"""Utility functions for platform detection and optional dependencies."""

import sys
from importlib import util
from platform import platform, architecture
from subprocess import call


PLATFORM = platform()
ARCHITECTURE = architecture()


def is_windows():
    return PLATFORM.lower().startswith('windows')


def is_mac():
    return PLATFORM.lower().startswith('darwin')


def is_linux():
    return PLATFORM.lower().startswith('linux')


def is_java():
    return PLATFORM.lower().startswith('java')

def has_retina():
    if is_mac():
        # Will return 0 if there is a retina display
        return call("system_profiler SPDisplaysDataType | grep 'Retina'", shell=True) == 0
    return False

def has_cv():
    try:
        import cv2  # noqa: F401
        return True
    except Exception:
        return False

def has_skimage():
    try:
        return "skimage" in sys.modules or util.find_spec("skimage") is not None
    except Exception:
        return False
