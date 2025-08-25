# ImageHorizonLibrary (Fork)

This repository is a fork of [Eficode's ImageHorizonLibrary](https://github.com/eficode/robotframework-imagehorizonlibrary).  
It exists to keep development moving.

Documentation has been adapted for GitHub Pages and is available at [https://3x3cut0r.github.io/robotframework-imagehorizonlibrary/](https://3x3cut0r.github.io/robotframework-imagehorizonlibrary/).


Notable differences to the original project:

- Merged all open RP on original repository.
- Added `edge` recognition strategy using `skimage` for robust matching on screens with pixel deviations.
- Additional edge preprocessing filters (`gaussian`, `median`, `erode`, `dilate`) for the `edge` recognition strategy.
- Optional multi-scale search that can be enabled with `Set Scale Range`.
- Keywords such as `Wait For` and `Locate` now have a working timout function.
- Keywords such as `Wait For` and `Locate` now return the correlation score (`score`) and detected scaling factor (`scale`).
- Compatibility with Python 3.11 and Robot Framework 7.
- Installation metadata managed via `pyproject.toml` and updated dependencies.
- Various documentation improvements and safe version handling.

## Introduction

This Robot Framework library provides facilities to automate GUIs based on
image recognition similar to Sikuli, but without any Java dependency (100% Python).

There are two different recognition strategies: _default_ (using `pyautogui`)
and _edge_ (using `skimage`). For non pixel perfect matches, there is a feature
called "confidence level" that allows to define the percentage of pixels which
_must_ match. In the _default_ strategy, confidence comes with a dependency to
OpenCV (`opencv-python`). This functionality is optional - you are not required
to install `opencv-python` if you do not use confidence level.

By default ImageHorizonLibrary uses the `default` strategy. The `edge`
strategy can be enabled when dealing with screens that contain unpredictable
pixel deviations.

## Edge preprocessing

When using the `edge` recognition strategy you can optionally apply a
pre-processing filter before edge detection. This requires OpenCV (`cv2`)
and is enabled with the `edge_preprocess` argument. Supported filters are
`gaussian`, `median`, `erode` and `dilate`. The size of the filter kernel
can be adjusted with `edge_kernel_size` (default is 3).

Example:

```
| Set Strategy | edge | edge_preprocess=gaussian | edge_kernel_size=5 |
```

Additional usage examples for the `edge` strategy:

```
| # use with auto-detected parameters
| Set Strategy | edge |
| Click Image | button.png |
| # provide explicit thresholds and confidence
| Set Strategy | edge | edge_sigma=2.0 | edge_low_threshold=0.1 | edge_high_threshold=0.3 | confidence=0.8 |
| # enable edge preprocessing with a custom kernel
| Set Strategy | edge | edge_preprocess=median | edge_kernel_size=7 |
```

Keywords such as `Wait For` and `Locate` return not only the match coordinates
but also the correlation score and an optional detected scaling factor
describing the match quality. The typical return signature is:

```
${x} ${y} ${score} ${scale}= | Wait For | example.png | timeout=5 |
```

The `score` ranges from `-1` (no correlation) to `1` (perfect match) and the
`scale` value is `1.0` when no resizing is detected.

ImageHorizonLibrary searches only at scale `1.0` by default. To enable
multi-scale matching (for example `0.8`–`1.2`), use:

```
| Set Scale Range | 0.8 | 1.2 | 9 |
```

Call `Reset Scale Range` to disable multi-scale search again.

## Keyword documentation

Generated keyword documentation is available in
[`docs/index.html`](docs/index.html).

## Prerequisites

- Python 3.x
- pip (via `python3 -m pip`) for easy installation
- Robot Framework

On Ubuntu, you need to take [special measures](https://pyautogui.readthedocs.org/en/latest/screenshot.html#special-notes-about-ubuntu)
to make the screenshot functionality work correctly. The keyboard functions
might not work on Ubuntu when run in VirtualBox on Windows.

## Installation

If you have pip, installation is straightforward:

```
python3 -m pip install robotframework-imagehorizonlibrary
```

This will automatically install dependencies as well as their dependencies.

If you want to use the optional Image Debugger that relies on matplotlib,
install the library with the ``debug`` extra:

```
python3 -m pip install "robotframework-imagehorizonlibrary[debug]"
```

### Windows

ImageHorizonLibrary should work on Windows "out-of-the-box". Just run the
commands above to install it.

### macOS

_NOTICE_ ImageHorizonLibrary does not currently work with XCode v.8. Please use a previous version.
You additionally need to install these for `pyautogui`:

```
python3 -m pip install pyobjc-core pyobjc
```

For these, you need to install [XCode](https://developer.apple.com/xcode/downloads/).

### Linux

You additionally need to install these for `pyautogui`:

```
sudo apt-get install python3-dev python3-xlib
```

You might also need, depending on your Python distribution, to install:

```
sudo apt-get install python3-tk
```

If you are using virtualenv, you must install [python-xlib](http://sourceforge.net/projects/python-xlib/files/)
manually to the virtual environment for `pyautogui`:

- Fetch the source distribution
- Install with:
  ```
  python3 -m pip install python-xlib-<latest version>.tar.gz
  ```

## Building from source

To create a wheel from the current sources without installing runtime
dependencies, first ensure the build requirements are available:

```
python3 -m pip install --upgrade pip setuptools wheel packaging
```

Then build the wheel:

```
pip wheel . --no-deps
```

## Running unit tests

```
python3 tests/utest/run_tests.py [verbosity=2]
```

## Running acceptance tests

Additionally to unit test dependencies, you also need OpenCV, Eel, `scrot` and Chrome/Chromium browser.
OpenCV is used because these tests are testing also confidence level.
Browser is used by Eel for cross-platform GUI demo application.
`scrot` is used for capturing screenshots.

```
python3 -m pip install opencv-python eel
python3 tests/atest/run_tests.py
```

## Updating docs

To regenerate documentation (`docs/ImageHorizonLibrary.html`), use this command:

```
python3 -m robot.libdoc -P ./src ImageHorizonLibrary docs/ImageHorizonLibrary.html
cp docs/ImageHorizonLibrary.html docs/index.html
```

The documentation is published on GitHub Pages at [https://3x3cut0r.github.io/robotframework-imagehorizonlibrary/](https://3x3cut0r.github.io/robotframework-imagehorizonlibrary/).
