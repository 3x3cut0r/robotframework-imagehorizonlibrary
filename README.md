# ImageHorizonLibrary (Fork)

This repository is a fork of [Eficode's ImageHorizonLibrary](https://github.com/eficode/robotframework-imagehorizonlibrary).  
It exists to keep development moving.

Documentation has been adapted for GitHub Pages and is available at [https://3x3cut0r.github.io/robotframework-imagehorizonlibrary/](https://3x3cut0r.github.io/robotframework-imagehorizonlibrary/).

## Table of Contents

- [Notable differences to the original project](#notable-differences-to-the-original-project)
- [Introduction](#introduction)
- [Edge preprocessing](#edge-preprocessing)
- [Edge detection parameters](#edge-detection-parameters)
- [Keyword documentation](#keyword-documentation)
- [Robot keywords](#robot-keywords)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Windows](#windows)
  - [macOS](#macos)
  - [Linux](#linux)
- [Building from source](#building-from-source)
- [Running unit tests](#running-unit-tests)
- [Running acceptance tests](#running-acceptance-tests)
- [Updating docs](#updating-docs)

## Notable differences to the original project

- Merged all open RP on original repository.
- Compatibility with Python 3.11 and Robot Framework 7.
- Added `edge` recognition strategy using OpenCV (`cv2`) for robust matching on screens with pixel deviations.
- Additional edge preprocessing filters (`gaussian`, `median`, `erode`, `dilate`) for the `edge` recognition strategy.
- Optional multi-scale search that can be enabled with `Set Scale Range`.
- Keywords such as `Wait For`, `Locate`, and `Locate All` now have a working timout function and return the correlation score (`score`) and detected scaling factor (`scale`).
- Debugger now displays the best match score for located images.
- Edge detection result messages in the debugger have been simplified.
- `Take A Screenshot` keyword can now capture a specific `region` or `window`.
- Installation metadata managed via `pyproject.toml`, updated dependencies and removed depricated `setup.py|cfg` files.
- Removed unused dependencies and replaced `scikit-image` with headless OpenCV (`opencv-python-headless`).
- Various documentation improvements and safe version handling.

## Introduction

This Robot Framework library provides facilities to automate GUIs based on
image recognition similar to Sikuli, but without any Java dependency (100% Python).

There are two different recognition strategies: _default_ (using `pyautogui`)
and _edge_ (using OpenCV (`cv2`)). For non pixel perfect matches, there is a feature
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

The filters have the following effects:

- `gaussian` – applies an additional Gaussian blur to reduce noise.
- `median` – replaces each pixel with the median of the neighbourhood to
  remove salt-and-pepper noise while preserving edges.
- `erode` – shrinks bright regions which can help remove small artifacts.
- `dilate` – expands bright regions to close small gaps.

Example:

```
| Set Strategy | edge | edge_preprocess=gaussian | edge_kernel_size=5 |
```

## Edge detection parameters

The `edge` strategy exposes three parameters that control the Canny edge
detector:

- `edge_sigma` – width of the Gaussian blur applied before edge
  extraction. Larger values smooth the image more aggressively which can
  reduce noise but may remove fine details.
- `edge_low_threshold` – lower gradient magnitude threshold (0.0–1.0).
  Edges with a value below this threshold are discarded.
- `edge_high_threshold` – upper gradient magnitude threshold (0.0–1.0).
  Values above this are considered strong edges. Edges between the two
  thresholds are kept only if connected to a strong edge.

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

Keywords such as `Wait For`, `Locate`, and `Locate All` return not only the match coordinates
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

## Robot keywords

| Keyword                     | Description                                                                                                                                                                                |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Click                       | Click once with the specified mouse button.                                                                                                                                                |
| Click Image                 | Locate an image on screen and click its center once.                                                                                                                                       |
| Click To The Above Of       | Click above a location by a given pixel offset.                                                                                                                                            |
| Click To The Above Of Image | Click above a located reference image by a pixel offset.                                                                                                                                   |
| Click To The Below Of       | Click below a location by a given pixel offset.                                                                                                                                            |
| Click To The Below Of Image | Click below a located reference image by a pixel offset.                                                                                                                                   |
| Click To The Left Of        | Click left of a location by a given pixel offset.                                                                                                                                          |
| Click To The Left Of Image  | Click left of a located reference image by a pixel offset.                                                                                                                                 |
| Click To The Right Of       | Click right of a location by a given pixel offset.                                                                                                                                         |
| Click To The Right Of Image | Click right of a located reference image by a pixel offset.                                                                                                                                |
| Copy                        | Copy currently selected text to the system clipboard.                                                                                                                                      |
| Copy From The Above Of      | Copy text above a reference image.                                                                                                                                                         |
| Copy From The Below Of      | Copy text below a reference image.                                                                                                                                                         |
| Copy From The Left Of       | Copy text left of a reference image.                                                                                                                                                       |
| Copy From The Right Of      | Copy text right of a reference image.                                                                                                                                                      |
| Debug Image                 | Halts the test execution and opens the image debugger UI; accepts an optional reference folder and a `minimize` flag to hide the debugger window during screenshots (defaults to visible). |
| Does Exist                  | Check whether a reference image exists on the screen.                                                                                                                                      |
| Double Click                | Double-click with the specified mouse button.                                                                                                                                              |
| Get Clipboard Content       | Return the current text from the system clipboard.                                                                                                                                         |
| Launch Application          | Launch an external application as a new process.                                                                                                                                           |
| Locate                      | Locate image on screen.                                                                                                                                                                    |
| Locate All                  | Locate all occurrences of an image on screen.                                                                                                                                              |
| Mouse Down                  | Press and hold a mouse button.                                                                                                                                                             |
| Mouse Up                    | Release a previously pressed mouse button.                                                                                                                                                 |
| Move To                     | Move the mouse pointer to absolute screen coordinates.                                                                                                                                     |
| Pause                       | Display a modal dialog to temporarily halt test execution.                                                                                                                                 |
| Press Combination           | Press multiple keyboard keys simultaneously.                                                                                                                                               |
| Reset Confidence            | Resets the confidence level to the library default.                                                                                                                                        |
| Reset Scale Range           | Disables multi-scale search and resets to defaults.                                                                                                                                        |
| Set Confidence              | Sets the accuracy when finding images.                                                                                                                                                     |
| Set Keyword On Failure      | Sets keyword to be run, when location-related keywords fail.                                                                                                                               |
| Set Reference Folder        | Sets where all reference images are stored.                                                                                                                                                |
| Set Scale Range             | Enables searching images across a range of scales.                                                                                                                                         |
| Set Screenshot Folder       | Sets the folder where screenshots are saved to.                                                                                                                                            |
| Set Strategy                | Changes the way how images are detected on the screen.                                                                                                                                     |
| Take A Screenshot           | Capture and save a screenshot of the current screen, a region or a specific window.                                                                                                        |
| Terminate Application       | Terminate a process started with :py:meth:launch_application.                                                                                                                              |
| Triple Click                | Triple-click with the specified mouse button.                                                                                                                                              |
| Type                        | Type a sequence of text fragments and/or special keys.                                                                                                                                     |
| Type With Keys Down         | Hold down keyboard keys while typing text.                                                                                                                                                 |
| Wait For                    | Wait until an image appears on the screen.                                                                                                                                                 |

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
install the library with the `debug` extra:

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
