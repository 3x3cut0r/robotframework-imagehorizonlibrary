# -*- coding: utf-8 -*-
from collections import OrderedDict
from contextlib import contextmanager
import inspect

from .errors import *  # import errors before checking dependencies!

try:
    import pyautogui as ag
except ImportError:
    raise ImageHorizonLibraryError(
        "There is something wrong with pyautogui or " "it is not installed."
    )

try:
    from robot.api import logger as LOGGER
    from robot.libraries.BuiltIn import BuiltIn
except ImportError:
    raise ImageHorizonLibraryError(
        "There is something wrong with " "Robot Framework or it is not installed."
    )

try:
    from tkinter import Tk as TK
except ImportError:
    raise ImageHorizonLibraryError(
        "There is either something wrong with "
        "Tkinter or you are running this on Java, "
        "which is not a supported platform. Please "
        "use Python and verify that Tkinter works."
    )

from . import utils
from .interaction import *
from .recognition import *
from .version import VERSION

__version__ = VERSION


class ImageHorizonLibrary(
    _Keyboard, _Mouse, _OperatingSystem, _RecognizeImages, _Screenshot
):
    """A cross-platform Robot Framework library for GUI automation.

    *Key features*:
    - Automates *keyboard and mouse actions* on the screen (based on [https://pyautogui.readthedocs.org|pyautogui]).
    - The regions to execute these actions on (buttons, sliders, input fields etc.) are determined by `reference images` which the library detects on the screen - independently of the OS or the application type.
    - Two different image `recognition strategies`: `default` (fast and reliable of predictable screen content), and `edge` (to facilitate the recognition of unpredictable pixel deviations)
    - The library can also take screenshots in case of failure or by intention.
    - Search keywords return coordinates together with match score and optional detected scale.

    = Image Recognition =

    == Reference images ==
    ``reference_image`` parameter can be either a single file or a folder.
    If ``reference_image`` is a folder, image recognition is tried separately
    for each image in that folder, in alphabetical order until a match is found.

    For ease of use, reference image names are automatically normalized
    according to the following rules:

    - The name is lower cased: ``MYPICTURE`` and ``mYPiCtUrE`` become
      ``mypicture``

    - All spaces are converted to underscore ``_``: ``my picture`` becomes
      ``my_picture``

    - If the image name does not end in ``.png``, it will be added:
      ``mypicture`` becomes ``mypicture.png``

    - Path to _reference folder_ is prepended. This option must be given when
      `importing` the library.

    Using good names for reference images is evident from easy-to-read test
    data:

    | `Import Library` | ImageHorizonLibrary                   | reference_folder=images |                                                            |
    | `Click Image`    | popup Window title                    |                         | # Path is images/popup_window_title.png                    |
    | `Click Image`    | button Login Without User Credentials |                         | # Path is images/button_login_without_user_credentials.png |

    == Recognition strategies ==
    Basically, image recognition works by searching a reference image on the
    another image (a screnshot of the current desktop).
    If there is a region with 100% matching pixels of the reference image, this
    area represents a match.

    By default, the reference image must be an exakt sub-image of the screenshot.
    This works flawlessly in most cases.

    But problems can arise when:

    - the application's GUI uses transpareny effects
    - the screen resolution/the window size changes
    - font aliasing is used for dynamic text
    - compression algorithms in RDP/Citrix cause invisible artifacts
    - ...and in many more situations.

    In those situations, a certain amount of the pixels do not match.

    To solve this, ImageHorizon comes with a parameter ``confidence level``. This is a decimal value
    between 0 and 1 (inclusive) and defines how many percent of the reference image pixels
    must match the found region's imag. It is set to 1.0 = 100% by default.

    Confidence level can be set during `library importing` and re-adjusted during the test
    case with the keyword `Set Confidence`.

    === Default image detection strategy ===

    If imported without any strategy argument, the library uses [https://pyautogui.readthedocs.org|pyautogui]
    under the hood to recognize images on the screen.
    This is the perfect choice to start writing tests.

    To use `confidence level in mode` ``default`` the
    [https://pypi.org/project/opencv-python|opencv-python] Python package
    must be installed separately:

    | $ python3 -m pip install opencv-python

    After installation, the library will automatically use OpenCV for confidence
    levels lower than 1.0.

    === The "edge" image detection strategy ===

    The default image recognition reaches its limitations when the area to
    match contains a *disproportionate amount of unpredictable pixels*.

    The idea for this strategy came from a problem in real life: a web application
    showing a topographical map (loaded from a 3rd party provider), with a layer of
    interstate highways as black lines. For some reasons, the pixels of topographic
    areas between the highway lines (which are the vast majority) showed a slight
    deviation in brightness - invisible for the naked eye, but enough to make the test failing.

    The abstract and simplified example for this is a horizontal black line of 1px width in a
    matrix of 10x10 white pixels. To neglect a (slight) brightness deviation of the white pixels,
    you would need a confidence level of 0.1 which allows 90% of the pixels to be
    different. This is insanse and leads to inpredictable results.

    That's why ``edge`` was implemented as an alternative recognition strategy.
    The key here lies in the approach to *reduce both images* (reference and screenshot
    image) *to the essential characteristics* and then *compare _those_ images*.

    "Essential characteristics" of an image are those areas where neighbouring pixels show a
    sharp change of brightness, better known as "edges". [https://en.wikipedia.org/wiki/Edge_detection|Edge detection]
    is the process of finding the edges in an image, done by [https://opencv.org/|OpenCV] in this library.

    As a brief digression, edge detection is a multi-step process:

    - apply a [https://en.wikipedia.org/wiki/Gaussian_filter|Gaussian filter] (blurs the image to remove noise; intensity set by parameter `sigma`)
    - apply a [https://en.wikipedia.org/wiki/Sobel_operator|Sobel filter] (remove non-max pixels, get a 1 pixel edge curve)
    - separate weak edges from strong ones with [https://en.wikipedia.org/wiki/Canny_edge_detector#Edge_tracking_by_hysteresis|hysteresis]
    - apply the `template_matching` routine to get a [https://en.wikipedia.org/wiki/Cross-correlation|cross correlation] matrix of values from -1 (no correlation) to +1 (perfect correlation).
    - Filter out only those coordinates with values greater than the ``confidence`` level, take the max

    The keyword `Debug Image` opens a debugger UI where confidence level, Gaussian sigma and low/high thresholds can be tested and adjusted to individual needs. It also accepts optional ``reference_folder`` and ``minimize`` arguments to inspect images stored in a different location or hide the debugger window during screenshots (visible by default).

    Edge detection costs some extra CPU time; you should always first try
    to use the ``default`` strategy and only selectively switch to ``edge``
    when a confidence level below 0.9 is not sufficient to detect images reliably anymore:

    | # use with defaults (parameters auto-detected per image)
    | Set Strategy  edge
    | # use with custom parameters
    | Set Strategy  edge  edge_sigma=2.0  edge_low_threshold=0.1  edge_high_threshold=0.3  confidence=0.8
    | # enable edge preprocessing with a custom kernel
    | Set Strategy  edge  edge_preprocess=gaussian  edge_kernel_size=5

    Keywords like ``Wait For`` return ``(x, y, score, scale)`` where
    ``score`` is the match quality between -1 and 1 and
    ``scale`` is the detected resize factor (1.0 means no scaling).

    By default only a scale of ``1.0`` is considered during image search.
    To search over a range of scales (e.g. to support multiple screen
    resolutions), enable multi-scale search with ``Set Scale Range`` and
    disable it with ``Reset Scale Range``.


    = Performance =

    Locating images on screen, especially if screen resolution is large and
    reference image is also large, might take considerable time, regardless
    of the strategy.
    It is therefore advisable to save the returned coordinates if you are
    manipulating the same context many times in the row:

    | `Wait For`                   | label Name |     |
    | `Click To The Left Of Image` | label Name | 200 |

    In the above example, same image is located twice. Below is an example how
    we can leverage the returned location:

    | ${location}=           | `Wait For`  | label Name |
    | `Click To The Left Of` | ${location} | 200        |
    """

    ROBOT_LIBRARY_SCOPE = "TEST SUITE"
    ROBOT_LIBRARY_VERSION = VERSION

    def __init__(
        self,
        reference_folder=None,
        screenshot_folder=None,
        keyword_on_failure="ImageHorizonLibrary.Take A Screenshot",
        confidence=None,
        strategy="default",
        edge_sigma=None,
        edge_low_threshold=None,
        edge_high_threshold=None,
        edge_preprocess=None,
        edge_kernel_size=3,
        validate_match=False,
        validation_margin=5,
    ):
        """ImageHorizonLibrary can be imported with several options.

        ``reference_folder`` is path to the folder where all reference images
        are stored. It must be a _valid absolute path_. As the library
        is suite-specific (ie. new instance is created for every suite),
        different suites can have different folders for it's reference images.

        ``screenshot_folder`` is path to the folder where screenshots are
        saved. If not given, screenshots are saved to the current working
        directory.

        ``keyword_on_failure`` is the keyword to be run, when location-related
        keywords fail. If you wish to not take screenshots, use for example
        `BuiltIn.No Operation`. Keyword must however be a valid keyword.

        ``strategy`` sets the way how images are detected on the screen. See also
        keyword `Set Strategy` to change the strategy during the test. Parameters:
        - ``default`` - (Default)
        - ``edge`` - Advanced image recognition options with canny edge detection

        The ``edge`` strategy allows these additional parameters:
          - ``edge_sigma`` – standard deviation of the Gaussian blur applied
            before edge extraction; larger values smooth more noise (auto if
            ``None``).
          - ``edge_low_threshold`` – lower gradient magnitude threshold
            (0.0–1.0); gradients below this value are discarded (auto if
            ``None``).
          - ``edge_high_threshold`` – upper gradient magnitude threshold
            (0.0–1.0) used to mark strong edges; values between the thresholds
            are kept only if connected to a strong edge (auto if ``None``).
          - ``edge_preprocess`` – optional pre-processing filter; allowed
            filters ``gaussian`` (extra blur), ``median`` (noise reduction),
            ``erode`` (shrink bright regions), ``dilate`` (expand bright
            regions). Requires OpenCV ``cv2``.
          - ``edge_kernel_size`` – kernel size for the pre-processing filter
          - ``validate_match`` - re-check match on original image with OpenCV
          - ``validation_margin`` - margin in pixels around match for validation
        """

        # _RecognizeImages.set_strategy(self, strategy)
        self.reference_folder = reference_folder
        self.screenshot_folder = screenshot_folder
        self.keyword_on_failure = keyword_on_failure
        self.open_applications = OrderedDict()
        self.screenshot_counter = 1
        self.is_windows = utils.is_windows()
        self.is_mac = utils.is_mac()
        self.is_linux = utils.is_linux()
        self.has_retina = utils.has_retina()
        self.has_cv = utils.has_cv()
        self.confidence = confidence
        self.initial_confidence = confidence
        self._class_bases = inspect.getmro(self.__class__)
        self.set_strategy(strategy, self.confidence)
        try:
            self.edge_sigma = float(edge_sigma) if edge_sigma is not None else None
        except (TypeError, ValueError):
            self.edge_sigma = None
        try:
            self.edge_low_threshold = (
                float(edge_low_threshold) if edge_low_threshold is not None else None
            )
        except (TypeError, ValueError):
            self.edge_low_threshold = None
        try:
            self.edge_high_threshold = (
                float(edge_high_threshold) if edge_high_threshold is not None else None
            )
        except (TypeError, ValueError):
            self.edge_high_threshold = None
        self.edge_preprocess = edge_preprocess
        try:
            self.edge_kernel_size = int(edge_kernel_size)
        except (TypeError, ValueError):
            self.edge_kernel_size = 3
        self.validate_match = validate_match
        try:
            self.validation_margin = int(validation_margin)
        except (TypeError, ValueError):
            self.validation_margin = 5

        # multi-scale search configuration
        self.scale_enabled = False
        self.scale_min = 0.8
        self.scale_max = 1.2
        self.scale_steps = 9

    def set_strategy(
        self,
        strategy,
        edge_sigma=None,
        edge_low_threshold=None,
        edge_high_threshold=None,
        confidence=None,
        edge_preprocess=None,
        edge_kernel_size=3,
        validate_match=False,
        validation_margin=5,
    ):
        """Changes the way how images are detected on the screen. This can also be done globally during `Importing`.
        Strategies:
        - ``default``
        - ``edge`` - Advanced image recognition options with canny edge detection

        The ``edge`` strategy allows these additional parameters:
          - ``edge_sigma`` - Gaussian blur intensity (auto if ``None``)
          - ``edge_low_threshold`` - low pixel gradient threshold (auto if ``None``)
          - ``edge_high_threshold`` - high pixel gradient threshold (auto if ``None``)
          - ``edge_preprocess`` - optional pre-processing filter; allowed filters:
            ``gaussian``, ``median``, ``erode``, ``dilate`` (requires OpenCV ``cv2``)
          - ``edge_kernel_size`` - kernel size for the pre-processing filter
          - ``validate_match`` - re-check match on original image with OpenCV
          - ``validation_margin`` - margin in pixels around match for validation

        Both strategies can optionally be initialized with a new confidence."""

        self.strategy = strategy
        if strategy == "default":
            self.strategy_instance = _StrategyPyautogui(self)
        elif strategy == "edge":
            self.strategy_instance = _StrategyCv2(self)
            try:
                self.edge_sigma = float(edge_sigma) if edge_sigma is not None else None
            except (TypeError, ValueError):
                self.edge_sigma = None
            try:
                self.edge_low_threshold = (
                    float(edge_low_threshold) if edge_low_threshold is not None else None
                )
            except (TypeError, ValueError):
                self.edge_low_threshold = None
            try:
                self.edge_high_threshold = (
                    float(edge_high_threshold) if edge_high_threshold is not None else None
                )
            except (TypeError, ValueError):
                self.edge_high_threshold = None
            self.edge_preprocess = edge_preprocess
            try:
                self.edge_kernel_size = int(edge_kernel_size)
            except (TypeError, ValueError):
                self.edge_kernel_size = 3
            self.validate_match = validate_match
            try:
                self.validation_margin = int(validation_margin)
            except (TypeError, ValueError):
                self.validation_margin = 5
        else:
            raise StrategyException('Invalid strategy: "%s"' % strategy)

        if not confidence is None:
            self.set_confidence(confidence)

        # Linking protected _try_locate to the strategy's method
        self._try_locate = self.strategy_instance._try_locate

    def _get_location(self, direction, location, offset):
        x, y = location
        offset = int(offset)
        if direction == "left":
            x = x - offset
        if direction == "up":
            y = y - offset
        if direction == "right":
            x = x + offset
        if direction == "down":
            y = y + offset
        return x, y

    def _click_to_the_direction_of(
        self, direction, location, offset, clicks, button, interval
    ):
        x, y = self._get_location(direction, location, offset)
        try:
            clicks = int(clicks)
        except ValueError:
            raise MouseException('Invalid argument "%s" for `clicks`')
        if button not in ["left", "middle", "right"]:
            raise MouseException('Invalid button "%s" for `button`')
        try:
            interval = float(interval)
        except ValueError:
            raise MouseException('Invalid argument "%s" for `interval`')

        LOGGER.info(
            "Clicking %d time(s) at (%d, %d) with "
            "%s mouse button at interval %f" % (clicks, x, y, button, interval)
        )
        ag.click(x, y, clicks=clicks, button=button, interval=interval)

    def _convert_to_valid_special_key(self, key):
        key = str(key).lower()
        if key.startswith("key."):
            key = key.split("key.", 1)[1]
        elif len(key) > 1:
            return None
        if key in ag.KEYBOARD_KEYS:
            return key
        return None

    def _validate_keys(self, keys):
        valid_keys = []
        for key in keys:
            valid_key = self._convert_to_valid_special_key(key)
            if not valid_key:
                raise KeyboardException(
                    'Invalid keyboard key "%s", valid '
                    "keyboard keys are:\n%r" % (key, ", ".join(ag.KEYBOARD_KEYS))
                )
            valid_keys.append(valid_key)
        return valid_keys

    def _press(self, *keys, **options):
        keys = self._validate_keys(keys)
        ag.hotkey(*keys, **options)

    @contextmanager
    def _tk(self):
        tk = TK()
        yield tk.clipboard_get()
        tk.destroy()

    def copy(self):
        """Copy currently selected text to the system clipboard.

        The platform specific shortcut is used (``Ctrl+C`` on Windows/Linux,
        ``⌘+C`` on macOS) and the clipboard content is returned.

        Returns
        -------
        str
            The text retrieved from the clipboard after the copy operation.
        """
        key = "Key.command" if self.is_mac else "Key.ctrl"
        self._press(key, "c")
        return self.get_clipboard_content()

    def get_clipboard_content(self):
        """Return the current text from the system clipboard.

        Returns
        -------
        str
            Contents of the clipboard.
        """
        with self._tk() as clipboard_content:
            return clipboard_content

    def pause(self):
        """Display a modal dialog to temporarily halt test execution.

        The dialog must be dismissed manually by clicking the Continue button.
        This is primarily intended for debugging or developing test cases.
        """
        ag.alert(text="Test execution paused.", title="Pause", button="Continue")

    def _run_on_failure(self):
        if not self.keyword_on_failure:
            return
        try:
            BuiltIn().run_keyword(self.keyword_on_failure)
        except Exception as e:
            LOGGER.debug(e)
            LOGGER.warn("Failed to take a screenshot. " "Is Robot Framework running?")

    def set_keyword_on_failure(self, keyword_on_failure):
        """Sets keyword to be run, when location-related
        keywords fail.
        This keyword might be used to temporarily diable screenshots, then re-enable them later in the test.
        See `library importing` for he usage of keyword_on_failure.
        """
        self.keyword_on_failure = keyword_on_failure

    def set_reference_folder(self, reference_folder_path):
        """Sets where all reference images are stored.

        See `library importing` for format of the reference folder path.
        """
        self.reference_folder = reference_folder_path

    def set_screenshot_folder(self, screenshot_folder_path):
        """Sets the folder where screenshots are saved to.

        See `library importing` for more specific information.
        """
        self.screenshot_folder = screenshot_folder_path

    def reset_confidence(self):
        """Resets the confidence level to the library default.
        If no confidence was given during import, this is None."""
        LOGGER.info("Resetting confidence level to {}.".format(self.initial_confidence))
        self.confidence = self.initial_confidence

    def set_confidence(self, new_confidence):
        """Sets the accuracy when finding images.

        ``new_confidence`` is a decimal number between 0 and 1 inclusive.

        See `Confidence level` about additional dependencies that needs to be
        installed before this keyword has any effect.
        """
        if new_confidence is not None:
            try:
                new_confidence = float(new_confidence)
                if not 1 >= new_confidence >= 0:
                    LOGGER.warn(
                        "Unable to set confidence to {}. Value "
                        "must be between 0 and 1, inclusive.".format(new_confidence)
                    )
                else:
                    self.confidence = new_confidence
            except (TypeError, ValueError):
                LOGGER.warn("Can't set confidence to {}".format(new_confidence))
        else:
            self.confidence = None

    def set_scale_range(self, scale_min=0.8, scale_max=1.2, scale_steps=9):
        """Enables searching images across a range of scales.

        When this keyword is used, the library tests multiple scaling factors
        between ``scale_min`` and ``scale_max``. ``scale_steps`` defines the
        number of values between those limits. Without calling this keyword,
        only a scale of ``1.0`` is considered during image search.
        """
        try:
            self.scale_min = float(scale_min)
            self.scale_max = float(scale_max)
            self.scale_steps = int(scale_steps)
            if self.scale_steps < 1 or self.scale_min <= 0 or self.scale_max <= 0:
                raise ValueError
            self.scale_enabled = True
        except (TypeError, ValueError):
            LOGGER.warn(
                "Invalid scale range: min=%s max=%s steps=%s" % (
                    scale_min, scale_max, scale_steps
                )
            )

    def reset_scale_range(self):
        """Disables multi-scale search and resets to defaults."""
        self.scale_enabled = False
        self.scale_min = 0.8
        self.scale_max = 1.2
        self.scale_steps = 9
