# -*- coding: utf-8 -*-
from os import listdir
from os.path import abspath, basename, dirname, isdir, isfile, splitext, join as path_join
from time import time, sleep
from contextlib import contextmanager

import pyautogui as ag
from robot.api import logger as LOGGER

# ``cv2`` imports ``numpy`` internally.  Import ``numpy`` explicitly first to
# ensure it is loaded only once and avoid "module reloaded" warnings on Python
# 3.12+.
import numpy as np

try:  # pragma: no cover - optional dependency
    import cv2
    # ``DictValue`` vanished from some OpenCV builds.  Tests exercising the
    # DNN module expect it to exist, so provide a tiny stand‑in when missing to
    # keep the library importable.  Accessing ``cv2.dnn`` may itself raise an
    # exception on some installations (or when mocked) so guard lookups
    # carefully.
    try:  # pragma: no cover - defensive lookup
        dnn_mod = getattr(cv2, "dnn", None)
    except Exception:
        dnn_mod = None
    if dnn_mod is not None and not hasattr(dnn_mod, "DictValue"):
        class _DummyDictValue:
            def __init__(self, *args, **kwargs):
                self.value = args[0] if args else None

            def __iter__(self):  # minimal interface used in tests
                return iter((self.value,))

        try:
            setattr(dnn_mod, "DictValue", _DummyDictValue)
        except Exception:  # pragma: no cover - attribute may be read-only
            pass
except Exception:  # pragma: no cover - graceful fallback when cv2 is missing
    cv2 = None

import traceback


from ..errors import (
    ImageNotFoundException,
    InvalidImageException,
    ReferenceFolderException,
    ScreenshotFolderException,
    StrategyException,
)


class _RecognizeImages(object):
    """Mixin implementing image recognition keywords."""

    dflt_timeout = 0

    pixel_ratio = 0.0

    def __get_pixel_ratio(self):
        """Calculate display pixel ratio once and cache it."""
        try:
            ratio = ag.screenshot().size[0] / ag.size().width
            self.pixel_ratio = float(ratio)
        except Exception:
            self.pixel_ratio = 1.0

    def _normalize(self, path):
        """Return an absolute path for a reference image or directory.

        Parameters
        ----------
        path : str
            Name of the reference image or folder. The value is normalized to
            lower case, spaces are converted to underscores and the reference
            folder path is prepended automatically. ``.png`` is appended if the
            file extension is missing.

        Returns
        -------
        str
            Absolute path pointing to the resolved reference image or
            directory.

        Raises
        ------
        ReferenceFolderException
            If the configured reference folder is invalid.
        InvalidImageException
            If ``path`` does not refer to an existing image or folder.
        """
        if (
            not self.reference_folder
            or not isinstance(self.reference_folder, str)
            or not isdir(self.reference_folder)
        ):
            raise ReferenceFolderException(
                "Reference folder is invalid: " '"%s"' % self.reference_folder
            )
        original_input = path
        if not isinstance(path, str) or path == "":
            raise InvalidImageException('"%s" is invalid image name.' % path)
        path = str(path.lower().replace(" ", "_"))
        path = abspath(path_join(self.reference_folder, path))
        if not path.endswith(".png") and not isdir(path):
            path += ".png"
        if not isfile(path) and not isdir(path):
            dir_name = dirname(path)
            file_name = basename(path)
            try:
                candidates = listdir(dir_name)
            except Exception:
                candidates = []
            matches = [c for c in candidates if c.lower() == file_name.lower()]
            if matches:
                selected = file_name if file_name in matches else matches[0]
                path = abspath(path_join(dir_name, selected))
            else:
                raise InvalidImageException('Image path not found: "%s".' % path)

        original = basename(original_input)
        actual = basename(path)
        orig_name, orig_ext = splitext(original.replace(" ", "_"))
        act_name, act_ext = splitext(actual)
        if orig_name.lower() == act_name.lower():
            name_case_diff = orig_name != act_name
            extension_diff = orig_ext != act_ext
            if name_case_diff or extension_diff:
                LOGGER.warn(f"Image '{original}' found as '{actual}'")
        return path

    def click_image(self, reference_image, timeout=dflt_timeout):
        """Locate an image on screen and click its center once.

        Parameters
        ----------
        reference_image : str
            Name of the reference image to search for. The value is normalized
            as described in `Reference image names`.
        timeout : int, optional
            Maximum time in seconds to wait for the image to appear. Defaults
            to :pyattr:`dflt_timeout` (0).

        Returns
        -------
        tuple
            A tuple ``(x, y, score, scale)`` with the coordinates of the match,
            optional matching score and detected scale factor.

        Raises
        ------
        ImageNotFoundException
            If ``reference_image`` cannot be located within ``timeout``.
        """
        try:
            x, y, score, scale = self.wait_for(reference_image, timeout)
        except (ImageNotFoundException, ag.ImageNotFoundException) as e:
            LOGGER.info(e)
            raise
        LOGGER.info(
            'Clicking image "%s" in position %s' % (reference_image, (x, y))
        )
        ag.click((x, y))
        return (x, y, score, scale)

    def _click_to_the_direction_of(
        self, direction, location, offset, clicks, button, interval
    ):
        """Click relative to ``location`` in ``direction`` by ``offset`` pixels.

        See :py:meth:`ImageHorizonLibrary._Mouse._click_to_the_direction_of`
        for parameter documentation.
        """
        raise NotImplementedError("This is defined in the main class.")

    def _locate_and_click_direction(
        self,
        direction,
        reference_image,
        offset,
        clicks,
        button,
        interval,
        timeout=dflt_timeout,
    ):
        """Locate ``reference_image`` and click towards ``direction``.

        Parameters are the same as in :py:meth:`click_to_the_above_of_image`.
        """
        try:
            x, y, score, scale = self.wait_for(reference_image, timeout)
        except (ImageNotFoundException, ag.ImageNotFoundException) as e:
            LOGGER.info(e)
            raise
        self._click_to_the_direction_of(
            direction, (x, y), offset, clicks, button, interval
        )

    def click_to_the_above_of_image(
        self,
        reference_image,
        offset,
        clicks=1,
        button="left",
        interval=0.0,
        timeout=dflt_timeout,
    ):
        """Click above a located reference image by a pixel offset.

        Parameters
        ----------
        reference_image : str
            Name of the reference image to locate.
        offset : int
            Distance in pixels above the image's center where the click occurs.
        clicks : int, optional
            Number of clicks to perform. Defaults to ``1``.
        button : str, optional
            Mouse button to use, e.g. ``"left"``. Defaults to ``"left"``.
        interval : float, optional
            Time between clicks in seconds. Defaults to ``0.0``.
        timeout : int, optional
            Maximum time to wait for the image in seconds. Defaults to
            :pyattr:`dflt_timeout`.

        Returns
        -------
        None
        """
        self._locate_and_click_direction(
            "up", reference_image, offset, clicks, button, interval, timeout
        )

    def click_to_the_below_of_image(
        self,
        reference_image,
        offset,
        clicks=1,
        button="left",
        interval=0.0,
        timeout=dflt_timeout,
    ):
        """Click below a located reference image by a pixel offset.

        Parameters are documented in :py:meth:`click_to_the_above_of_image`.
        """
        self._locate_and_click_direction(
            "down", reference_image, offset, clicks, button, interval, timeout
        )

    def click_to_the_left_of_image(
        self,
        reference_image,
        offset,
        clicks=1,
        button="left",
        interval=0.0,
        timeout=dflt_timeout,
    ):
        """Click left of a located reference image by a pixel offset.

        Parameters are documented in :py:meth:`click_to_the_above_of_image`.
        """
        self._locate_and_click_direction(
            "left", reference_image, offset, clicks, button, interval, timeout
        )

    def click_to_the_right_of_image(
        self,
        reference_image,
        offset,
        clicks=1,
        button="left",
        interval=0.0,
        timeout=dflt_timeout,
    ):
        """Click right of a located reference image by a pixel offset.

        Parameters are documented in :py:meth:`click_to_the_above_of_image`.
        """
        self._locate_and_click_direction(
            "right", reference_image, offset, clicks, button, interval, timeout
        )

    def copy_from_the_above_of(self, reference_image, offset, timeout=dflt_timeout):
        """Copy text above a reference image.

        The keyword triple-clicks above the located image and copies the
        selection using the platform specific copy shortcut.

        Parameters
        ----------
        reference_image : str
            Name of the reference image to locate.
        offset : int
            Offset in pixels above the image where the triple-click occurs.
        timeout : int, optional
            Maximum time to wait for the image. Defaults to
            :pyattr:`dflt_timeout`.

        Returns
        -------
        str
            The text content of the system clipboard after the copy action.
        """
        self._locate_and_click_direction(
            "up",
            reference_image,
            offset,
            clicks=3,
            button="left",
            interval=0.0,
            timeout=timeout,
        )
        return self.copy()

    def copy_from_the_below_of(self, reference_image, offset, timeout=dflt_timeout):
        """Copy text below a reference image.

        Parameters are documented in :py:meth:`copy_from_the_above_of`.
        """
        self._locate_and_click_direction(
            "down",
            reference_image,
            offset,
            clicks=3,
            button="left",
            interval=0.0,
            timeout=timeout,
        )
        return self.copy()

    def copy_from_the_left_of(self, reference_image, offset, timeout=dflt_timeout):
        """Copy text left of a reference image.

        Parameters are documented in :py:meth:`copy_from_the_above_of`.
        """
        self._locate_and_click_direction(
            "left",
            reference_image,
            offset,
            clicks=3,
            button="left",
            interval=0.0,
            timeout=timeout,
        )
        return self.copy()

    def copy_from_the_right_of(self, reference_image, offset, timeout=dflt_timeout):
        """Copy text right of a reference image.

        Parameters are documented in :py:meth:`copy_from_the_above_of`.
        """
        self._locate_and_click_direction(
            "right",
            reference_image,
            offset,
            clicks=3,
            button="left",
            interval=0.0,
            timeout=timeout,
        )
        return self.copy()

    @contextmanager
    def _suppress_keyword_on_failure(self):
        """Temporarily disable post-failure keyword during a block."""
        keyword = self.keyword_on_failure
        self.keyword_on_failure = None
        yield None
        self.keyword_on_failure = keyword

    def _get_reference_images(self, reference_image):
        """Resolve one or many reference image paths.

        Parameters
        ----------
        reference_image : str
            Name of the reference image or a directory containing images.

        Returns
        -------
        list[str]
            A list of absolute image paths. If ``reference_image`` is a single
            file, the list contains only that path. If it is a directory, all
            files within the directory are returned in alphabetical order.

        Raises
        ------
        InvalidImageException
            If ``reference_image`` refers to a directory containing
            non-image files.
        """
        is_dir = False
        try:
            if isdir(self._normalize(reference_image)):
                is_dir = True
        except InvalidImageException:
            pass
        is_file = False
        try:
            if isfile(self._normalize(reference_image)):
                is_file = True
        except InvalidImageException:
            pass
        reference_image = self._normalize(reference_image)

        reference_images = []
        if is_file:
            reference_images = [reference_image]
        elif is_dir:
            for f in listdir(self._normalize(reference_image)):
                if not isfile(self._normalize(path_join(reference_image, f))):
                    raise InvalidImageException(self._normalize(reference_image))
                reference_images.append(path_join(reference_image, f))
        return reference_images

    def _locate(self, reference_image, log_it=True):
        """Return location and scale for ``reference_image`` on screen.

        Parameters
        ----------
        reference_image : str
            Name or path of the image to locate.
        log_it : bool, optional
            If ``True`` (default), log informative messages about the search
            result.

        Returns
        -------
        tuple
            Tuple ``(x, y, score, scale)`` where ``(x, y)`` are coordinates of
            the image center, ``score`` is the matching score and ``scale``
            is the detected scaling factor or ``None`` when not available.

        Raises
        ------
        ImageNotFoundException
            If the image cannot be located.
        """
        reference_images = self._get_reference_images(reference_image)

        location = None
        score = None
        scale = 1.0
        best_score = None
        for ref_image in reference_images:
            try:
                result = self._try_locate(ref_image)
            except Exception as e:  # pragma: no cover - unexpected failures
                LOGGER.error(
                    f'Unexpected error locating "{ref_image}": {e}\n{traceback.format_exc()}'
                )
                raise

            if isinstance(result, tuple) and len(result) == 3:
                loc, scr, scl = result
            elif isinstance(result, np.ndarray) and result.shape == (3,):
                loc, scr, scl = result[0], result[1], result[2]
            else:
                loc, scr, scl = result, None, 1.0

            if loc is not None:
                if isinstance(loc, np.ndarray):
                    loc = tuple(np.asarray(loc).flatten().tolist())
                if isinstance(scr, np.ndarray):
                    scr = float(np.asarray(scr).flat[0])
                if isinstance(scl, np.ndarray):
                    scl = float(np.asarray(scl).flat[0])
                location, score, scale = loc, scr, scl
                break
            else:
                if scr is not None and (
                    best_score is None or scr > best_score
                ):
                    best_score = scr

        if location is None:
            confidence = getattr(self, "confidence", None)
            matches = 0
            if log_it:
                LOGGER.info(
                    'Image "%s" was not found on screen. '
                    "(strategy: %s, matches: %d, best score %.3f, confidence %.3f)"
                    % (
                        reference_image,
                        self.strategy,
                        matches,
                        best_score if best_score is not None else float('nan'),
                        confidence if confidence is not None else float('nan'),
                    )
                )
            self._run_on_failure()
            raise ImageNotFoundException(
                reference_image,
                best_score=best_score,
                confidence=confidence,
            )

        center_point = ag.center(location)
        x = center_point.x
        y = center_point.y
        if self.pixel_ratio == 0.0:
            self.__get_pixel_ratio()
        if self.pixel_ratio > 1:
            x = x / self.pixel_ratio
            y = y / self.pixel_ratio
        if log_it:
            LOGGER.info(
                'Image "%s" found at %r (score %.3f, scale %.2f, strategy: %s)'
                % (
                    reference_image,
                    (x, y),
                    score if score is not None else float('nan'),
                    scale,
                    self.strategy,
                )
            )
        return (x, y, score, scale)

    def _locate_all(self, reference_image, haystack_image=None):
        """Locate all occurrences of a reference image.

        Parameters
        ----------
        reference_image : str
            Name or path of the image to search for.
        haystack_image : array-like, optional
            Pre-captured screenshot to search in. If ``None``, a new screenshot
            of the screen is taken.

        Returns
        -------
        list[tuple]
            A list of tuples ``(location, score, scale)`` for each match. The
            list may be empty if no matches are found.

        Raises
        ------
        InvalidImageException
            If ``reference_image`` resolves to multiple files.
        """
        reference_images = self._get_reference_images(reference_image)
        if len(reference_images) > 1:
            raise InvalidImageException(
                f'Locating ALL occurences of MANY files ({", ".join(reference_images)}) is not supported.'
            )
        locations = self._try_locate(
            reference_images[0], locate_all=True, haystack_image=haystack_image
        )
        return locations

    def does_exist(self, reference_image):
        """Check whether a reference image exists on the screen.

        Parameters
        ----------
        reference_image : str
            Name of the reference image to locate.

        Returns
        -------
        bool
            ``True`` if the image was found, ``False`` otherwise. The keyword
            never raises an exception.
        """
        with self._suppress_keyword_on_failure():
            try:
                self._locate(reference_image, log_it=True)
                return True
            except (ImageNotFoundException, ag.ImageNotFoundException):
                return False

    def locate(self, reference_image):
        """Locate image on screen.

        Parameters
        ----------
        reference_image : str
            Name or path of the image to locate.

        Returns
        -------
        tuple
            Tuple ``(x, y, score, scale)`` describing the match.

        Raises
        ------
        ImageNotFoundException
            If the image is not found on screen.
        """
        try:
            return self._locate(reference_image)
        except (ImageNotFoundException, ag.ImageNotFoundException) as e:
            LOGGER.info(e)
            raise

    def locate_all(self, reference_image):
        """Locate all occurrences of an image on screen.

        Parameters
        ----------
        reference_image : str
            Name or path of the image to locate.

        Returns
        -------
        list[tuple]
            List of tuples ``(x, y, score, scale)`` describing each match.

        Raises
        ------
        InvalidImageException
            If ``reference_image`` resolves to multiple files.
        """
        matches = []
        locations = self._locate_all(reference_image)
        if self.pixel_ratio == 0.0:
            self.__get_pixel_ratio()
        for loc, score, scale in locations:
            center = ag.center(loc)
            x, y = center.x, center.y
            if self.pixel_ratio > 1:
                x = x / self.pixel_ratio
                y = y / self.pixel_ratio
            matches.append((x, y, score, scale))
        return matches

    def wait_for(self, reference_image, timeout=10):
        """Wait until an image appears on the screen.

        Parameters
        ----------
        reference_image : str
            Name of the reference image to locate.
        timeout : float, optional
            Maximum number of seconds to wait. Defaults to ``10``.

        Returns
        -------
        tuple
            Tuple ``(x, y, score, scale)`` describing the match.

        Raises
        ------
        ImageNotFoundException
            If the image is not found within the timeout.
        """
        stop_time = time() + float(timeout)
        location = None
        last_exc = None
        with self._suppress_keyword_on_failure():
            while True:
                try:
                    location = self._locate(reference_image, log_it=True)
                    break
                except (
                    InvalidImageException,
                    ReferenceFolderException,
                    StrategyException,
                    ScreenshotFolderException,
                ):
                    # These indicate a permanent misconfiguration and should not
                    # be retried within this loop.
                    raise
                except Exception as e:  # pragma: no cover - defensive catch
                    last_exc = e
                    if time() > stop_time:
                        break
                    sleep(0.1)
        if location is None:
            self._run_on_failure()
            if last_exc is not None:
                # Propagate original error after waiting for the timeout. If the
                # error was not our ``ImageNotFoundException`` it represents an
                # unexpected failure and is raised as-is.
                raise last_exc
            raise ImageNotFoundException(self._normalize(reference_image))
        x, y, score, scale = location
        LOGGER.info(
            'Image "%s" found at %r (score %.3f, scale %.2f)'
            % (
                reference_image,
                (x, y),
                score if score is not None else float('nan'),
                scale,
            )
        )
        return location

    def debug_image(self, reference_folder=None, minimize=False, dialog_default_dir=None):
        """Halts the test execution and opens the image debugger UI.

        Whenever you encounter problems with the recognition accuracy of a reference image,
        you should place this keyword just before the line in question. Example:

        | Debug Image
        | Wait For  hard_to_find_button

        The test will halt at this position and open the debugger UI. Use it as follows:

        - Select the reference image (`hard_to_find_button`)
        - Click the button "Detect reference image" for the strategy you want to test (default/edge). The GUI hides itself while it takes the screenshot of the current application when ``minimize`` is ``True``.
        - The Image Viewer at the botton shows the screenshot with all regions where the reference image was found.
        - "Matches Found": More than one match means that either `conficence` is set too low or that the reference image is visible multiple times. If the latter is the case, you should first detect a unique UI element and use relative keywords like `Click To The Right Of`.
        - "Max peak value" (only `edge`) gives feedback about the detection accuracy of the best match and is measured as a float number between 0 and 1. A peak value above _confidence_ results in a match.
        - "Edge detection debugger" (only `edge`) opens another window where both the reference and screenshot images are shown before and after the edge detection and is very helpful to loearn how the sigma and low/high threshold parameters lead to different results.
        - The field "Keyword to use this strategy" shows how to set the strategy to the current settings. Just copy the line and paste it into the test:

        | Set Strategy  edge  edge_sigma=2.0  edge_low_threshold=0.1  edge_high_threshold=0.3
        | Wait For  hard_to_find_button

        ``reference_folder`` can be given to temporarily override the folder
        from which reference images are loaded.

        ``minimize`` controls whether the debugger window is minimised before
        taking a screenshot. By default the window is kept visible; set
        ``minimize`` to ``True`` to minimise it. When visible, the window's area
        is masked so that matches inside the debugger are ignored.

        ``dialog_default_dir`` specifies the folder shown when the directory
        chooser is opened and the current reference folder is invalid or not
        accessible. If omitted, the user's home directory is used.

        The purpose of this keyword is *solely for debugging purposes*; don't
        use it in production!"""
        from .ImageDebugger import ImageDebugger

        previous_reference_folder = self.reference_folder
        if reference_folder is not None:
            self.set_reference_folder(reference_folder)
        try:
            debug_app = ImageDebugger(
                self,
                minimize=minimize,
                dialog_default_dir=dialog_default_dir,
            )
        finally:
            if reference_folder is not None:
                self.set_reference_folder(previous_reference_folder)


class _StrategyPyautogui:
    """Image matching strategy based on PyAutoGUI."""

    def __init__(self, image_horizon_instance):
        """Store reference to the owning ImageHorizonLibrary instance."""
        self.ih_instance = image_horizon_instance

    def _try_locate(self, ref_image, haystack_image=None, locate_all=False):
        """Locate a reference image using PyAutoGUI's matching.

        Parameters
        ----------
        ref_image : str
            Path to the reference image.
        haystack_image : image, optional
            Screenshot to search in. If ``None``, a screenshot is taken.
        locate_all : bool, optional
            If ``True``, return all matches; otherwise only the first match.

        Returns
        -------
        list or tuple
            When ``locate_all`` is ``False`` a tuple ``(location, score, scale)``
            is returned. ``location`` may be ``None`` if no match was found. When
            ``locate_all`` is ``True`` a list of such tuples is returned.
        """
        ih = self.ih_instance

        if haystack_image is None:
            haystack_image = ag.screenshot()

        if getattr(ih, "has_cv", False) and getattr(ih, "scale_enabled", False):
            haystack_np = np.array(haystack_image)
            if haystack_np.ndim == 3:
                haystack_gray = cv2.cvtColor(haystack_np, cv2.COLOR_RGB2GRAY)
            else:
                haystack_gray = haystack_np
            needle_gray = cv2.imread(ref_image, cv2.IMREAD_GRAYSCALE)
            scales = np.linspace(ih.scale_min, ih.scale_max, ih.scale_steps)
            confidence = ih.confidence or 0.9
            if locate_all:
                matches = []
                for scale in scales:
                    scaled_height = max(1, int(needle_gray.shape[0] * scale))
                    scaled_width = max(1, int(needle_gray.shape[1] * scale))
                    if (
                        scaled_height > haystack_gray.shape[0]
                        or scaled_width > haystack_gray.shape[1]
                    ):
                        continue
                    scaled_needle = cv2.resize(
                        needle_gray,
                        (scaled_width, scaled_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    res = cv2.matchTemplate(
                        haystack_gray, scaled_needle, cv2.TM_CCOEFF_NORMED
                    )
                    peaks = np.where(res >= confidence)
                    for y, x in zip(*peaks):
                        matches.append(
                            (
                                (x, y, scaled_width, scaled_height),
                                float(res[y][x]),
                                scale,
                            )
                        )
                return matches
            else:
                best_loc = None
                best_score = -1.0
                best_scale = 1.0
                for scale in scales:
                    scaled_height = max(1, int(needle_gray.shape[0] * scale))
                    scaled_width = max(1, int(needle_gray.shape[1] * scale))
                    if (
                        scaled_height > haystack_gray.shape[0]
                        or scaled_width > haystack_gray.shape[1]
                    ):
                        continue
                    scaled_needle = cv2.resize(
                        needle_gray,
                        (scaled_width, scaled_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    res = cv2.matchTemplate(
                        haystack_gray, scaled_needle, cv2.TM_CCOEFF_NORMED
                    )
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    if max_val > best_score:
                        best_score = float(max_val)
                        best_loc = (
                            max_loc[0],
                            max_loc[1],
                            scaled_width,
                            scaled_height,
                        )
                        best_scale = scale
                if best_loc is None or best_score < confidence:
                    return (None, best_score if best_score >= 0 else None, best_scale)
                return (best_loc, best_score, best_scale)

        if locate_all:
            locate_func = ag.locateAll
        else:
            locate_func = ag.locate  # Copy below,take screenshots

        with ih._suppress_keyword_on_failure():
            try:
                if ih.has_cv and ih.confidence:
                    location_res = locate_func(
                        ref_image, haystack_image, confidence=ih.confidence
                    )
                else:
                    if ih.confidence:
                        LOGGER.warn(
                            "Can't set confidence because you don't "
                            "have OpenCV (python3-opencv) installed "
                            "or a confidence level was not given."
                        )
                    location_res = locate_func(ref_image, haystack_image)
            except (ImageNotFoundException, ag.ImageNotFoundException) as ex:
                LOGGER.info(ex)
                location_res = None

        if locate_all:
            if location_res is None:
                locations = []
            else:
                locations = [tuple(box) for box in location_res]
            scores = []
            if ih.has_cv and locations:
                try:
                    haystack_np = np.array(haystack_image)
                    if haystack_np.ndim == 3:
                        haystack_gray = cv2.cvtColor(haystack_np, cv2.COLOR_RGB2GRAY)
                    else:
                        haystack_gray = haystack_np
                    needle_gray = cv2.imread(ref_image, cv2.IMREAD_GRAYSCALE)
                    result = cv2.matchTemplate(
                        haystack_gray, needle_gray, cv2.TM_CCOEFF_NORMED
                    )
                    for x, y, w, h in locations:
                        scores.append(float(result[y][x]))
                except Exception:
                    scores = [None] * len(locations)
            else:
                scores = [None] * len(locations)
            return [(loc, scr, 1.0) for loc, scr in zip(locations, scores)]

        location = location_res
        score = None
        if ih.has_cv and location is not None:
            try:
                haystack_np = np.array(haystack_image)
                if haystack_np.ndim == 3:
                    haystack_gray = cv2.cvtColor(haystack_np, cv2.COLOR_RGB2GRAY)
                else:
                    haystack_gray = haystack_np
                needle_gray = cv2.imread(ref_image, cv2.IMREAD_GRAYSCALE)
                res = cv2.matchTemplate(
                    haystack_gray, needle_gray, cv2.TM_CCOEFF_NORMED
                )
                _, max_val, _, _ = cv2.minMaxLoc(res)
                score = float(max_val)
            except Exception:
                score = None
        return (location, score, 1.0)
class _StrategyCv2:
    """Image matching strategy using OpenCV edge detection."""

    _CV_DEFAULT_CONFIDENCE = 0.9

    def __init__(self, image_horizon_instance):
        """Store reference to the owning ImageHorizonLibrary instance."""
        self.ih_instance = image_horizon_instance

    def _try_locate(self, ref_image, haystack_image=None, locate_all=False):
        """Locate a reference image using OpenCV edge detection.

        Parameters
        ----------
        ref_image : str
            Path to the reference image.
        haystack_image : array-like, optional
            Screenshot to search in. If ``None``, a screenshot is taken.
        locate_all : bool, optional
            If ``True``, return all matches for debugging.

        Returns
        -------
        tuple or list
            When ``locate_all`` is ``False``: ``(location, score, scale)`` or
            ``(None, score, scale)`` if not found.
            When ``locate_all`` is ``True``: a list of such tuples for each
            detected match.
        """

        ih = self.ih_instance
        confidence = ih.confidence or self._CV_DEFAULT_CONFIDENCE
        with ih._suppress_keyword_on_failure():
            needle_img = cv2.imread(ref_image, cv2.IMREAD_GRAYSCALE)
            if haystack_image is None:
                haystack_img_gray = cv2.cvtColor(
                    np.array(ag.screenshot()), cv2.COLOR_RGB2GRAY
                )
            else:
                if len(haystack_image.shape) == 2:
                    haystack_img_gray = haystack_image
                else:
                    haystack_img_gray = cv2.cvtColor(
                        haystack_image, cv2.COLOR_BGR2GRAY
                    )

            ih.haystack_edge = self.detect_edges(haystack_img_gray)

            if ih.scale_enabled:
                scales = np.linspace(ih.scale_min, ih.scale_max, ih.scale_steps)
            else:
                scales = [1.0]
            best_location = None
            best_scale = 1.0
            best_score = -1.0
            best_needle = None
            locations = []

            for scale in scales:
                scaled_height = max(1, int(needle_img.shape[0] * scale))
                scaled_width = max(1, int(needle_img.shape[1] * scale))
                scaled_needle = cv2.resize(
                    needle_img,
                    (scaled_width, scaled_height),
                    interpolation=cv2.INTER_AREA,
                )
                needle_edge = self.detect_edges(scaled_needle)
                haystack_edge_uint8 = (ih.haystack_edge * 255).astype(np.uint8)
                needle_edge_uint8 = (needle_edge * 255).astype(np.uint8)
                peakmap = cv2.matchTemplate(
                    haystack_edge_uint8, needle_edge_uint8, cv2.TM_CCOEFF_NORMED
                )

                if locate_all:
                    peaks = np.where(peakmap >= confidence)
                    for y, x in zip(*peaks):
                        peak = peakmap[y][x]
                        if peak > confidence:
                            loc = (x, y, scaled_width, scaled_height)
                            locations.append((loc, peak, scale))
                        if peak > best_score:
                            best_score = peak
                            best_scale = scale
                            best_location = loc
                            ih.needle_edge = needle_edge
                            ih.peakmap = peakmap
                            best_needle = scaled_needle
                else:
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(peakmap)
                    peak = max_val
                    x, y = max_loc
                    if peak > best_score:
                        best_score = peak
                        best_scale = scale
                        best_location = (
                            x,
                            y,
                            scaled_width,
                            scaled_height,
                        )
                        ih.needle_edge = needle_edge
                        ih.peakmap = peakmap
                        best_needle = scaled_needle

            if locate_all:
                return locations

            if best_score > confidence and ih.validate_match and ih.has_cv:
                margin = ih.validation_margin or 0
                haystack_uint8 = haystack_img_gray.astype(np.uint8)
                needle_uint8 = best_needle.astype(np.uint8)
                x0 = int(max(0, best_location[0] - margin))
                y0 = int(max(0, best_location[1] - margin))
                x1 = int(
                    min(
                        haystack_uint8.shape[1],
                        best_location[0] + best_location[2] + margin,
                    )
                )
                y1 = int(
                    min(
                        haystack_uint8.shape[0],
                        best_location[1] + best_location[3] + margin,
                    )
                )
                region = haystack_uint8[y0:y1, x0:x1]
                if region.shape[0] >= needle_uint8.shape[0] and region.shape[1] >= needle_uint8.shape[1]:
                    res = cv2.matchTemplate(
                        region, needle_uint8, cv2.TM_CCOEFF_NORMED
                    )
                    if res.size == 0 or res.max() < confidence:
                        return None, best_score, best_scale

            if best_score > confidence:
                return best_location, best_score, best_scale
            return None, best_score, best_scale

    def _auto_edge_parameters(self, img):
        """Derive edge detection parameters from image statistics.

        Parameters
        ----------
        img : array-like
            Grayscale image data.

        Returns
        -------
        tuple
            Tuple ``(sigma, low, high)`` containing suitable parameters for the
            Canny edge detector.

        Notes
        -----
        If the image uses 8-bit integers, it is converted to floating point so
        that thresholding works reliably. ``cv2.threshold`` with ``OTSU``
        provides a reasonable high threshold, while the standard deviation of
        the image is used to scale ``sigma`` and the low threshold is derived as
        a fraction of the high threshold.
        """

        # ``img`` may be provided as a list or other array-like type; make sure
        # we are working with a NumPy array of floats to avoid type promotion
        # issues which manifested as ``umr_maximum``/``float`` errors on some
        # platforms.
        img_float = np.asarray(img, dtype=np.float64)
        max_val = float(np.max(img_float)) if img_float.size else 0.0
        if max_val > 1.0:
            img_float /= 255.0

        std = float(np.std(img_float))
        sigma = float(np.clip(std * 3, 0.1, 5.0))

        img_uint8 = np.rint(img_float * 255).astype(np.uint8)
        t, _ = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = float(t) / 255.0
        low = float(max(0.0, t * 0.5))
        high = float(min(1.0, t * 1.5))
        if high <= low:
            high = float(min(1.0, low + 0.05))

        return float(sigma), float(low), float(high)

    def _detect_edges(self, img, sigma, low, high):
        """Run Canny edge detection with optional preprocessing.

        Available filters for preprocessing are ``gaussian``, ``median``,
        ``erode`` and ``dilate``. Preprocessing is only performed when the
        ``edge_preprocess`` option is set *and* OpenCV is available.

        Parameters
        ----------
        img : array-like
            Grayscale image data.
        sigma : float
            Standard deviation of the Gaussian blur applied before edge
            detection.
        low : float
            Lower gradient magnitude threshold (0.0–1.0).
        high : float
            Upper gradient magnitude threshold (0.0–1.0).

        Returns
        -------
        array-like
            Edge-detected binary image.
        """
        preprocess = self.ih_instance.edge_preprocess
        ksize = int(self.ih_instance.edge_kernel_size or 3)

        img_float = img.astype(np.float32)
        if img_float.max() > 1.0:
            img_uint8 = img_float.astype(np.uint8)
        else:
            img_uint8 = (img_float * 255).astype(np.uint8)

        if preprocess and self.ih_instance.has_cv:
            if preprocess == "gaussian":
                img_uint8 = cv2.GaussianBlur(img_uint8, (ksize, ksize), 0)
            elif preprocess == "median":
                img_uint8 = cv2.medianBlur(img_uint8, ksize)
            elif preprocess == "erode":
                img_uint8 = cv2.erode(img_uint8, np.ones((ksize, ksize), np.uint8))
            elif preprocess == "dilate":
                img_uint8 = cv2.dilate(img_uint8, np.ones((ksize, ksize), np.uint8))

        if sigma and sigma > 0:
            img_uint8 = cv2.GaussianBlur(img_uint8, (0, 0), sigma)

        edge_img = cv2.Canny(img_uint8, int(low * 255), int(high * 255))
        return edge_img.astype(np.float32) / 255.0

    def detect_edges(self, img):
        """Apply edge detection on a given image.

        Parameters
        ----------
        img : array-like
            Grayscale image data to process.

        Returns
        -------
        array-like
            Binary edge image computed with the Canny detector. If explicit
            parameters are not provided, sensible defaults are derived from the
            image statistics.
        """

        sigma = self.ih_instance.edge_sigma
        low = self.ih_instance.edge_low_threshold
        high = self.ih_instance.edge_high_threshold
        try:
            sigma = float(sigma) if sigma is not None else None
            low = float(low) if low is not None else None
            high = float(high) if high is not None else None
        except (TypeError, ValueError):
            sigma = low = high = None

        if sigma is None or low is None or high is None:
            auto_sigma, auto_low, auto_high = self._auto_edge_parameters(img)
            if sigma is None:
                sigma = auto_sigma
            if low is None:
                low = auto_low
            if high is None:
                high = auto_high

        return self._detect_edges(img, sigma, low, high)
