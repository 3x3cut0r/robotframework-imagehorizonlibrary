# -*- coding: utf-8 -*-
import pyautogui as ag
"""Mouse interaction keywords."""

from ..errors import MouseException


class _Mouse(object):
    """Mixin implementing mouse related actions."""

    def _normalize_point(self, x_value, y_value):
        """Return a tuple of integers from separate values or a sequence."""

        if y_value is None:
            if isinstance(x_value, (list, tuple)):
                if len(x_value) < 2:
                    raise MouseException(
                        "Invalid number of coordinates. Please give either (x, y) or x, y."
                    )
                x_value, y_value = x_value[:2]
            else:
                raise MouseException(
                    "Invalid number of coordinates. Please give either (x, y) or x, y."
                )
        try:
            return int(x_value), int(y_value)
        except (TypeError, ValueError):
            raise MouseException(
                "Coordinates %s are not integers" % ((x_value, y_value),)
            )

    def _normalize_offsets(self, x_value, y_value):
        """Return relative offsets as integers from values or sequences."""

        if y_value is None:
            if isinstance(x_value, (list, tuple)):
                if len(x_value) < 2:
                    raise MouseException(
                        "Invalid number of offsets. Please give either (x, y) or x, y."
                    )
                x_value, y_value = x_value[:2]
            else:
                raise MouseException(
                    "Invalid number of offsets. Please give either (x, y) or x, y."
                )
        try:
            return int(x_value), int(y_value)
        except (TypeError, ValueError):
            raise MouseException(
                "Offsets %s are not integers" % ((x_value, y_value),)
            )

    def _to_boolean(self, value, argument_name):
        """Convert Robot Framework truthy/falsey values to bool."""

        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(int(value))
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("true", "1", "yes", "on"):
                return True
            if normalized in ("false", "0", "no", "off"):
                return False
        raise MouseException('Invalid argument "%s" for `%s`' % (value, argument_name))

    def _click_to_the_direction_of(self, direction, location, offset,
                                   clicks, button, interval):
        """Click relative to a location in a given direction.

        Parameters
        ----------
        direction : str
            One of ``'up'``, ``'down'``, ``'left'`` or ``'right'``.
        location : sequence
            Sequence ``(x, y[, score, scale])`` representing a screen
            coordinate. Additional values such as match score or scale are
            ignored.
        offset : int
            Distance in pixels from ``location`` towards ``direction``.
        clicks : int
            Number of clicks to perform.
        button : str
            Mouse button to use, e.g. ``'left'``.
        interval : float
            Delay between clicks.

        Returns
        -------
        None
        """
        raise NotImplementedError('This is defined in the main class.')

    def click_to_the_above_of(self, location, offset, clicks=1,
                              button='left', interval=0.0):
        """Click above a location by a given pixel offset.

        Parameters
        ----------
        location : sequence
            Sequence ``(x, y[, score, scale])`` describing a screen
            coordinate. Extra values like match score or scale are ignored.
        offset : int
            Number of pixels the click is moved upwards from ``location``.
        clicks : int, optional
            How many times to click. Defaults to ``1``.
        button : str, optional
            Mouse button to use. Defaults to ``'left'``. See ``Click`` for
            valid values.
        interval : float, optional
            Delay between consecutive clicks. Defaults to ``0.0`` seconds.

        Returns
        -------
        None

        Examples
        --------
        | ${image location}=    | Locate             | my image |        |
        | Click To The Above Of | ${image location}  | 50       |        |
        | @{coordinates}=       | Create List        | ${600}   | ${500} |
        | Click To The Above Of | ${coordinates}     | 100      |        |
        """
        self._click_to_the_direction_of('up', location, offset,
                                        clicks, button, interval)

    def click_to_the_below_of(self, location, offset, clicks=1,
                              button='left', interval=0.0):
        """Click below a location by a given pixel offset.

        Parameters are documented in :py:meth:`click_to_the_above_of`.
        """
        self._click_to_the_direction_of('down', location, offset,
                                        clicks, button, interval)

    def click_to_the_left_of(self, location, offset, clicks=1,
                             button='left', interval=0.0):
        """Click left of a location by a given pixel offset.

        Parameters are documented in :py:meth:`click_to_the_above_of`.
        """
        self._click_to_the_direction_of('left', location, offset,
                                        clicks, button, interval)

    def click_to_the_right_of(self, location, offset, clicks=1,
                              button='left', interval=0.0):
        """Click right of a location by a given pixel offset.

        Parameters are documented in :py:meth:`click_to_the_above_of`.
        """
        self._click_to_the_direction_of('right', location, offset,
                                        clicks, button, interval)

    def move_to(self, *coordinates):
        """Move the mouse pointer to absolute screen coordinates.

        Parameters
        ----------
        *coordinates
            Either a two-item sequence ``(x, y)`` or separate ``x`` and ``y``
            values.

        Returns
        -------
        None

        Examples
        --------
        | Move To         | 25             | 150       |     |
        | @{coordinates}= | Create List    | 25        | 150 |
        | Move To         | ${coordinates} |           |     |
        | ${coords}=      | Evaluate       | (25, 150) |     |
        | Move To         | ${coords}      |           |     |

        Notes
        -----
        X grows from left to right and Y grows from top to bottom, meaning the
        top-left corner of the screen is ``(0, 0)``.
        """
        if len(coordinates) > 2 or (len(coordinates) == 1 and
                                    type(coordinates[0]) not in (list, tuple)):
            raise MouseException('Invalid number of coordinates. Please give '
                                 'either (x, y) or x, y.')
        if len(coordinates) == 2:
            coordinates = (coordinates[0], coordinates[1])
        else:
            coordinates = coordinates[0]
        try:
            coordinates = [int(coord) for coord in coordinates]
        except ValueError:
            raise MouseException('Coordinates %s are not integers' %
                                 (coordinates,))
        ag.moveTo(*coordinates)

    def move_rel(self, x_offset, y_offset=None, duration=0.0):
        """Move the mouse pointer relative to its current position.

        Parameters
        ----------
        x_offset : int or sequence
            Horizontal distance to move. Positive values move right, negative
            values move left. Can also be a two-item sequence ``(x, y)``
            containing both offsets.
        y_offset : int, optional
            Vertical distance to move. Positive values move down, negative
            values move up. Ignored when ``x_offset`` is a sequence.
        duration : float, optional
            Time in seconds for the move. Defaults to ``0.0`` for an instant
            jump.

        Returns
        -------
        None

        Examples
        --------
        | Move Rel | 50 | -25 |            |
        | Move Rel | ${offsets} |          | # where ${offsets} is [50, -25]
        | Move Rel | 10 | 20 | duration=1 |
        """

        x_offset, y_offset = self._normalize_offsets(x_offset, y_offset)
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            raise MouseException('Invalid argument "%s" for `duration`' % duration)
        ag.moveRel(x_offset, y_offset, duration=duration)

    def mouse_down(self, button='left'):
        """Press and hold a mouse button.

        Parameters
        ----------
        button : str, optional
            Mouse button to press. Defaults to ``'left'``.

        Returns
        -------
        None
        """
        ag.mouseDown(button=button)

    def mouse_up(self, button='left'):
        """Release a previously pressed mouse button.

        Parameters
        ----------
        button : str, optional
            Mouse button to release. Defaults to ``'left'``.

        Returns
        -------
        None
        """
        ag.mouseUp(button=button)

    def click(self, button='left'):
        """Click once with the specified mouse button.

        Parameters
        ----------
        button : str, optional
            Which mouse button to use. Valid values are ``'left'``, ``'right'``
            and ``'middle'``. Defaults to ``'left'``.

        Returns
        -------
        None
        """
        ag.click(button=button)

    def double_click(self, button='left', interval=0.0):
        """Double-click with the specified mouse button.

        Parameters
        ----------
        button : str, optional
            Mouse button to use. See :py:meth:`click` for valid values.
        interval : float, optional
            Time between the two clicks in seconds. Defaults to ``0.0``.

        Returns
        -------
        None
        """
        ag.doubleClick(button=button, interval=float(interval))

    def triple_click(self, button='left', interval=0.0):
        """Triple-click with the specified mouse button.

        Parameters
        ----------
        button : str, optional
            Mouse button to use. See :py:meth:`click` for valid values.
        interval : float, optional
            Delay between individual clicks in seconds. See
            :py:meth:`double_click` for details. Defaults to ``0.0``.

        Returns
        -------
        None
        """
        ag.tripleClick(button=button, interval=float(interval))

    def drag_to(self, x, y=None, duration=0.0, button='left', mouse_down_up=True):
        """Drag the mouse pointer to absolute screen coordinates.

        Parameters
        ----------
        x : int or sequence
            Target ``x`` coordinate or sequence ``(x, y[, score, scale])``.
        y : int, optional
            Target ``y`` coordinate. Ignored when ``x`` is a sequence.
        duration : float, optional
            Time in seconds for the drag. Defaults to ``0.0``.
        button : str, optional
            Mouse button to use for the drag. Defaults to ``'left'``.
        mouse_down_up : bool, optional
            Whether to automatically press and release the button during the
            drag. Defaults to ``True``.

        Returns
        -------
        None
        """

        x, y = self._normalize_point(x, y)
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            raise MouseException('Invalid argument "%s" for `duration`' % duration)
        mouse_down_up = self._to_boolean(mouse_down_up, 'mouse_down_up')
        ag.dragTo(x, y, duration=duration, button=button, mouseDownUp=mouse_down_up)

    def drag_rel(self, x_offset, y_offset=None, duration=0.0, button='left',
                 mouse_down_up=True):
        """Drag the mouse pointer relative to its current position.

        Parameters
        ----------
        x_offset : int or sequence
            Horizontal offset for the drag. Can be given as ``(x, y[, ...])``
            sequence containing both offsets.
        y_offset : int, optional
            Vertical offset for the drag. Ignored when ``x_offset`` is a
            sequence.
        duration : float, optional
            Time in seconds for the drag. Defaults to ``0.0``.
        button : str, optional
            Mouse button to use. Defaults to ``'left'``.
        mouse_down_up : bool, optional
            Whether to automatically press and release the button. Defaults to
            ``True``.

        Returns
        -------
        None
        """

        x_offset, y_offset = self._normalize_offsets(x_offset, y_offset)
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            raise MouseException('Invalid argument "%s" for `duration`' % duration)
        mouse_down_up = self._to_boolean(mouse_down_up, 'mouse_down_up')
        ag.dragRel(x_offset, y_offset, duration=duration, button=button,
                   mouseDownUp=mouse_down_up)

    def scroll(self, clicks, x=None, y=None):
        """Scroll vertically by a number of clicks.

        Parameters
        ----------
        clicks : int
            Number of scroll steps. Positive values scroll up, negative values
            scroll down.
        x : int or sequence, optional
            X coordinate at which to perform the scroll. Accepts ``(x, y[, ...])``
            sequences. Defaults to the current mouse position when omitted.
        y : int, optional
            Y coordinate for the scroll. Ignored when ``x`` is a sequence.

        Returns
        -------
        None
        """

        try:
            clicks = int(clicks)
        except (TypeError, ValueError):
            raise MouseException('Invalid argument "%s" for `clicks`' % clicks)
        if x is None and y is None:
            ag.scroll(clicks)
        else:
            x, y = self._normalize_point(x, y)
            ag.scroll(clicks, x=x, y=y)

    def scroll_horizontally(self, clicks):
        """Scroll horizontally by a number of clicks.

        Parameters
        ----------
        clicks : int
            Number of scroll steps. Positive values scroll right, negative
            values scroll left.

        Returns
        -------
        None
        """

        try:
            clicks = int(clicks)
        except (TypeError, ValueError):
            raise MouseException('Invalid argument "%s" for `clicks`' % clicks)
        ag.hscroll(clicks)

    def scroll_vertically(self, clicks):
        """Scroll vertically by a number of clicks.

        Parameters
        ----------
        clicks : int
            Number of scroll steps. Positive values scroll down, negative
            values scroll up.

        Returns
        -------
        None
        """

        try:
            clicks = int(clicks)
        except (TypeError, ValueError):
            raise MouseException('Invalid argument "%s" for `clicks`' % clicks)
        ag.vscroll(clicks)

    def set_global_pause(self, seconds):
        """Set the global PyAutoGUI pause between actions in seconds."""

        try:
            ag.PAUSE = float(seconds)
        except (TypeError, ValueError):
            raise MouseException('Invalid argument "%s" for `seconds`' % seconds)

    def set_failsafe(self, enabled=True):
        """Enable or disable the PyAutoGUI failsafe corner detection."""

        ag.FAILSAFE = self._to_boolean(enabled, 'enabled')
