# -*- coding: utf-8 -*-
import pyautogui as ag
"""Mouse interaction keywords."""

from ..errors import MouseException


class _Mouse(object):
    """Mixin implementing mouse related actions."""

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
