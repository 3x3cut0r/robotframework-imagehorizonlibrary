# -*- coding: utf-8 -*-
"""Keyboard interaction keywords."""

import pyautogui as ag


class _Keyboard(object):
    """Mixin implementing keyboard related actions."""

    def press_combination(self, *keys):
        """Press multiple keyboard keys simultaneously.

        Parameters
        ----------
        *keys
            Keyboard keys to press. Each key must be given as a string and
            prefixed with ``Key.``. Keys are case-insensitive.

        Returns
        -------
        None

        Examples
        --------
        | Press Combination | Key.ALT | Key.F4 |
        | Press Combination | key.ctrl | key.end |

        See also
        --------
        `PyAutoGUI keyboard keys <https://pyautogui.readthedocs.org/en/latest/keyboard.html#keyboard-keys>`_
        for the full list of supported key names.
        """
        self._press(*keys)

    def type(self, *keys_or_text):
        """Type a sequence of text fragments and/or special keys.

        Parameters
        ----------
        *keys_or_text
            Sequence of strings representing either plain text or keyboard
            keys. Special keys must be prefixed with ``Key.``.

        Returns
        -------
        None

        Examples
        --------
        | Type | separated              | Key.ENTER | by linebreak |
        | Type | Submit this with enter | Key.enter |              |
        | Type | key.windows            | notepad   | Key.enter    |
        """
        for key_or_text in keys_or_text:
            key = self._convert_to_valid_special_key(key_or_text)
            if key:
                ag.press(key)
            else:
                ag.typewrite(key_or_text)


    def type_with_keys_down(self, text, *keys):
        """Hold down keyboard keys while typing text.

        The given keys are pressed and held, the ``text`` is written, and then
        the keys are released again. This is useful for typing with modifiers
        such as ``Key.Shift``.

        Parameters
        ----------
        text : str
            The text to type while the modifier keys are held down.
        *keys
            Keyboard keys to hold down. Each key must be prefixed with ``Key.``.

        Returns
        -------
        None

        Examples
        --------
        | Type With Keys Down | write this in caps | Key.Shift |
        """
        valid_keys = self._validate_keys(keys)
        for key in valid_keys:
            ag.keyDown(key)
        ag.typewrite(text)
        for key in valid_keys:
            ag.keyUp(key)
