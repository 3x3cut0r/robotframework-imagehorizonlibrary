# -*- coding: utf-8 -*-
import os
import shlex

from os.path import abspath, dirname, join as path_join
from subprocess import PIPE, Popen
from unittest import SkipTest, TestCase
from warnings import warn

from unittest.mock import MagicMock, patch


SRCDIR = path_join(abspath(dirname(__file__)), '..', '..', 'src')


class TestMainClass(TestCase):
    def setUp(self):
        self.pyautogui_mock = MagicMock()
        self.Tk_mock = MagicMock()
        self.clipboard_mock = MagicMock()
        self.clipboard_mock.clipboard_get.return_value = 'copied text'
        self.Tk_mock.Tk.return_value = self.clipboard_mock
        self.patcher = patch.dict('sys.modules',
                                  {'pyautogui': self.pyautogui_mock,
                                   'tkinter': self.Tk_mock})
        self.patcher.start()
        from ImageHorizonLibrary import ImageHorizonLibrary
        self.lib = ImageHorizonLibrary()

    def tearDown(self):
        for mock in (self.Tk_mock, self.clipboard_mock, self.pyautogui_mock):
            mock.reset_mock()
        self.patcher.stop()

    def test_copy(self):
        from ImageHorizonLibrary import ImageHorizonLibrary

        with patch.object(ImageHorizonLibrary, '_press') as press_mock:
            retval = self.lib.copy()
            self.assertEqual(retval, 'copied text')
            self.clipboard_mock.clipboard_get.assert_called_once_with()
            if self.lib.is_mac:
                press_mock.assert_called_once_with('Key.command', 'c')
            else:
                press_mock.assert_called_once_with('Key.ctrl', 'c')

    def test_clipboard_content(self):
        retval = self.lib.get_clipboard_content()
        self.assertEqual(retval, 'copied text')
        self.clipboard_mock.clipboard_get.assert_called_once_with()

    def test_alert(self):
        self.lib.pause()
        self.pyautogui_mock.alert.assert_called_once_with(
            button='Continue', text='Test execution paused.', title='Pause')

    def _get_cmd(self, jython, path):
        cmd = ('JYTHONPATH={path} {jython} -c '
               '"from ImageHorizonLibrary import ImageHorizonLibrary"')
        return cmd.format(jython=jython, path=path)

    @patch.dict('sys.modules', {'tkinter': None})
    def test_importing_fails_on_java(self):
        """Importing should fail when Tkinter is unavailable."""
        from importlib import import_module
        from ImageHorizonLibrary.errors import ImageHorizonLibraryError
        import sys

        sys.modules.pop('ImageHorizonLibrary', None)
        with self.assertRaises(ImageHorizonLibraryError):
            import_module('ImageHorizonLibrary')

    def test_set_reference_folder(self):
        self.assertEqual(self.lib.reference_folder, None)
        self.lib.set_reference_folder('/test/path')
        self.assertEqual(self.lib.reference_folder, '/test/path')

    def test_set_screenshot_folder(self):
        self.assertEqual(self.lib.screenshot_folder, None)
        self.lib.set_screenshot_folder('/test/path')
        self.assertEqual(self.lib.screenshot_folder, '/test/path')

    def test_set_confidence(self):
        self.assertEqual(self.lib.confidence, None)

        self.lib.set_confidence(0)
        self.assertEqual(self.lib.confidence, 0)

        self.lib.set_confidence(0.5)
        self.assertEqual(self.lib.confidence, 0.5)

        self.lib.set_confidence(-1)
        self.assertEqual(self.lib.confidence, 0.5)

        self.lib.set_confidence(2)
        self.assertEqual(self.lib.confidence, 0.5)

        self.lib.set_confidence(1)
        self.assertEqual(self.lib.confidence, 1)

        self.lib.set_confidence(None)
        self.assertEqual(self.lib.confidence, None)

    def test_set_confidence_with_non_numeric_string_logs_warning(self):
        with patch('ImageHorizonLibrary.LOGGER') as logger_mock:
            self.lib.set_confidence('invalid')
            logger_mock.warn.assert_called_once_with(
                "Can't set confidence to invalid"
            )
            self.assertIsNone(self.lib.confidence)
