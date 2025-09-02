from unittest import TestCase
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

mock_pyautogui = MagicMock()
mock_tk = SimpleNamespace(
    Tk=MagicMock(), ttk=MagicMock(), filedialog=MagicMock(), messagebox=MagicMock()
)
patcher = patch.dict('sys.modules', {'pyautogui': mock_pyautogui, 'tkinter': mock_tk})
patcher.start()
from ImageHorizonLibrary.recognition.ImageDebugger.image_debugger_controller import UILocatorController
patcher.stop()


class FakeView:
    def __init__(self, fail_minimise=False, fail_restore=False):
        self.fail_minimise = fail_minimise
        self.fail_restore = fail_restore
        self.state_calls = []
        self.after_called = None
        self.after_cancel_called = None

    def iconify(self):
        if self.fail_minimise:
            raise Exception('minimise failed')
        self.state_calls.append('iconify')

    def deiconify(self):
        if self.fail_restore:
            raise Exception('restore failed')
        self.state_calls.append('deiconify')

    def lift(self):
        self.state_calls.append('lift')

    def after(self, delay, func):
        self.after_called = (delay, func)
        return 'id'

    def after_cancel(self, ident):
        self.after_cancel_called = ident


class TestTakeScreenshot(TestCase):
    def _get_controller(self, view=None, model=None):
        ctrl = UILocatorController.__new__(UILocatorController)
        ctrl.view = view or FakeView()
        ctrl.model = model or MagicMock()
        return ctrl

    def test_restores_window_after_successful_capture(self):
        model = MagicMock()
        model.capture_desktop.return_value = 'img'
        view = FakeView()
        ctrl = self._get_controller(view=view, model=model)

        result = ctrl._take_screenshot()

        self.assertEqual(result, 'img')
        self.assertEqual(view.state_calls, ['iconify', 'deiconify', 'lift'])
        self.assertEqual(view.after_called[0], 10000)
        self.assertEqual(view.after_cancel_called, 'id')

    def test_restores_window_on_capture_failure(self):
        model = MagicMock()
        model.capture_desktop.side_effect = RuntimeError('boom')
        view = FakeView()
        ctrl = self._get_controller(view=view, model=model)

        with self.assertRaises(RuntimeError):
            ctrl._take_screenshot()

        self.assertEqual(view.state_calls, ['iconify', 'deiconify', 'lift'])
        self.assertEqual(view.after_cancel_called, 'id')

    def test_minimise_failure_raises_runtime_error(self):
        view = FakeView(fail_minimise=True)
        ctrl = self._get_controller(view=view, model=MagicMock())

        with self.assertRaises(RuntimeError):
            ctrl._take_screenshot()

        self.assertIsNone(view.after_called)
        self.assertIsNone(view.after_cancel_called)
