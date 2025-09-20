from unittest import TestCase
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

mock_pyautogui = MagicMock()
mock_tk = SimpleNamespace(
    Tk=MagicMock(), ttk=MagicMock(), filedialog=MagicMock(), messagebox=MagicMock()
)
patcher = patch.dict('sys.modules', {'pyautogui': mock_pyautogui, 'tkinter': mock_tk})
patcher.start()
from ImageHorizonLibrary.recognition.ImageDebugger.image_debugger_controller import UILocatorController
patcher.stop()


class FakeView:
    def __init__(self, fail_minimise=False, fail_restore=False, initial_state='normal'):
        self.fail_minimise = fail_minimise
        self.fail_restore = fail_restore
        self.state_calls = []
        self.after_called = None
        self.after_cancel_called = None
        self._state = initial_state
        self._x = 1
        self._y = 2
        self._w = 3
        self._h = 4

    def state(self, value=None):
        if value is None:
            return self._state
        self._state = value
        self.state_calls.append(f'state({value})')

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

    def winfo_rootx(self):
        return self._x

    def winfo_rooty(self):
        return self._y

    def winfo_width(self):
        return self._w

    def winfo_height(self):
        return self._h


class DummyCombobox:
    def __init__(self):
        self._values = None
        self.selected = None

    def __setitem__(self, key, value):
        if key != "values":
            raise KeyError(key)
        self._values = value

    @property
    def values(self):
        return self._values

    def set(self, value):
        self.selected = value


class TestTakeScreenshot(TestCase):
    def _get_controller(self, view=None, model=None):
        ctrl = UILocatorController.__new__(UILocatorController)
        ctrl.view = view or FakeView()
        ctrl.model = model or MagicMock()
        ctrl._minimize = True
        return ctrl

    def test_restores_window_after_successful_capture(self):
        model = MagicMock()
        model.capture_desktop.return_value = 'img'
        view = FakeView()
        ctrl = self._get_controller(view=view, model=model)

        result = ctrl._take_screenshot()

        self.assertEqual(result, 'img')
        self.assertEqual(view.state_calls, ['iconify', 'deiconify', 'state(normal)', 'lift'])
        self.assertEqual(view.after_called[0], 10000)
        self.assertEqual(view.after_cancel_called, 'id')

    def test_restores_window_on_capture_failure(self):
        model = MagicMock()
        model.capture_desktop.side_effect = RuntimeError('boom')
        view = FakeView()
        ctrl = self._get_controller(view=view, model=model)

        with self.assertRaises(RuntimeError):
            ctrl._take_screenshot()

        self.assertEqual(view.state_calls, ['iconify', 'deiconify', 'state(normal)', 'lift'])
        self.assertEqual(view.after_cancel_called, 'id')

    def test_minimise_failure_raises_runtime_error(self):
        view = FakeView(fail_minimise=True)
        ctrl = self._get_controller(view=view, model=MagicMock())

        with self.assertRaises(RuntimeError):
            ctrl._take_screenshot()

        self.assertIsNone(view.after_called)
        self.assertIsNone(view.after_cancel_called)

    def test_restores_zoomed_state(self):
        model = MagicMock()
        model.capture_desktop.return_value = 'img'
        view = FakeView(initial_state='zoomed')
        ctrl = self._get_controller(view=view, model=model)

        ctrl._take_screenshot()

        self.assertIn('state(zoomed)', view.state_calls)

    def test_capture_without_minimise_masks_window(self):
        model = MagicMock()
        model.capture_desktop.return_value = 'img'
        view = FakeView()
        ctrl = self._get_controller(view=view, model=model)
        ctrl._minimize = False
        with patch('PIL.ImageDraw.Draw') as mock_draw:
            mock_draw.return_value = MagicMock()
            result = ctrl._take_screenshot()
        self.assertEqual(result, 'img')
        self.assertEqual(view.state_calls, [])
        mock_draw.assert_called_once_with('img')
        mock_draw.return_value.rectangle.assert_called_once()


class TestLoadNeedleImageNames(TestCase):
    def _controller(self, reference_folder):
        ctrl = UILocatorController.__new__(UILocatorController)
        ctrl.image_horizon_instance = SimpleNamespace(reference_folder=reference_folder)
        ctrl.view = SimpleNamespace()
        return ctrl

    def test_handles_missing_reference_folder(self):
        ctrl = self._controller(reference_folder=None)
        combo = DummyCombobox()

        ctrl.load_needle_image_names(combo)

        self.assertEqual(combo.values, [])
        self.assertEqual(
            combo.selected,
            "__ __ __ Select a reference image __ __ __",
        )

    def test_populates_valid_directory(self):
        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "first.png").touch()
            Path(tmpdir, "second.png").touch()
            ctrl = self._controller(reference_folder=tmpdir)
            combo = DummyCombobox()

            ctrl.load_needle_image_names(combo)

            self.assertCountEqual(combo.values, ["first.png", "second.png"])
            self.assertEqual(
                combo.selected,
                "__ __ __ Select a reference image __ __ __",
            )
