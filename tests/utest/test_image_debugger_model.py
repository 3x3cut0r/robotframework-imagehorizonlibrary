from unittest import TestCase
from unittest.mock import MagicMock, patch


class TestUILocatorModel(TestCase):
    def setUp(self):
        self.mock = MagicMock()
        self.patcher = patch.dict('sys.modules', {'pyautogui': self.mock})
        self.patcher.start()
        from ImageHorizonLibrary.recognition.ImageDebugger.image_debugger_model import UILocatorModel
        self.model_cls = UILocatorModel

    def tearDown(self):
        self.patcher.stop()
        self.mock.reset_mock()

    def test_capture_desktop_failure_logs_and_raises(self):
        model = self.model_cls()
        self.mock.screenshot.side_effect = Exception('boom')
        with patch('ImageHorizonLibrary.recognition.ImageDebugger.image_debugger_model.LOGGER') as logger:
            with self.assertRaises(RuntimeError):
                model.capture_desktop()
            self.mock.screenshot.assert_called_once()
            logger.error.assert_called_once()
