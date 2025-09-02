from unittest import TestCase
from unittest.mock import MagicMock, patch
from os.path import dirname, join as path_join
from types import SimpleNamespace
from PIL import Image

CURDIR = dirname(__file__)
TESTIMG_DIR = path_join(CURDIR, 'reference_images')


class TestLocateKeywordStrategies(TestCase):
    def setUp(self):
        ref_path = path_join(TESTIMG_DIR, 'my_picture.png')
        ref_img = Image.open(ref_path)
        self.haystack_img = Image.new('RGB', (ref_img.width + 10, ref_img.height + 10), 'white')
        self.haystack_img.paste(ref_img, (2, 3))
        self.location = (2, 3, ref_img.width, ref_img.height)
        self.expected_x = self.location[0] + self.location[2] / 2
        self.expected_y = self.location[1] + self.location[3] / 2

    def _mock_pyautogui(self):
        mock = MagicMock()
        mock.locate.return_value = self.location
        mock.center.side_effect = lambda box: SimpleNamespace(
            x=box[0] + box[2] / 2, y=box[1] + box[3] / 2
        )
        mock.screenshot.side_effect = lambda: self.haystack_img
        return mock

    def _import_library(self, mock_pyautogui):
        with patch.dict('sys.modules', {'pyautogui': mock_pyautogui}):
            import sys, importlib

            sys.modules.pop('ImageHorizonLibrary', None)
            return importlib.import_module('ImageHorizonLibrary')

    def test_locate_returns_values_with_default_strategy(self):
        lib_module = self._import_library(self._mock_pyautogui())
        lib = lib_module.ImageHorizonLibrary(reference_folder=TESTIMG_DIR)
        x, y, score, scale = lib.locate('my_picture.png')
        self.assertAlmostEqual(x, self.expected_x)
        self.assertAlmostEqual(y, self.expected_y)
        self.assertIsInstance(score, (float, type(None)))
        self.assertEqual(scale, 1.0)

    def test_locate_returns_values_with_edge_strategy(self):
        lib_module = self._import_library(self._mock_pyautogui())
        lib = lib_module.ImageHorizonLibrary(reference_folder=TESTIMG_DIR)
        lib.set_strategy('edge')
        lib._try_locate = lambda ref_image: (self.location, 0.9, 1.0)
        x, y, score, scale = lib.locate('my_picture.png')
        self.assertAlmostEqual(x, self.expected_x)
        self.assertAlmostEqual(y, self.expected_y)
        self.assertEqual(score, 0.9)
        self.assertEqual(scale, 1.0)

    def test_auto_edge_parameters_returns_scalars(self):
        from unittest.mock import MagicMock, patch
        import numpy as np

        fake_cv2 = MagicMock()
        fake_cv2.threshold.return_value = (127, None)
        fake_cv2.THRESH_BINARY = 0
        fake_cv2.THRESH_OTSU = 0

        with patch.dict(
            "sys.modules", {"pyautogui": MagicMock(), "cv2": fake_cv2}
        ):
            from ImageHorizonLibrary.recognition._recognize_images import _StrategyCv2

            class DummyIH:
                pass

            strat = _StrategyCv2(DummyIH())
            img = np.zeros((5, 5), dtype=np.uint8)
            sigma, low, high = strat._auto_edge_parameters(img)
        self.assertIsInstance(sigma, float)
        self.assertIsInstance(low, float)
        self.assertIsInstance(high, float)
