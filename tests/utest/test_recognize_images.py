# -*- coding: utf-8 -*-
import time

from unittest import TestCase, skip
from os.path import abspath, dirname, join as path_join
from unittest.mock import call, MagicMock, patch, ANY
import cv2
import numpy as np

CURDIR = abspath(dirname(__file__))
TESTIMG_DIR = path_join(CURDIR, 'reference_images')

class TestRecognizeImages(TestCase):
    def setUp(self):
        self.mock = MagicMock()
        self.patcher = patch.dict('sys.modules', {'pyautogui': self.mock})
        self.patcher.start()
        from ImageHorizonLibrary import ImageHorizonLibrary
        self.lib = ImageHorizonLibrary(reference_folder=TESTIMG_DIR)
        self.locate = 'ImageHorizonLibrary.ImageHorizonLibrary.locate'
        self._locate = 'ImageHorizonLibrary.ImageHorizonLibrary._locate'

    def tearDown(self):
        self.mock.reset_mock()
        self.patcher.stop()

    def test_find_with_confidence(self):
        self.lib.reference_folder = path_join(CURDIR, 'symbolic_link')
        self.lib.set_confidence(0.5)
        self.lib.has_cv = True
        self.lib.locate('mY_PiCtURe')
        expected_path = path_join(CURDIR, 'symbolic_link', 'my_picture.png')
        # haystack image can be anything
        self.mock.locate.assert_called_once_with(expected_path, ANY, confidence=0.5)
        self.mock.reset_mock()

    def test_find_with_confidence_no_opencv(self):
        self.lib.reference_folder = path_join(CURDIR, 'symbolic_link')
        self.lib.set_confidence(0.5)
        self.lib.has_cv = False
        self.lib.locate('mY_PiCtURe')
        expected_path = path_join(CURDIR, 'symbolic_link', 'my_picture.png')
        self.mock.locate.assert_called_once_with(expected_path, ANY)
        self.mock.reset_mock()

    def test_click_image(self):
        with patch(self._locate, return_value=(0, 0, 1.0, 1.0)):
            self.lib.click_image('my_picture')
            self.mock.click.assert_called_once_with((0, 0))

    def _call_all_directional_functions(self, fn_name):
        from ImageHorizonLibrary import ImageHorizonLibrary
        retvals = []
        for direction in ['above', 'below', 'left', 'right']:
            fn = getattr(self.lib, fn_name % direction)
            with patch(self._locate, return_value=(0, 0, 1.0, 1.0)):
                retvals.append(fn('my_picture', 10))
        return retvals

    def _verify_calls_to_pyautogui(self, mock_calls, clicks=1):
        self.assertEqual(
            mock_calls,
            [call(0, -10, button='left', interval=0.0, clicks=clicks),
             call(0, 10, button='left', interval=0.0, clicks=clicks),
             call(-10, 0, button='left', interval=0.0, clicks=clicks),
             call(10, 0, button='left', interval=0.0, clicks=clicks)])

    def test_directional_clicks(self):
        self._call_all_directional_functions('click_to_the_%s_of_image')
        self._verify_calls_to_pyautogui(self.mock.click.mock_calls)

    def test_directional_copies(self):
        copy = 'ImageHorizonLibrary.ImageHorizonLibrary.copy'
        with patch(copy, return_value='Some Text'):
            ret = self._call_all_directional_functions('copy_from_the_%s_of')
        self._verify_calls_to_pyautogui(self.mock.click.mock_calls, clicks=3)
        for retval in ret:
            self.assertEqual(retval, 'Some Text')

    def test_does_exist(self):
        from ImageHorizonLibrary import ImageNotFoundException

        with patch(self._locate, return_value=(0, 0, 1.0, 1.0)):
            self.assertTrue(self.lib.does_exist('my_picture'))

        run_on_failure = MagicMock()
        with patch(self._locate, side_effect=ImageNotFoundException('')), \
             patch.object(self.lib, '_run_on_failure', run_on_failure):
            self.assertFalse(self.lib.does_exist('my_picture'))
            self.assertEqual(len(run_on_failure.mock_calls), 0)

    def test_does_exist_when_locate_returns_array(self):
        with patch(self._locate, return_value=np.array([0, 0, 10, 10])):
            self.assertTrue(self.lib.does_exist('my_picture'))

    def test_wait_for_happy_path(self):
        from ImageHorizonLibrary import InvalidImageException
        run_on_failure = MagicMock()

        with patch(self._locate, return_value=(0, 0, 1.0, 1.0)), \
             patch.object(self.lib, '_run_on_failure', run_on_failure):
            self.lib.wait_for('my_picture', timeout=1)
            self.assertEqual(len(run_on_failure.mock_calls), 0)

    def test_wait_for_negative_path(self):
        from ImageHorizonLibrary import InvalidImageException
        run_on_failure = MagicMock()

        with self.assertRaises(InvalidImageException), \
             patch(self._locate, side_effect=InvalidImageException('')), \
             patch.object(self.lib, '_run_on_failure', run_on_failure):

            start = time.time()
            self.lib.wait_for('notfound', timeout='1')
            stop = time.time()

            run_on_failure.assert_called_once_with()
            # check that timeout given as string works and it does not use
            # default timeout
            self.assertLess(stop-start, 10)

    def _verify_path_works(self, image_name, expected):
        self.lib.locate(image_name)
        expected_path = path_join(TESTIMG_DIR, expected)
        self.mock.locate.assert_called_once_with(expected_path, ANY)
        self.mock.reset_mock()

    def test_locate(self):
        from ImageHorizonLibrary import InvalidImageException

        for image_name in ('my_picture.png', 'my picture.png', 'MY PICTURE',
                           'mY_PiCtURe'):
            self._verify_path_works(image_name, 'my_picture.png')

        self.mock.locate.return_value = None
        run_on_failure = MagicMock()
        with self.assertRaises(InvalidImageException), \
             patch.object(self.lib, '_run_on_failure', run_on_failure):
            self.lib.locate('nonexistent')
            run_on_failure.assert_called_once_with()

    def test_locate_with_valid_reference_folder(self):
        for ref, img in (('reference_images', 'my_picture.png'),
                         ('./reference_images', 'my picture.png'),
                         ('../../tests/utest/reference_images', 'MY PICTURE')):

            self.lib.set_reference_folder(path_join(CURDIR, ref))
            self._verify_path_works(img, 'my_picture.png')

        self.lib.reference_folder = path_join(CURDIR, 'symbolic_link')
        self.lib.locate('mY_PiCtURe')
        expected_path = path_join(CURDIR, 'symbolic_link', 'my_picture.png')
        self.mock.locate.assert_called_once_with(expected_path, ANY)
        self.mock.reset_mock()

        self.lib.reference_folder = path_join(CURDIR, 'rëförence_imägës')
        self.lib.locate('mŸ PäKSÖR')
        expected_path = path_join(CURDIR, 'rëförence_imägës',
                                  'mÿ_päksör.png')
        self.mock.locate.assert_called_once_with(expected_path, ANY)
        self.mock.reset_mock()

    def test_locate_with_invalid_reference_folder(self):
        from ImageHorizonLibrary import ReferenceFolderException

        for invalid_folder in (None, 123, 'nonexistent', 'nönëxistänt'):
            self.lib.reference_folder = invalid_folder
            with self.assertRaises(ReferenceFolderException):
                self.lib.locate('my_picture')

        if not self.lib.is_windows:
            self.lib.reference_folder = TESTIMG_DIR.replace('/', '\\')
            with self.assertRaises(ReferenceFolderException):
                self.lib.locate('my_picture')

    def test_locate_with_invalid_image_name(self):
        from ImageHorizonLibrary import InvalidImageException

        for invalid_image_name in (None, 123, 1.2, True, self.lib.__class__()):
            with self.assertRaises(InvalidImageException):
                self.lib.locate(invalid_image_name)

    def test_strategy_locate_returns_score(self):
        from PIL import Image
        from contextlib import contextmanager
        from ImageHorizonLibrary.recognition import _recognize_images as rec
        from ImageHorizonLibrary.recognition._recognize_images import _StrategyPyautogui

        # create a haystack image containing the reference picture
        ref_path = path_join(TESTIMG_DIR, 'my_picture.png')
        ref_img = Image.open(ref_path)
        haystack_img = Image.new('RGB', (ref_img.width + 10, ref_img.height + 10), 'white')
        haystack_img.paste(ref_img, (2, 3))

        # pyautogui.locate is mocked in setUp; define its return value
        self.mock.locate.return_value = (2, 3, ref_img.width, ref_img.height)

        @contextmanager
        def no_suppress():
            yield

        class DummyIH:
            confidence = 0.5
            has_cv = True
            scale_enabled = False
            scale_min = 0.8
            scale_max = 1.2
            scale_steps = 9

            def _suppress_keyword_on_failure(self):
                return no_suppress()

        rec.cv2 = cv2
        strat = _StrategyPyautogui(DummyIH())
        location, score, scale = strat._try_locate(ref_path, haystack_image=haystack_img)

        self.assertEqual(location, (2, 3, ref_img.width, ref_img.height))
        self.assertIsNotNone(score)
        self.assertGreater(score, 0.9)
        self.assertEqual(scale, 1.0)

    def test_strategy_locate_handles_scaling(self):
        from PIL import Image
        from contextlib import contextmanager
        from ImageHorizonLibrary.recognition import _recognize_images as rec
        from ImageHorizonLibrary.recognition._recognize_images import _StrategyPyautogui

        ref_path = path_join(TESTIMG_DIR, 'my_picture.png')
        ref_img = Image.open(ref_path)
        scale_factor = 1.2
        scaled_ref = ref_img.resize((int(ref_img.width * scale_factor), int(ref_img.height * scale_factor)))
        haystack_img = Image.new('RGB', (scaled_ref.width + 10, scaled_ref.height + 10), 'white')
        haystack_img.paste(scaled_ref, (4, 5))

        @contextmanager
        def no_suppress():
            yield

        class DummyIH:
            confidence = 0.8
            has_cv = True
            scale_enabled = True
            scale_min = 0.8
            scale_max = 1.4
            scale_steps = 7

            def _suppress_keyword_on_failure(self):
                return no_suppress()

        rec.cv2 = cv2
        strat = _StrategyPyautogui(DummyIH())
        location, score, scale = strat._try_locate(ref_path, haystack_image=haystack_img)

        self.assertEqual(location, (4, 5, scaled_ref.width, scaled_ref.height))
        self.assertIsNotNone(score)
        self.assertGreater(score, 0.9)
        self.assertAlmostEqual(scale, scale_factor, delta=0.05)

class TestEdgeDetection(TestCase):
    def test_high_threshold_affects_edge_detection(self):
        from unittest.mock import MagicMock, patch
        import numpy as np

        with patch.dict("sys.modules", {"pyautogui": MagicMock()}):
            from ImageHorizonLibrary.recognition._recognize_images import _StrategyCv2

            class DummyIH:
                edge_sigma = 1
                edge_low_threshold = 0.1
                edge_high_threshold = 0.2
                edge_preprocess = None
                edge_kernel_size = 3
                has_cv = True
                scale_enabled = False
                scale_min = 0.8
                scale_max = 1.2
                scale_steps = 9

            ih = DummyIH()
            strategy = _StrategyCv2(ih)

            with patch.object(
                _StrategyCv2,
                "_detect_edges",
                autospec=True,
                side_effect=lambda self, img, sigma, low, high: img * high,
            ):
                img = np.array(
                    [
                        [0, 0.1, 0.2, 0.3, 0.4],
                        [0, 0.1, 0.2, 0.3, 0.4],
                        [0, 0.1, 0.2, 0.3, 0.4],
                        [0, 0.1, 0.2, 0.3, 0.4],
                        [0, 0.1, 0.2, 0.3, 0.4],
                    ],
                    dtype=float,
                )

                edges_low_high = strategy.detect_edges(img)

                ih.edge_high_threshold = 0.9
                edges_high_high = strategy.detect_edges(img)

        self.assertFalse(np.array_equal(edges_low_high, edges_high_high))


class TestMultiScaleSearch(TestCase):
    def test_finds_scaled_reference(self):
        from contextlib import contextmanager
        from unittest.mock import MagicMock, patch
        import numpy as np

        scale_factor = 1.2
        ref_path = path_join(TESTIMG_DIR, "my_picture.png")
        needle = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        scaled_ref = cv2.resize(
            needle,
            (int(needle.shape[1] * scale_factor), int(needle.shape[0] * scale_factor)),
        )
        haystack_img = np.full(
            (scaled_ref.shape[0] + 10, scaled_ref.shape[1] + 10),
            255,
            dtype=np.uint8,
        )
        haystack_img[5 : 5 + scaled_ref.shape[0], 4 : 4 + scaled_ref.shape[1]] = scaled_ref

        with patch.dict("sys.modules", {"pyautogui": MagicMock()}):
            from ImageHorizonLibrary.recognition._recognize_images import _StrategyCv2

        class DummyIH:
            confidence = 0.8
            edge_sigma = 1
            edge_low_threshold = 0.1
            edge_high_threshold = 0.3
            edge_preprocess = None
            edge_kernel_size = 3
            has_cv = True
            validate_match = False
            validation_margin = 0
            scale_enabled = True
            scale_min = 0.8
            scale_max = 1.2
            scale_steps = 9

            @contextmanager
            def _suppress_keyword_on_failure(self):
                yield None

        strategy = _StrategyCv2(DummyIH())

        location, score, scale = strategy._try_locate(ref_path, haystack_image=haystack_img)
        self.assertIsNotNone(location)
        self.assertEqual(
            location,
            (4, 5, scaled_ref.shape[1], scaled_ref.shape[0]),
        )
        self.assertAlmostEqual(scale, scale_factor, delta=0.05)
        self.assertGreater(score, 0.8)


class TestPreprocessingAndValidation(TestCase):
    def test_gaussian_preprocess_calls_cv2(self):
        fake_cv2 = MagicMock()
        fake_cv2.GaussianBlur.return_value = np.zeros((5, 5), dtype=np.uint8)
        fake_cv2.Canny.return_value = np.zeros((5, 5), dtype=np.uint8)

        with patch.dict("sys.modules", {"pyautogui": MagicMock()}):
            from ImageHorizonLibrary.recognition import _recognize_images
            from ImageHorizonLibrary.recognition._recognize_images import _StrategyCv2

        with patch.object(_recognize_images, "cv2", fake_cv2):
            class DummyIH:
                edge_sigma = 0
                edge_low_threshold = 0.1
                edge_high_threshold = 0.2
                edge_preprocess = "gaussian"
                edge_kernel_size = 3
                has_cv = True

            ih = DummyIH()
            strategy = _StrategyCv2(ih)
            img = np.zeros((5, 5), dtype=np.uint8)
            strategy._detect_edges(
                img, ih.edge_sigma, ih.edge_low_threshold, ih.edge_high_threshold
            )
            fake_cv2.GaussianBlur.assert_called_once_with(ANY, (3, 3), 0)

    def test_match_validation_uses_opencv(self):
        from unittest.mock import MagicMock, patch
        import numpy as np
        fake_cv2 = MagicMock()
        fake_cv2.TM_CCOEFF_NORMED = 0
        fake_cv2.Canny.return_value = np.zeros((1, 1), dtype=np.uint8)
        class DummyRes:
            size = 1
            def max(self):
                return 1.0
        fake_cv2.matchTemplate.return_value = DummyRes()
        fake_cv2.minMaxLoc.return_value = (0, 1.0, (0, 0), (5, 5))
        with patch.dict("sys.modules", {"pyautogui": MagicMock(), "cv2": fake_cv2}):
            from ImageHorizonLibrary.recognition._recognize_images import _StrategyCv2
            from contextlib import contextmanager

            class DummyIH:
                confidence = 0.1
                edge_sigma = 1
                edge_low_threshold = 0.1
                edge_high_threshold = 0.3
                edge_preprocess = None
                edge_kernel_size = 3
                has_cv = True
                validate_match = True
                validation_margin = 0
                scale_enabled = False
                scale_min = 0.8
                scale_max = 1.2
                scale_steps = 9

                @contextmanager
                def _suppress_keyword_on_failure(self):
                    yield None

            ih = DummyIH()
            strategy = _StrategyCv2(ih)
            haystack = np.ones((30, 30))
            needle = np.ones((10, 10))
            peakmap = np.zeros((20, 20))
            peakmap[5, 5] = 1.0
            fake_cv2.imread.return_value = needle
            fake_cv2.resize.return_value = needle
            with patch.object(strategy, "detect_edges", return_value=needle):
                strategy._try_locate("dummy", haystack_image=haystack)
            assert fake_cv2.matchTemplate.called
