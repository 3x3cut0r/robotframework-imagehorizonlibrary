from unittest.mock import MagicMock, patch
from PIL import Image


def test_strategy_pyautogui_handles_pyautogui_image_not_found():
    class DummyIH:
        confidence = None
        has_cv = False
        scale_enabled = False

        def _suppress_keyword_on_failure(self):
            from contextlib import contextmanager

            @contextmanager
            def dummy():
                yield

            return dummy()

    dummy_ih = DummyIH()
    haystack = Image.new("RGB", (10, 10))

    mock_ag = MagicMock()

    class DummyImageNotFoundException(Exception):
        pass

    mock_ag.ImageNotFoundException = DummyImageNotFoundException
    mock_ag.locate.side_effect = DummyImageNotFoundException("missing")

    with patch.dict("sys.modules", {"pyautogui": mock_ag}):
        from ImageHorizonLibrary.recognition import _recognize_images as ri
        strat = ri._StrategyPyautogui(dummy_ih)
        location, score, scale = strat._try_locate("dummy", haystack_image=haystack)

    assert location is None
    assert score is None
    assert scale == 1.0
