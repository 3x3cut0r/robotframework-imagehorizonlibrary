import unittest
from PIL import Image
from unittest.mock import patch
import importlib.util
from os.path import abspath, dirname, join as path_join

module_path = path_join(
    abspath(dirname(__file__)),
    '..', '..', 'src',
    'ImageHorizonLibrary', 'recognition', 'ImageDebugger', 'image_manipulation.py'
)
spec = importlib.util.spec_from_file_location('image_manipulation', module_path)
image_manipulation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_manipulation)

ImageContainer = image_manipulation.ImageContainer
RESAMPLE_FILTER = image_manipulation.RESAMPLE_FILTER


class TestImageContainer(unittest.TestCase):
    def test_save_to_img_container_uses_lanczos(self):
        img = Image.new('RGB', (500, 500))
        container = ImageContainer()
        with patch.object(img, 'resize', wraps=img.resize) as resize_mock:
            container.save_to_img_container(img, is_haystack_img=True)
            resize_mock.assert_called_once_with((384, 216), RESAMPLE_FILTER)

