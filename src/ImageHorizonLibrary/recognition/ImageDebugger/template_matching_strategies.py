"""Template matching strategies used by the debugger UI."""

from PIL import ImageDraw
from abc import ABC, abstractmethod
from .image_manipulation import ImageFormat, ImageContainer


class TemplateMatchingStrategy(ABC):
    """Base class for template matching strategies."""

    @abstractmethod
    def find_num_of_matches(self) -> int:
        """Return all coordinates where the template was matched."""

    def highlight_matches(self):
        """Draw rectangles around matched regions in the haystack image."""
        haystack_img = self.image_container.get_haystack_image_orig_size(ImageFormat.PILIMG)
        draw = ImageDraw.Draw(haystack_img, "RGBA")

        for loc in self.coord:
            draw.rectangle([loc[0], loc[1], loc[0]+loc[2], loc[1]+loc[3]], fill=(256, 0, 0, 127))
            draw.rectangle([loc[0], loc[1], loc[0]+loc[2], loc[1]+loc[3]], outline=(256, 0, 0, 127), width=3)

        self.image_container.save_to_img_container(img=haystack_img, is_haystack_img=True)

class Pyautogui(TemplateMatchingStrategy):
    """Strategy using PyAutoGUI for template matching."""

    def __init__(self, image_container: ImageContainer, image_horizon_instance):
        """Store references to images and the ImageHorizonLibrary instance."""
        self.image_container = image_container
        self.image_horizon_instance = image_horizon_instance

    def find_num_of_matches(self):
        """Return coordinates of all matches using PyAutoGUI."""
        haystack_image = self.image_container.get_haystack_image_orig_size(
            ImageFormat.PILIMG
        )
        matches = list(
            self.image_horizon_instance._locate_all(
                self.image_container.get_needle_image(ImageFormat.PATHSTR),
                haystack_image,
            )
        )
        # Each match is ``(location, score, scale)``; keep only location and
        # determine the best score among all matches.
        self.coord = [loc for loc, _, _ in matches]
        self.best_score = max(
            (score for _, score, _ in matches if score is not None),
            default=None,
        )
        try:
            _, best_score, _ = self.image_horizon_instance._try_locate(
                self.image_container.get_needle_image(ImageFormat.PATHSTR),
                haystack_image=haystack_image,
            )
        except Exception:
            best_score = None
        if best_score is not None:
            self.best_score = best_score
        return self.coord


class Cv2(TemplateMatchingStrategy):
    """Strategy using OpenCV for template matching."""

    def __init__(self, image_container: ImageContainer, image_horizon_instance):
        """Store references to images and the ImageHorizonLibrary instance."""
        self.image_container = image_container
        self.image_horizon_instance = image_horizon_instance

    def find_num_of_matches(self):
        """Return coordinates of all matches using OpenCV."""
        haystack_image = self.image_container.get_haystack_image_orig_size(
            ImageFormat.NUMPYARRAY
        )
        matches = list(
            self.image_horizon_instance._locate_all(
                self.image_container.get_needle_image(ImageFormat.PATHSTR),
                haystack_image,
            )
        )
        # Keep only location information for highlighting/plotting and store
        # the best score for later display.
        self.coord = [loc for loc, _, _ in matches]
        self.best_score = max(
            (score for _, score, _ in matches if score is not None),
            default=None,
        )
        try:
            _, best_score, _ = self.image_horizon_instance._try_locate(
                self.image_container.get_needle_image(ImageFormat.PATHSTR),
                haystack_image=haystack_image,
            )
        except Exception:
            best_score = None
        if best_score is not None:
            self.best_score = best_score
        return self.coord
