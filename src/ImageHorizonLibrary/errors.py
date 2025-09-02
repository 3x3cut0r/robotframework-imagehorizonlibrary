# -*- coding: utf-8 -*-
class ImageHorizonLibraryError(ImportError):
    pass


class ImageNotFoundException(Exception):
    def __init__(
        self,
        image_name,
        matches=None,
        best_score=None,
        confidence=None,
    ):
        """Describe missing reference image with optional diagnostics.

        Parameters
        ----------
        image_name : str
            Name of the image that was not found.
        matches : int, optional
            Number of matches detected above the confidence threshold. Defaults
            to ``None``.
        best_score : float, optional
            Highest score returned by the matching algorithm.
        confidence : float, optional
            Confidence threshold used for the search.
        """

        self.image_name = image_name
        self.matches = matches
        self.best_score = best_score
        self.confidence = confidence

    def __str__(self):
        msg = 'Reference image "%s" was not found on screen' % self.image_name
        details = []
        if self.matches:
            details.append(f"matches found: {self.matches}")
        if self.best_score is not None and self.confidence is not None:
            details.append(
                f"best score {self.best_score:.2f} (confidence {self.confidence:.2f})"
            )
        elif self.best_score is not None:
            details.append(f"best score {self.best_score:.2f}")
        elif self.confidence is not None:
            details.append(f"confidence {self.confidence:.2f}")
        if details:
            msg += ". " + ", ".join(details)
        return msg


class InvalidImageException(Exception):
    pass


class KeyboardException(Exception):
    pass


class MouseException(Exception):
    pass


class OSException(Exception):
    pass


class ReferenceFolderException(Exception):
    pass


class ScreenshotFolderException(Exception):
    pass

class StrategyException(Exception):
    pass
