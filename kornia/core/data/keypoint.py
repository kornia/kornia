from .polygon import Polygon


class Keypoint(Polygon):  # B, N, 1, 2
    """Keypoint.
    """

    def validate(self) -> None:
        """Validate data.
        """
