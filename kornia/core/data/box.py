from .quadrilateral import Quadrilateral


class Box(Quadrilateral):  # B, N, 4, 2 and orthogonal

    def validate(self) -> None:
        """Validate the box.

        If quadrilateral, if orthogonal.
        """
