from torch import Tensor

from .tensor_base import TensorBase


class Polygon(TensorBase):  # B, N, M, 2, quadrilateral when m=4, keypoint when m=1. Need to pad both N, M channel.

    def transform(self, mat: Tensor) -> "Polygon":
        """Apply a transformation matrix.

        To transform any (B, N, M, 2) data.
        """
        raise NotImplementedError

    def to_mask(self, height: int, width: int) -> Tensor:
        """Convert the covered area into a binary mask.

        Returns (B, N, H, W)
        """
        raise NotImplementedError

    def validate(self) -> None:
        """Validate data."""
