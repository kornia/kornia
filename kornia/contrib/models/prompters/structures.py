from __future__ import annotations

from dataclasses import dataclass

from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK


@dataclass
class Prompts:
    """Encapsulate the prompts inputs for a Model.

    Args:
        points: A tuple with the keypoints (coordinates x, y) and their respective labels. Shape :math:`(K, N, 2)` for
                the keypoints, and :math:`(K, N)`
        boxes: Batched box inputs, with shape :math:`(K, 4)`. Expected to be into xyxy format.
        masks: Batched mask prompts to the model with shape :math:`(K, 1, H, W)`
    """

    points: tuple[Tensor, Tensor] | None = None
    boxes: Tensor | None = None
    masks: Tensor | None = None

    def __post_init__(self) -> None:
        if isinstance(self.keypoints, Tensor) and isinstance(self.boxes, Tensor):
            KORNIA_CHECK(self.keypoints.shape[0] == self.boxes.shape[0], 'The prompts should have the same batch size!')

    @property
    def keypoints(self) -> Tensor | None:
        """The keypoints from the `points`"""
        return self.points[0] if isinstance(self.points, tuple) else None

    @property
    def keypoints_labels(self) -> Tensor | None:
        """The keypoints labels from the `points`"""
        return self.points[1] if isinstance(self.points, tuple) else None
