from typing import Any, Dict, Optional

from torch import Tensor

from kornia.augmentation._3d.intensity.base import IntensityAugmentationBase3D
from kornia.enhance import equalize3d


class RandomEqualize3D(IntensityAugmentationBase3D):
    r"""Apply random equalization to 3D volumes (5D tensor).

    Args:
        p: probability of the image being equalized.
        same_on_batch): apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
          to the batch form (False).

    Shape:
        - Input: :math:`(C, D, H, W)` or :math:`(B, C, D, H, W)`, Optional: :math:`(B, 4, 4)`
        - Output: :math:`(B, C, D, H, W)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 4, 4)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> import torch
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 3, 3, 3)
        >>> aug = RandomEqualize3D(p=1.0)
        >>> aug(input)
        tensor([[[[[0.4963, 0.7682, 0.0885],
                   [0.1320, 0.3074, 0.6341],
                   [0.4901, 0.8964, 0.4556]],
        <BLANKLINE>
                  [[0.6323, 0.3489, 0.4017],
                   [0.0223, 0.1689, 0.2939],
                   [0.5185, 0.6977, 0.8000]],
        <BLANKLINE>
                  [[0.1610, 0.2823, 0.6816],
                   [0.9152, 0.3971, 0.8742],
                   [0.4194, 0.5529, 0.9527]]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.rand(1, 3, 32, 32, 32)
        >>> aug = RandomEqualize3D(p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(self, p: float = 0.5, same_on_batch: bool = False, keepdim: bool = False) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        return self.identity_matrix(input)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return equalize3d(input)
