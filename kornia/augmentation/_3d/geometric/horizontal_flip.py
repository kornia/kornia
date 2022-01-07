from typing import Dict, Optional

import torch

from kornia.augmentation._3d.base import AugmentationBase3D


class RandomHorizontalFlip3D(AugmentationBase3D):
    r"""Apply random horizontal flip to 3D volumes (5D tensor).

    Args:
        p: probability of the image being flipped.
        return_transform: if ``True`` return the matrix describing the transformation applied to each
          input tensor. If ``False`` and the input is a tuple the applied transformation
          won't be concatenated.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.

    Shape:
        - Input: :math:`(C, D, H, W)` or :math:`(B, C, D, H, W)`, Optional: :math:`(B, 4, 4)`
        - Output: :math:`(B, C, D, H, W)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 4, 4)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> x = torch.eye(3).repeat(3, 1, 1)
        >>> seq = RandomHorizontalFlip3D(p=1.0, return_transform=True)
        >>> seq(x)
        (tensor([[[[[0., 0., 1.],
                   [0., 1., 0.],
                   [1., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 1.],
                   [0., 1., 0.],
                   [1., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 1.],
                   [0., 1., 0.],
                   [1., 0., 0.]]]]]), tensor([[[-1.,  0.,  0.,  2.],
                 [ 0.,  1.,  0.,  0.],
                 [ 0.,  0.,  1.,  0.],
                 [ 0.,  0.,  0.,  1.]]]))

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.rand(1, 3, 32, 32, 32)
        >>> aug = RandomHorizontalFlip3D(p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self, return_transform: bool = False, same_on_batch: bool = False, p: float = 0.5, keepdim: bool = False
    ) -> None:
        super().__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch, keepdim=keepdim)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        w: int = input.shape[-1]
        flip_mat: torch.Tensor = torch.tensor(
            [[-1, 0, 0, w - 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], device=input.device, dtype=input.dtype
        )
        return flip_mat.repeat(input.size(0), 1, 1)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return torch.flip(input, [-1])
