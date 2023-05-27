from typing import Any, Dict, Optional

import torch
from torch import Tensor

from kornia.augmentation._3d.geometric.base import GeometricAugmentationBase3D


class RandomDepthicalFlip3D(GeometricAugmentationBase3D):
    r"""Apply random flip along the depth axis of 3D volumes (5D tensor).

    Input should be a tensor of shape :math:`(C, D, H, W)` or a batch of tensors :math:`(*, C, D, H, W)`.
    If Input is a tuple it is assumed that the first element contains the aforementioned tensors and the second,
    the corresponding transformation matrix that has been applied to them. In this case the module
    will Depthically flip the tensors and concatenate the corresponding transformation matrix to the
    previous one. This is especially useful when using this functionality as part of an ``nn.Sequential`` module.

    Args:
        p: probability of the image being flipped.
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
        >>> import torch
        >>> x = torch.eye(3).repeat(3, 1, 1)
        >>> seq = RandomDepthicalFlip3D(p=1.0)
        >>> seq(x), seq.transform_matrix
        (tensor([[[[[1., 0., 0.],
                   [0., 1., 0.],
                   [0., 0., 1.]],
        <BLANKLINE>
                  [[1., 0., 0.],
                   [0., 1., 0.],
                   [0., 0., 1.]],
        <BLANKLINE>
                  [[1., 0., 0.],
                   [0., 1., 0.],
                   [0., 0., 1.]]]]]), tensor([[[ 1.,  0.,  0.,  0.],
                 [ 0.,  1.,  0.,  0.],
                 [ 0.,  0., -1.,  2.],
                 [ 0.,  0.,  0.,  1.]]]))

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.rand(1, 3, 32, 32, 32)
        >>> aug = RandomDepthicalFlip3D(p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(self, same_on_batch: bool = False, p: float = 0.5, keepdim: bool = False) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        d: int = input.shape[-3]
        flip_mat: Tensor = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, d - 1], [0, 0, 0, 1]], device=input.device, dtype=input.dtype
        )
        return flip_mat.expand(input.shape[0], 4, 4)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return torch.flip(input, [-3])
