from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.geometry.transform import vflip


class RandomVerticalFlip(GeometricAugmentationBase2D):
    r"""Apply a random vertical flip to a tensor image or a batch of tensor images with a given probability.

    .. image:: _static/img/RandomVerticalFlip.png

    Args:
        p: probability of the image being flipped.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.geometry.transform.vflip`.

    Examples:
        >>> input = torch.tensor([[[[0., 0., 0.],
        ...                         [0., 0., 0.],
        ...                         [0., 1., 1.]]]])
        >>> seq = RandomVerticalFlip(p=1.0)
        >>> seq(input), seq.transform_matrix
        (tensor([[[[0., 1., 1.],
                  [0., 0., 0.],
                  [0., 0., 0.]]]]), tensor([[[ 1.,  0.,  0.],
                 [ 0., -1.,  2.],
                 [ 0.,  0.,  1.]]]))
        >>> seq.inverse(seq(input)).equal(input)
        True

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> seq = RandomVerticalFlip(p=1.0)
        >>> (seq(input) == seq(input, params=seq._params)).all()
        tensor(True)
    """

    def compute_transformation(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any]) -> Tensor:
        h: int = int(params["forward_input_shape"][-2])
        flip_mat: Tensor = torch.tensor([[1, 0, 0], [0, -1, h - 1], [0, 0, 1]], device=input.device, dtype=input.dtype)

        return flip_mat.repeat(input.size(0), 1, 1)

    def apply_transform(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None
    ) -> Tensor:
        return vflip(input)

    def inverse_transform(
        self, input: Tensor, flags: dict[str, Any], transform: Tensor | None = None, size: tuple[int, int] | None = None
    ) -> Tensor:
        return self.apply_transform(
            input,
            params=self._params,
            transform=torch.as_tensor(transform, device=input.device, dtype=input.dtype),
            flags=flags,
        )
