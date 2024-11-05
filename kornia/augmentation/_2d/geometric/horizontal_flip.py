from typing import Any, Dict, Optional, Tuple

from torch import Tensor

from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.core import as_tensor, tensor
from kornia.geometry.transform import hflip


class RandomHorizontalFlip(GeometricAugmentationBase2D):
    r"""Apply a random horizontal flip to a tensor image or a batch of tensor images with a given probability.

    .. image:: _static/img/RandomHorizontalFlip.png

    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(B, C, H, W)`.
    If Input is a tuple it is assumed that the first element contains the aforementioned tensors and the second,
    the corresponding transformation matrix that has been applied to them. In this case the module
    will Horizontally flip the tensors and concatenate the corresponding transformation matrix to the
    previous one. This is especially useful when using this functionality as part of an ``nn.Sequential`` module.

    Args:
        p: probability of the image being flipped.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.geometry.transform.hflip`.

    Examples:
        >>> import torch
        >>> input = torch.tensor([[[[0., 0., 0.],
        ...                         [0., 0., 0.],
        ...                         [0., 1., 1.]]]])
        >>> seq = RandomHorizontalFlip(p=1.0)
        >>> seq(input), seq.transform_matrix
        (tensor([[[[0., 0., 0.],
                  [0., 0., 0.],
                  [1., 1., 0.]]]]), tensor([[[-1.,  0.,  2.],
                 [ 0.,  1.,  0.],
                 [ 0.,  0.,  1.]]]))
        >>> seq.inverse(seq(input)).equal(input)
        True

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> seq = RandomHorizontalFlip(p=1.0)
        >>> (seq(input) == seq(input, params=seq._params)).all()
        tensor(True)
    """

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        w: int = int(params["forward_input_shape"][-1])
        flip_mat: Tensor = tensor([[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]], device=input.device, dtype=input.dtype)

        return flip_mat.expand(input.shape[0], 3, 3)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return hflip(input)

    def inverse_transform(
        self,
        input: Tensor,
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        if not isinstance(transform, Tensor):
            raise TypeError(f"Expected the `transform` be a Tensor. Got {type(transform)}.")
        return self.apply_transform(
            input,
            params=self._params,
            transform=as_tensor(transform, device=input.device, dtype=input.dtype),
            flags=flags,
        )
