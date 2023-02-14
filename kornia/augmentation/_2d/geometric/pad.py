from typing import Any, Dict, Tuple, Union

import torch
from torch import Tensor

from kornia.augmentation._2d.base import AugmentationBase2D


class PadTo(AugmentationBase2D):
    r"""Pad the given sample to a specific size. Always occurs (p=1.0).

    .. image:: _static/img/PadTo.png

    Args:
        size: a tuple of ints in the format (height, width) that give the spatial
            dimensions to pad inputs to.
        pad_mode: the type of padding to perform on the image (valid values
            are those accepted by torch.nn.functional.pad)
        pad_value: fill value for 'constant' padding applied to the image
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`torch.nn.functional.pad`.

    Examples:
        >>> import torch
        >>> img = torch.tensor([[[[0., 0., 0.],
        ...                       [0., 0., 0.],
        ...                       [0., 0., 0.]]]])
        >>> pad = PadTo((4, 5), pad_value=1.)
        >>> out = pad(img)
        >>> out
        tensor([[[[0., 0., 0., 1., 1.],
                  [0., 0., 0., 1., 1.],
                  [0., 0., 0., 1., 1.],
                  [1., 1., 1., 1., 1.]]]])
        >>> pad.inverse(out)
        tensor([[[[0., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.]]]])
    """

    def __init__(
        self, size: Tuple[int, int], pad_mode: str = "constant", pad_value: Union[int, float] = 0, keepdim: bool = False
    ) -> None:
        super().__init__(p=1.0, same_on_batch=True, p_batch=1.0, keepdim=keepdim)
        self.flags = dict(size=size, pad_mode=pad_mode, pad_value=pad_value)

    def apply_transform(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        _, _, height, width = input.shape
        height_pad: int = flags["size"][0] - height
        width_pad: int = flags["size"][1] - width
        return torch.nn.functional.pad(
            input, [0, width_pad, 0, height_pad], mode=flags["pad_mode"], value=flags["pad_value"]
        )

    def inverse_transform(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        size = params['forward_input_shape'].numpy().tolist()
        size = (size[-2], size[-1])

        return input[..., : size[0], : size[1]]
