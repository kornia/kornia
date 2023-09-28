from abc import ABC
from typing import List, Optional

import torch
from torch.nn.functional import interpolate

from kornia.core import Module, Tensor, tensor
from kornia.core.check import KORNIA_CHECK_IS_GRAY

RGBColor = List[float]


def _list_color_to_tensor(
    colors: List[RGBColor], dtype: Optional[torch.dtype], device: Optional[torch.device]
) -> Tensor:
    return tensor(list(colors), dtype=dtype, device=device).T


class ColorMap(ABC):
    r"""Abstract class to represents a color map."""
    colors: Tensor

    def __init__(
        self,
        base_colormap: List[RGBColor],
        num_colors: int,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
    ) -> None:
        self._dtype = dtype
        self._device = device

        self.colors = interpolate(
            _list_color_to_tensor(base_colormap, dtype=self._dtype, device=self._device)[None, ...],
            size=num_colors,
            mode='linear',
        )[0, ...]

    def __len__(self) -> int:
        return self.colors.shape[-1]


# TODO: Add more colormaps
_BASE_AUTUMN: List[RGBColor] = [
    [1.0, 0.0, 0.0],
    [1.0, 0.01587301587301587, 0.0],
    [1.0, 0.03174603174603174, 0.0],
    [1.0, 0.04761904761904762, 0.0],
    [1.0, 0.06349206349206349, 0.0],
    [1.0, 0.07936507936507936, 0.0],
    [1.0, 0.09523809523809523, 0.0],
    [1.0, 0.1111111111111111, 0.0],
    [1.0, 0.126984126984127, 0.0],
    [1.0, 0.1428571428571428, 0.0],
    [1.0, 0.1587301587301587, 0.0],
    [1.0, 0.1746031746031746, 0.0],
    [1.0, 0.1904761904761905, 0.0],
    [1.0, 0.2063492063492063, 0.0],
    [1.0, 0.2222222222222222, 0.0],
    [1.0, 0.2380952380952381, 0.0],
    [1.0, 0.253968253968254, 0.0],
    [1.0, 0.2698412698412698, 0.0],
    [1.0, 0.2857142857142857, 0.0],
    [1.0, 0.3015873015873016, 0.0],
    [1.0, 0.3174603174603174, 0.0],
    [1.0, 0.3333333333333333, 0.0],
    [1.0, 0.3492063492063492, 0.0],
    [1.0, 0.3650793650793651, 0.0],
    [1.0, 0.3809523809523809, 0.0],
    [1.0, 0.3968253968253968, 0.0],
    [1.0, 0.4126984126984127, 0.0],
    [1.0, 0.4285714285714285, 0.0],
    [1.0, 0.4444444444444444, 0.0],
    [1.0, 0.4603174603174603, 0.0],
    [1.0, 0.4761904761904762, 0.0],
    [1.0, 0.492063492063492, 0.0],
    [1.0, 0.5079365079365079, 0.0],
    [1.0, 0.5238095238095238, 0.0],
    [1.0, 0.5396825396825397, 0.0],
    [1.0, 0.5555555555555556, 0.0],
    [1.0, 0.5714285714285714, 0.0],
    [1.0, 0.5873015873015873, 0.0],
    [1.0, 0.6031746031746031, 0.0],
    [1.0, 0.6190476190476191, 0.0],
    [1.0, 0.6349206349206349, 0.0],
    [1.0, 0.6507936507936508, 0.0],
    [1.0, 0.6666666666666666, 0.0],
    [1.0, 0.6825396825396826, 0.0],
    [1.0, 0.6984126984126984, 0.0],
    [1.0, 0.7142857142857143, 0.0],
    [1.0, 0.7301587301587301, 0.0],
    [1.0, 0.746031746031746, 0.0],
    [1.0, 0.7619047619047619, 0.0],
    [1.0, 0.7777777777777778, 0.0],
    [1.0, 0.7936507936507936, 0.0],
    [1.0, 0.8095238095238095, 0.0],
    [1.0, 0.8253968253968254, 0.0],
    [1.0, 0.8412698412698413, 0.0],
    [1.0, 0.8571428571428571, 0.0],
    [1.0, 0.873015873015873, 0.0],
    [1.0, 0.8888888888888888, 0.0],
    [1.0, 0.9047619047619048, 0.0],
    [1.0, 0.9206349206349206, 0.0],
    [1.0, 0.9365079365079365, 0.0],
    [1.0, 0.9523809523809523, 0.0],
    [1.0, 0.9682539682539683, 0.0],
    [1.0, 0.9841269841269841, 0.0],
    [1.0, 1.0, 0.0],
]


# Generate complete color map
class AUTUMN(ColorMap):
    r"""The GNU Octave colormap `autumn`

    .. image:: _static/img/AUTUMN.png
    """

    def __init__(
        self, num_colors: int = 64, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> None:
        super().__init__(base_colormap=_BASE_AUTUMN, num_colors=num_colors, device=device, dtype=dtype)


def apply_colormap(input_tensor: Tensor, colormap: ColorMap) -> Tensor:
    r"""Apply to a gray tensor a colormap.

        The image data is assumed to be integer values.

    .. image:: _static/img/apply_colormap.png

    Args:
        input_tensor: the input tensor of a gray image.
        colormap: the colormap desired to be applied to the input tensor.

    Returns:
        A RGB tensor with the applied color map into the input_tensor

    Example:
        >>> input_tensor = torch.tensor([[[0, 1, 2], [25, 50, 63]]])
        >>> colormap = AUTUMN()
        >>> apply_colormap(input_tensor, colormap)
        tensor([[[1.0000, 1.0000, 1.0000],
                 [1.0000, 1.0000, 1.0000]],
        <BLANKLINE>
                [[0.0000, 0.0159, 0.0317],
                 [0.3968, 0.7937, 1.0000]],
        <BLANKLINE>
                [[0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000]]])
    """
    # FIXME: implement to work with RGB images
    # should work with KORNIA_CHECK_SHAPE(x, ["B","C", "H", "W"])

    KORNIA_CHECK_IS_GRAY(input_tensor)

    if len(input_tensor.shape) == 4 and input_tensor.shape[1] == 1:  # if (B x 1 X H x W)
        input_tensor = input_tensor[:, 0, ...]  # (B x H x W)
    elif len(input_tensor.shape) == 3 and input_tensor.shape[0] == 1:  # if (1 X H x W)
        input_tensor = input_tensor[0, ...]  # (H x W)

    keys = torch.arange(0, len(colormap) - 1, dtype=input_tensor.dtype, device=input_tensor.device)  # (num_colors)

    index = torch.bucketize(input_tensor, keys)  # shape equals <input_tensor>: (B x H x W) or (H x W)

    output = colormap.colors[:, index]  # (3 x B x H x W) or (3 x H x W)

    if len(output.shape) == 4:
        output = output.permute(1, 0, -2, -1)  # (B x 3 x H x W)

    return output  # (B x 3 x H x W) or (3 x H x W)


class ApplyColorMap(Module):
    r"""Module that apply to a gray tensor (integer tensor) a colormap.

    Args:
        input_tensor: the input tensor of a gray image.
        colormap: the colormap desired to be applied to the input tensor.

    Returns:
        A RGB tensor with the applied color map into the input_tensor

    Example:
        >>> input_tensor = torch.tensor([[[0, 1, 2], [25, 50, 63]]])
        >>> colormap = AUTUMN()
        >>> apply_cm = ApplyColorMap(colormap)
        >>> apply_cm(input_tensor)
        tensor([[[1.0000, 1.0000, 1.0000],
                 [1.0000, 1.0000, 1.0000]],
        <BLANKLINE>
                [[0.0000, 0.0159, 0.0317],
                 [0.3968, 0.7937, 1.0000]],
        <BLANKLINE>
                [[0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000]]])
    """

    def __init__(self, colormap: ColorMap) -> None:
        super().__init__()
        self.colormap = colormap

    def forward(self, input_tensor: Tensor) -> Tensor:
        return apply_colormap(input_tensor, self.colormap)
