from __future__ import annotations

from enum import Enum
from typing import Optional, Union

import torch
from torch.nn.functional import interpolate

import kornia.color._colormap_data as cm_data
from kornia.color._colormap_data import RGBColor
from kornia.core import Module, Tensor, tensor
from kornia.core.check import KORNIA_CHECK
from kornia.utils.helpers import deprecated


class ColorMapType(Enum):
    r"""An enumeration for available colormaps.

    List of available colormaps:

    .. image:: _static/img/ColorMapType.png
    """

    autumn = 1
    bone = 2
    jet = 3
    winter = 4
    rainbow = 5
    ocean = 6
    summer = 7
    spring = 8
    cool = 9
    hsv = 10
    brg = 11
    pink = 12
    hot = 13
    plasma = 14
    viridis = 15
    cividis = 16
    twilight = 17
    turbo = 18
    seismic = 19

    def _load_base(self) -> list[RGBColor]:
        r"""Load the base colormap corresponding to the enumeration member.

        Returns:
            The base colormap.
        """
        return {
            "autumn": cm_data.get_autumn_base,
            "bone": cm_data.get_bone_base,
            "jet": cm_data.get_jet_base,
            "winter": cm_data.get_winter_base,
            "rainbow": cm_data.get_rainbow_base,
            "ocean": cm_data.get_ocean_base,
            "summer": cm_data.get_summer_base,
            "spring": cm_data.get_spring_base,
            "cool": cm_data.get_cool_base,
            "hsv": cm_data.get_hsv_base,
            "brg": cm_data.get_bgr_base,
            "pink": cm_data.get_pink_base,
            "hot": cm_data.get_hot_base,
            "plasma": cm_data.get_plasma_base,
            "viridis": cm_data.get_viridis_base,
            "cividis": cm_data.get_cividis_base,
            "twilight": cm_data.get_twilight_base,
            "turbo": cm_data.get_turbo_base,
            "seismic": cm_data.get_seismic_base,
        }[self.name]()

    @classmethod
    def list(cls) -> list[str]:
        r"""Returns a list of names of enumeration members.

        Returns:
            A list containing the names of enumeration members.
        """
        return [c.name for c in cls]


class ColorMap:
    r"""Class to represent a colour map. It can be created or selected from the built-in colour map. Please refer to
    the `ColorMapType` enum class to view all available colormaps.

    Args:
        base: A list of RGB colors to define a new custom colormap or the name of a built-in colormap as str or
        using `ColorMapType` class.
        num_colors: Number of colors in the colormap.
        device: The device to put the generated colormap on.
        dtype: The data type of the generated colormap.

    Returns:
        An object of the colormap with the num_colors length.

    Examples:
        >>> ColorMap(base='viridis', num_colors=8).colors
        tensor([[0.2813, 0.2621, 0.2013, 0.1505, 0.1210, 0.2463, 0.5259, 0.8557],
                [0.0842, 0.2422, 0.3836, 0.5044, 0.6258, 0.7389, 0.8334, 0.8886],
                [0.4072, 0.5207, 0.5543, 0.5574, 0.5334, 0.4519, 0.2880, 0.0989]])

    Create a color map from the first color (RGB with range[0-1]) to the last one with num_colors length.
        >>> ColorMap(base=[[0., 0.5 , 1.0], [1., 0.5, 0.]], num_colors=8).colors
        tensor([[0.0000, 0.0000, 0.1250, 0.3750, 0.6250, 0.8750, 1.0000, 1.0000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [1.0000, 1.0000, 0.8750, 0.6250, 0.3750, 0.1250, 0.0000, 0.0000]])
    """

    def __init__(
        self,
        base: Union[list[RGBColor], str, ColorMapType],
        num_colors: int = 64,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self._dtype = dtype
        self._device = device

        if isinstance(base, str):
            base = base.lower()
            if base not in ColorMapType.list():
                raise ValueError(f"Unsupported colormap: {base}. Available colormaps are {ColorMapType.list()}")
            base_colormap_data = ColorMapType[base]._load_base()
            self.name = base
        elif isinstance(base, ColorMapType):
            base_colormap_data = base._load_base()
            self.name = base.name
        elif isinstance(base, list):
            base_colormap_data = base
            self.name = "CustomColorMapType"
        else:
            raise ValueError(
                "Base should be one of the available `ColorMapType` or a base colormap data (list[RGBColor])"
            )

        self.colors = self._generate_color_map(base_colormap_data, num_colors)

    def _generate_color_map(self, base_colormap: list[RGBColor], num_colors: int) -> Tensor:
        r"""Generates a colormap tensor using interpolation.

        Args:
            base_colormap: A list of RGB colors defining the colormap.
            num_colors: Number of colors in the colormap.

        Returns:
            A tensor representing the colormap.
        """
        tensor_colors = tensor(list(base_colormap), dtype=self._dtype, device=self._device).T
        return interpolate(tensor_colors[None, ...], size=num_colors, mode="linear")[0, ...]

    def __len__(self) -> int:
        r"""Returns the number of colors in the colormap.

        Returns:
            Number of colors in the colormap.
        """
        return self.colors.shape[-1]


def apply_colormap(input_tensor: Tensor, colormap: ColorMap) -> Tensor:
    r"""Apply to a gray tensor a colormap.

    .. image:: _static/img/apply_colormap.png

    Args:
        input_tensor: the input tensor of image.
        colormap: the colormap desired to be applied to the input tensor.

    Returns:
        A RGB tensor with the applied color map into the input_tensor.

    Raises:
        ValueError: If `colormap` is not a ColorMap object.

    .. note::
        The input tensor must be integer values in the range of [0-255] or float values in the range of [0-1].

    Example:
        >>> input_tensor = torch.tensor([[[0, 1, 2], [15, 25, 33], [128, 158, 188]]])
        >>> colormap = ColorMap(base=ColorMapType.autumn)
        >>> apply_colormap(input_tensor, colormap)
        tensor([[[[1.0000, 1.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000]],
        <BLANKLINE>
                 [[0.0000, 0.0159, 0.0159],
                  [0.0635, 0.1111, 0.1429],
                  [0.5079, 0.6190, 0.7302]],
        <BLANKLINE>
                 [[0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000]]]])
    """

    KORNIA_CHECK(isinstance(input_tensor, Tensor), f"`input_tensor` must be a Tensor. Got: {type(input_tensor)}")
    valid_types = [torch.half, torch.float, torch.double, torch.uint8, torch.int, torch.long, torch.short]
    KORNIA_CHECK(
        input_tensor.dtype in valid_types, f"`input_tensor` must be a {valid_types}. Got: {input_tensor.dtype}"
    )
    KORNIA_CHECK(len(input_tensor.shape) in (3, 4), "Wrong input tensor dimension.")
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze_(0)

    B, C, H, W = input_tensor.shape
    input_tensor = input_tensor.reshape(B, C, -1)
    max_value = 1.0 if input_tensor.max() <= 1.0 else 255.0
    input_tensor = input_tensor.float().div_(max_value)

    colors = colormap.colors.permute(1, 0)
    num_colors, channels_cmap = colors.shape
    keys = torch.linspace(0.0, 1.0, num_colors - 1, device=input_tensor.device, dtype=input_tensor.dtype)
    indices = torch.bucketize(input_tensor, keys).unsqueeze(-1).expand(-1, -1, -1, 3)

    output = torch.gather(colors.expand(B, C, -1, -1), 2, indices)
    # (B, C, H*W, channels_cmap) -> (B, C*channels_cmap, H, W)
    output = output.permute(0, 1, 3, 2).reshape(B, C * channels_cmap, H, W)

    return output


class ApplyColorMap(Module):
    r"""Class for applying a colormap to images.

    .. image:: _static/img/ApplyColorMap.png

    Args:
        colormap: Either the name of a built-in colormap or a ColorMap object.
        num_colors: Number of colors in the colormap. Default is 256.
        device: The device to put the generated colormap on.
        dtype: The data type of the generated colormap.

    Returns:
        A RGB tensor with the applied color map into the input_tensor

    Raises:
        ValueError: If `colormap` is not a ColorMap object.

    .. note::
        The input tensor must be integer values in the range of [0-255] or float values in the range of [0-1].

    Example:
        >>> input_tensor = torch.tensor([[[0, 1, 2], [15, 25, 33], [128, 158, 188]]])
        >>> colormap = ColorMap(base=ColorMapType.autumn)
        >>> ApplyColorMap(colormap=colormap)(input_tensor)
        tensor([[[[1.0000, 1.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000]],
        <BLANKLINE>
                 [[0.0000, 0.0159, 0.0159],
                  [0.0635, 0.1111, 0.1429],
                  [0.5079, 0.6190, 0.7302]],
        <BLANKLINE>
                 [[0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000]]]])
    """

    def __init__(
        self,
        colormap: ColorMap,
    ) -> None:
        super().__init__()
        self.colormap = colormap

    def forward(self, input_tensor: Tensor) -> Tensor:
        r"""Applies the colormap to the input tensor.

        Args:
            input_tensor: The input tensor representing the grayscale image.

        .. note::
        The input tensor must be integer values in the range of [0-255] or float values in the range of [0-1].

        Returns:
            The output tensor representing the image with the applied colormap.
        """
        return apply_colormap(input_tensor, self.colormap)


# Generate complete color map
@deprecated(
    "0.7.2",
    extra_reason="The `AUTUMN()` class is deprecated and will be removed in next kornia versions (0.8.0 - dec 2024).\
    In favor of using `ColorMap(base='autumn')` instead.",
)
class AUTUMN(ColorMap):
    r"""The GNU Octave colormap `autumn`

    .. image:: _static/img/AUTUMN.png
    """

    def __init__(
        self, num_colors: int = 64, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> None:
        super().__init__(base=ColorMapType.autumn, num_colors=num_colors, device=device, dtype=dtype)
