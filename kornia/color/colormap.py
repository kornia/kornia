from abc import ABC
from typing import List, Optional, Union

import torch
from torch.nn.functional import interpolate

from kornia.color import colormap_data as cm_data
from kornia.core import Module, Tensor, tensor
from kornia.core.check import KORNIA_CHECK_IS_GRAY

RGBColor = cm_data.RGBColor

colormap_options = {
    "autumn": cm_data._AUTUMN_BASE,
    "bone": cm_data._BONE_BASE,
    "jet": cm_data._JET_BASE,
    "winter": cm_data._WINTER_BASE,
    "rainbow": cm_data._RAINBOW_BASE,
    "ocean": cm_data._OCEAN_BASE,
    "summer": cm_data._SUMMER_BASE,
    "spring": cm_data._SPRING_BASE,
    "cool": cm_data._COOL_BASE,
    "hsv": cm_data._HSV_BASE,
    "brg": cm_data._BRG_BASE,
    "pink": cm_data._PINK_BASE,
    "hot": cm_data._HOT_BASE,
    "plasma": cm_data._PLASMA_BASE,
    "viridis": cm_data._VIRIDIS_BASE,
    "cividis": cm_data._CIVIDIS_BASE,
    "twilight": cm_data._TWILIGHT_BASE,
    "turbo": cm_data._TURBO_BASE,
    "seismic": cm_data._SEISMIC_BASE,
}


class ColorMap(ABC):
    r"""Class to represent a colour map. It can be created or selected from the built-in colour map.

    Args:
        base_colormap: A list of RGB colors to define a custom colormap.
        If specified, `name_colormap` should be None.
        name_colormap: The name of a built-in colormap.
        If specified, `base_colormap` should be None.
        num_colors: Number of colors in the colormap.
        device: The device to put the generated colormap on.
        dtype: The data type of the generated colormap.

    Returns:
            ColorMap: An object of the colormap with the num_colors length.

    Raises:
        ValueError: If both `base_colormap` and `name_colormap` are provided or if neither is provided.

    Example:
        >>> ColorMap(name_colormap='viridis', num_colors=8).colors
        tensor([[0.2813, 0.2621, 0.2013, 0.1505, 0.1210, 0.2463, 0.5259, 0.8557],
                [0.0842, 0.2422, 0.3836, 0.5044, 0.6258, 0.7389, 0.8334, 0.8886],
                [0.4072, 0.5207, 0.5543, 0.5574, 0.5334, 0.4519, 0.2880, 0.0989]])

        # Create color map from first color (RGB with range[0-1]) to last one with num_colors length.
        >>> ColorMap(base_colormap=[[0., 0.5 , 1.0], [1., 0.5, 0.]], num_colors=8).colors
        tensor([[0.0000, 0.0000, 0.1250, 0.3750, 0.6250, 0.8750, 1.0000, 1.0000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [1.0000, 1.0000, 0.8750, 0.6250, 0.3750, 0.1250, 0.0000, 0.0000]])
    """

    def __init__(
        self,
        base_colormap: Optional[List[RGBColor]] = None,
        name_colormap: Optional[str] = None,
        num_colors: Optional[int] = 64,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self._dtype = dtype
        self._device = device
        self.colormap_options = colormap_options
        self.colormap_names = list(self.colormap_options.keys())

        if name_colormap is not None:
            cmap = name_colormap.lower()
            if cmap not in self.colormap_names:
                raise ValueError(f"Unsupported colormap: {cmap}. Available colormaps are {self.colormap_names}")
            self.colors = self._generate_color_map(self.colormap_options[cmap], num_colors)
        elif base_colormap is not None:
            self.colors = self._generate_color_map(base_colormap, num_colors)
        else:
            raise ValueError("You should use `base_colormap` (List[RGBColor]) or `name_colormap` (str).")

    def _generate_color_map(self, base_colormap: List[RGBColor], num_colors: int) -> Tensor:
        r"""Generates a colormap tensor using interpolation.

        Args:
            base_colormap: A list of RGB colors defining the colormap.
            num_colors: Number of colors in the colormap.

        Returns:
            Tensor: A tensor representing the colormap.
        """
        tensor_colors = tensor(list(base_colormap), dtype=self._dtype, device=self._device).T
        return interpolate(tensor_colors[None, ...], size=num_colors, mode="linear")[0, ...]

    def __len__(self) -> int:
        r"""Returns the number of colors in the colormap.

        Returns:
            int: Number of colors in the colormap.
        """
        return self.colors.shape[-1]

    def colormap_list(self) -> List:
        r"""Returns the list of colormaps available.

        Returns:
            List: Name of colorsmaps available.
        """
        return self.colormap_names


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
        >>> colormap = ColorMap(name_colormap='autumn')
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
    r"""Class for applying a colormap to images.

    Args:
        colormap: Either the name of a built-in colormap or a ColorMap object.
        num_colors: Number of colors in the colormap. Default is 256.
        device: The device to put the generated colormap on.
        dtype: The data type of the generated colormap.

    Returns:
        A RGB tensor with the applied color map into the input_tensor

    Raises:
        ValueError: If `colormap` is neither a string nor a ColorMap object.

    .. note::
        The image data is assumed to be integer values in range of [0-255].

    Example:
        >>> input_tensor = torch.tensor([[[0, 1, 2], [25, 50, 63]]])
        >>> ApplyColorMap(colormap='autumn')(input_tensor)
        tensor([[[1.0000, 1.0000, 1.0000],
                 [1.0000, 1.0000, 1.0000]],
        <BLANKLINE>
                [[0.0000, 0.0159, 0.0317],
                 [0.3968, 0.7937, 1.0000]],
        <BLANKLINE>
                [[0.0000, 0.0000, 0.0000],
                 [0.0000, 0.0000, 0.0000]]])
    """

    def __init__(
        self,
        colormap: Union[str, ColorMap],
        num_colors: Optional[int] = 64,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.num_colors = num_colors
        self.device = device
        self.dtype = dtype

        if isinstance(colormap, str):
            self.cmap = ColorMap(
                name_colormap=colormap,
                num_colors=self.num_colors,
                device=self.device,
                dtype=self.dtype,
            )
        elif isinstance(colormap, ColorMap):
            self.cmap = colormap
        else:
            raise ValueError("Please provide a correct colormap. Could be a str or ColorMap object.")

    def forward(self, input_tensor: Tensor) -> Tensor:
        r"""Applies the colormap to the input tensor.

        Args:
            input_tensor: The input tensor representing the grayscale image.

        .. note::
        The image data is assumed to be integer values in range of [0-255].

        Returns:
            Tensor: The output tensor representing the image with the applied colormap.
        """
        return apply_colormap(input_tensor, self.cmap)
