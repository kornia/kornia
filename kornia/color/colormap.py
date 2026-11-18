# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Union

import torch
from torch import nn
from torch.nn.functional import interpolate

import kornia.color._colormap_data as cm_data
from kornia.color._colormap_data import RGBColor
from kornia.core.check import KORNIA_CHECK


class ColorMapType(Enum):
    r"""Names of the built-in colormaps.

    ``ColorMapType.viridis``, or its name ``'viridis'``, selects a built-in colormap in :class:`ColorMap`.
    :meth:`list` returns all names.

    .. image:: _static/img/ColorMapType.png

    See Also:
        - :class:`ColorMap`: builds the palette of a member.
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
        r"""Return a list of names of enumeration members.

        Returns:
            A list containing the names of enumeration members.

        """
        return [c.name for c in cls]


class ColorMap:
    r"""Color palette for :func:`apply_colormap`.

    ``cmap = ColorMap(base, num_colors)`` builds a built-in colormap, or one from your own RGB triples, with
    ``num_colors`` entries and stores it as ``cmap.colors``.

    Convention:
        colors has shape (3, num_colors) in RGB channel order. Built-in names are case-insensitive;
        a custom base is an ordered list of RGB triples in :math:`[0, 1]` and is linearly resampled to num_colors.
        For ``num_colors > 1``, entry ``i`` sits at ``i / (num_colors - 1)`` along the base, so the first and
        last colors of the base are the first and last of the map. With ``num_colors == 1``, the map holds the
        base's first color.

    Args:
        base: list of RGB triples, or a built-in colormap as a :class:`ColorMapType` or its name.
        num_colors: number of palette entries.
        device: device of the palette.
        dtype: dtype of the palette. Defaults to ``torch.get_default_dtype()``.

    See Also:
        - :class:`ColorMapType`: the built-in colormaps.
        - :func:`apply_colormap`: colors an image with the palette.
        - :class:`ApplyColorMap`: the module form of :func:`apply_colormap`.

    Examples:
        >>> ColorMap(base='viridis', num_colors=8).colors
        tensor([[0.2670, 0.2752, 0.2124, 0.1534, 0.1223, 0.2889, 0.6266, 0.9932],
                [0.0049, 0.1949, 0.3597, 0.4970, 0.6332, 0.7584, 0.8546, 0.9062],
                [0.3294, 0.4960, 0.5517, 0.5577, 0.5304, 0.4284, 0.2234, 0.1439]])

        >>> ColorMap(base=[[0., 0.5 , 1.0], [1., 0.5, 0.]], num_colors=8).colors
        tensor([[0.0000, 0.1429, 0.2857, 0.4286, 0.5714, 0.7143, 0.8571, 1.0000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [1.0000, 0.8571, 0.7143, 0.5714, 0.4286, 0.2857, 0.1429, 0.0000]])

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

    def _generate_color_map(self, base_colormap: list[RGBColor], num_colors: int) -> torch.Tensor:
        r"""Generate a colormap torch.Tensor using interpolation.

        Args:
            base_colormap: A list of RGB colors defining the colormap.
            num_colors: Number of colors in the colormap.

        Returns:
            A torch.Tensor representing the colormap.

        """
        dtype = self._dtype if self._dtype is not None else torch.get_default_dtype()
        tensor_colors = torch.tensor(list(base_colormap), dtype=dtype, device=self._device).T
        # align_corners=True keeps the base's end colors at the ends of the map (#6556): cell-centre sampling
        # repeated them when upsampling and dropped them when downsampling.
        return interpolate(tensor_colors[None, ...], size=num_colors, mode="linear", align_corners=True)[0, ...]

    def __len__(self) -> int:
        r"""Return the number of colors in the colormap.

        Returns:
            Number of colors in the colormap.

        """
        return self.colors.shape[-1]


def apply_colormap(input_tensor: torch.Tensor, colormap: ColorMap) -> torch.Tensor:
    r"""Color each channel of an image with a colormap.

    ``out = apply_colormap(input_tensor, colormap)`` replaces each value of each channel with a color of
    ``colormap``.

    .. image:: _static/img/apply_colormap.png

    Convention:
        Accepts a rank-3 (C, H, W) or rank-4 (B, C, H, W) tensor and returns
        (B, 3*C, H, W), with B = 1 for rank-3 input; input channel c becomes output channels
        3c to 3c + 2. Integer inputs use [0, 255]. Floating inputs use [0, 1] or [0, 255]
        independently for each sample and channel, based on that channel's maximum.
        A value x in [0, 1] selects entry floor(x * N) of the N palette colors, and x = 1 selects the last,
        as in matplotlib, so every color covers an equal share of the range. Values below or above the range
        select the first or the last color.

    Args:
        input_tensor: image with shape ``(C, H, W)`` or ``(B, C, H, W)``.
        colormap: palette to apply.

    Returns:
        Color image with shape ``(B, 3 * C, H, W)`` and the dtype of the palette.

    See Also:
        - :class:`ApplyColorMap`: the module form.
        - :class:`ColorMap`: builds the palette.

    Example:
        >>> input_tensor = torch.tensor([[[0, 1, 2], [15, 25, 33], [128, 158, 188]]])
        >>> colormap = ColorMap(base=ColorMapType.autumn)
        >>> apply_colormap(input_tensor, colormap)
        tensor([[[[1.0000, 1.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000]],
        <BLANKLINE>
                 [[0.0000, 0.0000, 0.0000],
                  [0.0476, 0.0952, 0.1270],
                  [0.5079, 0.6190, 0.7460]],
        <BLANKLINE>
                 [[0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000]]]])

    """
    return _apply_colormap(input_tensor, colormap.colors)


def _apply_colormap(input_tensor: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
    """Apply the palette ``colors`` of shape (3, N); shared by :func:`apply_colormap` and :class:`ApplyColorMap`."""
    KORNIA_CHECK(
        isinstance(input_tensor, torch.Tensor), f"`input_tensor` must be a torch.Tensor. Got: {type(input_tensor)}"
    )
    valid_types = [
        torch.bfloat16,
        torch.half,
        torch.float,
        torch.double,
        torch.uint8,
        torch.int,
        torch.long,
        torch.short,
    ]
    KORNIA_CHECK(
        input_tensor.dtype in valid_types, f"`input_tensor` must be a {valid_types}. Got: {input_tensor.dtype}"
    )
    KORNIA_CHECK(len(input_tensor.shape) in (3, 4), "Wrong input torch.Tensor dimension.")
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)

    B, C, H, W = input_tensor.shape
    # Name the pixel count: `-1` cannot be inferred when the tensor has no elements.
    input_tensor = input_tensor.reshape(B, C, H * W)
    if input_tensor.is_floating_point():
        # The tensor-valued condition keeps range selection compile-safe and independent
        # of other samples and channels. `all` rather than `amax`, which refuses a channel without pixels;
        # a NaN selects 255 either way.
        max_value = torch.where(
            (input_tensor <= 1.0).all(dim=-1, keepdim=True),
            torch.tensor(1.0, device=input_tensor.device, dtype=torch.float),
            torch.tensor(255.0, device=input_tensor.device, dtype=torch.float),
        )
        input_tensor = input_tensor.float()
        # Compiled CUDA division can misclassify colormap bin edges (#6656).
        # Use addcdiv for precise division while preserving eager and ONNX behavior.
        if torch.compiler.is_compiling():
            input_tensor = torch.addcdiv(input_tensor.new_zeros(()), input_tensor, max_value)
        else:
            input_tensor = input_tensor / max_value
    else:
        input_tensor = input_tensor.float() / 255.0

    colors = colors.permute(1, 0)
    num_colors, channels_cmap = colors.shape
    # matplotlib's lookup (#6622): equal bins, floor(x * N), with x = 1 and anything above in the last bin. Bucketing
    # on x * N instead of casting it keeps a NaN on the last entry on every device, as before.
    keys = torch.arange(1, num_colors, device=input_tensor.device, dtype=input_tensor.dtype)
    indices = torch.bucketize(input_tensor * num_colors, keys, right=True)
    indices = indices.unsqueeze(-1).expand(-1, -1, -1, 3)

    output = torch.gather(colors.expand(B, C, -1, -1), 2, indices)
    # (B, C, H*W, channels_cmap) -> (B, C*channels_cmap, H, W)

    return output.permute(0, 1, 3, 2).reshape(B, C * channels_cmap, H, W)


class ApplyColorMap(nn.Module):
    r"""Color each channel of an image with a colormap (module form).

    ``cmap = ApplyColorMap(colormap)`` stores the palette as the buffer ``colors``, and ``out = cmap(input_tensor)``
    is :func:`apply_colormap` with it.

    .. image:: _static/img/ApplyColorMap.png

    Convention:
        - See the convention block of :func:`apply_colormap`.

    Args:
        colormap: palette to apply.

    Shape:
        - input_tensor: ``(C, H, W)`` or ``(B, C, H, W)``
        - output: ``(B, 3 * C, H, W)``

    See Also:
        - :func:`apply_colormap`: the one-call form.
        - :class:`ColorMap`: builds the palette.

    Example:
        >>> input_tensor = torch.tensor([[[0, 1, 2], [15, 25, 33], [128, 158, 188]]])
        >>> colormap = ColorMap(base=ColorMapType.autumn)
        >>> ApplyColorMap(colormap=colormap)(input_tensor)
        tensor([[[[1.0000, 1.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000]],
        <BLANKLINE>
                 [[0.0000, 0.0000, 0.0000],
                  [0.0476, 0.0952, 0.1270],
                  [0.5079, 0.6190, 0.7460]],
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
        self.register_buffer("colors", colormap.colors)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # State dicts saved before the palette became a buffer have no ``colors`` key. Keep the palette this
        # module was built with, so they still load with ``strict=True``, including when nested in another module.
        state_dict.setdefault(prefix + "colors", self.colors)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        r"""Color an image with the stored palette.

        Args:
            input_tensor: Image with shape ``(C, H, W)`` or ``(B, C, H, W)``, in the value ranges of
                :func:`apply_colormap`.

        Returns:
            Color image with shape ``(B, 3 * C, H, W)``.

        """
        return _apply_colormap(input_tensor, self.colors)
