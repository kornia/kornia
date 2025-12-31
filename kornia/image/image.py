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

from pathlib import Path
from typing import Any, Union

import torch
from torch.utils.dlpack import from_dlpack, to_dlpack

import kornia.color
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE
from kornia.image.base import ChannelsOrder, ColorSpace, ImageLayout, ImageSize, PixelFormat
from kornia.io.io import ImageLoadType, load_image, write_image
from kornia.utils.image_print import image_to_string

# placeholder for numpy
np_ndarray = Any
DLPack = Any


class Image:
    r"""Class that holds an Image torch.Tensor representation.

    .. note::

        Disclaimer: This class provides the minimum functionality for image manipulation. However, as soon
        as you start to experiment with advanced torch.tensor manipulation, you might expect fancy
        polymorphic behaviours.

    .. warning::

        This API is experimental and might suffer changes in the future.

    Args:
        data: a torch torch.tensor containing the image data.
        layout: a dataclass containing the image layout information.

    Examples:
        >>> # from a torch.tensor
        >>> data = torch.randint(0, 255, (3, 4, 5), dtype=torch.uint8)  # CxHxW
        >>> pixel_format = PixelFormat(
        ...     color_space=ColorSpace.RGB,
        ...     bit_depth=8,
        ... )
        >>> layout = ImageLayout(
        ...     image_size=ImageSize(4, 5),
        ...     channels=3,
        ...     channels_order=ChannelsOrder.CHANNELS_FIRST,
        ... )
        >>> img = Image(data, pixel_format, layout)
        >>> assert img.channels == 3

        >>> # from a numpy array (like opencv)
        >>> data = np.ones((4, 5, 3), dtype=np.uint8)  # HxWxC
        >>> img = Image.from_numpy(data, color_space=ColorSpace.RGB)
        >>> assert img.channels == 3
        >>> assert img.width == 5
        >>> assert img.height == 4

    """

    def __init__(self, data: torch.Tensor, pixel_format: PixelFormat, layout: ImageLayout) -> None:
        """Image constructor.

        Args:
            data: a torch torch.tensor containing the image data.
            pixel_format: the pixel format of the image.
            layout: a dataclass containing the image layout information.

        """
        # TODO: move this to a function KORNIA_CHECK_IMAGE_LAYOUT
        if layout.channels_order == ChannelsOrder.CHANNELS_FIRST:
            shape = [str(layout.channels), str(layout.image_size.height), str(layout.image_size.width)]
        elif layout.channels_order == ChannelsOrder.CHANNELS_LAST:
            shape = [str(layout.image_size.height), str(layout.image_size.width), str(layout.channels)]
        else:
            raise NotImplementedError(f"Layout {layout.channels_order} not implemented.")

        KORNIA_CHECK_SHAPE(data, shape)
        KORNIA_CHECK(data.element_size() == pixel_format.bit_depth // 8, "Invalid bit depth.")

        self._data = data
        self._pixel_format = pixel_format
        self._layout = layout

    def __repr__(self) -> str:
        return f"Image data: {self.data}\nPixel Format: {self.pixel_format}\n Layout: {self.layout}"

    # TODO: explore use TensorWrapper
    def to(self, device: Union[str, torch.device, None] = None, dtype: Union[torch.dtype, None] = None) -> Image:
        """Move the image to the given device and dtype.

        Args:
            device: the device to move the image to.
            dtype: the data type to cast the image to.

        Returns:
            Image: the image moved to the given device and dtype.

        """
        if device is not None and isinstance(device, torch.dtype):
            dtype, device = device, None
        # put the data to the device and dtype
        self._data = self.data.to(device, dtype)
        return self

    # TODO: explore use TensorWrapper
    def clone(self) -> Image:
        """Return a copy of the image."""
        return Image(self.data.clone(), self.pixel_format, self.layout)

    @property
    def data(self) -> torch.Tensor:
        """Return the underlying torch.tensor data."""
        return self._data

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the image shape."""
        return self.data.shape

    @property
    def dtype(self) -> torch.dtype:
        """Return the image data type."""
        return self.data.dtype

    @property
    def device(self) -> torch.device:
        """Return the image device."""
        return self.data.device

    @property
    def pixel_format(self) -> PixelFormat:
        """Return the pixel format."""
        return self._pixel_format

    @property
    def layout(self) -> ImageLayout:
        """Return the image layout."""
        return self._layout

    @property
    def channels(self) -> int:
        """Return the number channels of the image."""
        return self.layout.channels

    @property
    def image_size(self) -> ImageSize:
        """Return the image size."""
        return self.layout.image_size

    @property
    def height(self) -> int:
        """Return the image height (columns)."""
        return int(self.layout.image_size.height)

    @property
    def width(self) -> int:
        """Return the image width (rows)."""
        return int(self.layout.image_size.width)

    @property
    def channels_order(self) -> ChannelsOrder:
        """Return the channels order."""
        return self.layout.channels_order

    # TODO: figure out a better way map this function
    def float(self) -> Image:
        """Return the image as float."""
        self._data = self.data.float()
        return self

    def to_gray(self) -> Image:
        """Convert the image to grayscale."""
        src = self._pixel_format.color_space
        data = self._data

        if src == ColorSpace.GRAY:
            return self

        is_channels_last = self._layout.channels_order == ChannelsOrder.CHANNELS_LAST
        if is_channels_last:
            data = data.permute(0, 3, 1, 2) if data.ndim == 4 else data.permute(2, 0, 1)

        # Perform the color space conversion
        if src == ColorSpace.RGB:
            out = kornia.color.rgb_to_grayscale(data)
        elif src == ColorSpace.BGR:
            out = kornia.color.bgr_to_grayscale(data)
        else:
            raise ValueError(f"Unsupported source color space for to_gray(): {src}")

        if is_channels_last:
            if out.ndim == 4:
                out = out.permute(0, 2, 3, 1)
            elif out.ndim == 3:
                out = out.permute(1, 2, 0)
            else:
                raise ValueError(f"Unexpected shape after grayscale conversion: {out.shape}")

        new_pf = PixelFormat(color_space=ColorSpace.GRAY, bit_depth=self._pixel_format.bit_depth)
        new_layout = ImageLayout(self._layout.image_size, channels=1, channels_order=self._layout.channels_order)
        return Image(out, new_pf, new_layout)

    def to_rgb(self) -> Image:
        """Convert the image to RGB."""
        src = self._pixel_format.color_space
        data = self._data

        if src == ColorSpace.RGB:
            return self

        is_channels_last = self._layout.channels_order == ChannelsOrder.CHANNELS_LAST
        if is_channels_last:
            data = data.permute(0, 3, 1, 2) if data.ndim == 4 else data.permute(2, 0, 1)

        if src == ColorSpace.GRAY:
            out = kornia.color.grayscale_to_rgb(data)
        elif src == ColorSpace.BGR:
            out = data[:, [2, 1, 0], ...] if data.ndim == 4 else data[[2, 1, 0], ...]
        else:
            raise ValueError(f"Unsupported source color space for to_rgb(): {src}")

        if is_channels_last:
            out = out.permute(0, 2, 3, 1) if out.ndim == 4 else out.permute(1, 2, 0)

        new_pf = PixelFormat(color_space=ColorSpace.RGB, bit_depth=self._pixel_format.bit_depth)
        new_layout = ImageLayout(self._layout.image_size, channels=3, channels_order=self._layout.channels_order)
        return Image(out, new_pf, new_layout)

    def to_bgr(self) -> Image:
        """Convert the image to BGR."""
        src = self._pixel_format.color_space
        data = self._data

        if src == ColorSpace.BGR:
            return self

        is_channels_last = self._layout.channels_order == ChannelsOrder.CHANNELS_LAST
        if is_channels_last:
            data = data.permute(0, 3, 1, 2) if data.ndim == 4 else data.permute(2, 0, 1)

        if src == ColorSpace.GRAY:
            rgb_data = kornia.color.grayscale_to_rgb(data)
            out = rgb_data[:, [2, 1, 0], ...] if rgb_data.ndim == 4 else rgb_data[[2, 1, 0], ...]
        elif src == ColorSpace.RGB:
            out = data[:, [2, 1, 0], ...] if data.ndim == 4 else data[[2, 1, 0], ...]
        else:
            raise ValueError(f"Unsupported source color space for to_bgr(): {src}")

        if is_channels_last:
            out = out.permute(0, 2, 3, 1) if out.ndim == 4 else out.permute(1, 2, 0)

        new_pf = PixelFormat(color_space=ColorSpace.BGR, bit_depth=self._pixel_format.bit_depth)
        new_layout = ImageLayout(self._layout.image_size, channels=3, channels_order=self._layout.channels_order)
        return Image(out, new_pf, new_layout)

    @classmethod
    def from_numpy(
        cls,
        data: np_ndarray,
        color_space: ColorSpace = ColorSpace.RGB,
        channels_order: ChannelsOrder = ChannelsOrder.CHANNELS_LAST,
    ) -> Image:
        """Construct an image torch.tensor from a numpy array.

        Args:
            data: a numpy array containing the image data.
            color_space: the color space of the image.
            pixel_format: the pixel format of the image.
            channels_order: what dimension the channels are in the image torch.tensor.

        Example:
            >>> data = np.ones((4, 5, 3), dtype=np.uint8)  # HxWxC
            >>> img = Image.from_numpy(data, color_space=ColorSpace.RGB)
            >>> assert img.channels == 3
            >>> assert img.width == 5
            >>> assert img.height == 4

        """
        if channels_order == ChannelsOrder.CHANNELS_LAST:
            image_size = ImageSize(height=data.shape[0], width=data.shape[1])
            channels = data.shape[2]
        elif channels_order == ChannelsOrder.CHANNELS_FIRST:
            image_size = ImageSize(height=data.shape[1], width=data.shape[2])
            channels = data.shape[0]
        else:
            raise ValueError("channels_order must be either `CHANNELS_LAST` or `CHANNELS_FIRST`")

        # create the pixel format based on the input data
        pixel_format = PixelFormat(color_space=color_space, bit_depth=data.itemsize * 8)

        # create the image layout based on the input data
        layout = ImageLayout(image_size=image_size, channels=channels, channels_order=channels_order)

        # create the image torch.tensor
        return cls(torch.from_numpy(data), pixel_format, layout)

    def to_numpy(self) -> np_ndarray:
        """Return a numpy array in cpu from the image torch.tensor."""
        return self.data.cpu().detach().numpy()

    @classmethod
    def from_dlpack(cls, data: DLPack) -> Image:
        """Construct an image torch.tensor from a DLPack capsule.

        Args:
            data: a DLPack capsule from numpy, tvm or jax.

        Example:
            >>> x = np.ones((4, 5, 3))
            >>> img = Image.from_dlpack(x.__dlpack__())

        """
        _data: torch.Tensor = from_dlpack(data)

        pixel_format = PixelFormat(color_space=ColorSpace.RGB, bit_depth=_data.element_size() * 8)

        # create the image layout based on the input data
        layout = ImageLayout(
            image_size=ImageSize(height=_data.shape[1], width=_data.shape[2]),
            channels=_data.shape[0],
            channels_order=ChannelsOrder.CHANNELS_FIRST,
        )

        return cls(_data, pixel_format, layout)

    def to_dlpack(self) -> DLPack:
        """Return a DLPack capsule from the image torch.tensor."""
        return to_dlpack(self.data)

    @classmethod
    def from_file(cls, file_path: str | Path) -> Image:
        """Construct an image torch.tensor from a file.

        Args:
            file_path: the path to the file to read the image from.

        """
        # TODO: allow user to specify the desired type and device
        data: torch.Tensor = load_image(file_path, desired_type=ImageLoadType.RGB8, device="cpu")

        pixel_format = PixelFormat(color_space=ColorSpace.RGB, bit_depth=data.element_size() * 8)

        layout = ImageLayout(
            image_size=ImageSize(height=data.shape[1], width=data.shape[2]),
            channels=data.shape[0],
            channels_order=ChannelsOrder.CHANNELS_FIRST,
        )
        return cls(data, pixel_format, layout)

    def write(self, file_path: str | Path) -> None:
        """Write the image to a file.

        For now, only support writing to JPEG format.

        Args:
            file_path: the path to the file to write the image to.

        Example:
            >>> data = np.ones((4, 5, 3), dtype=np.uint8)  # HxWxC
            >>> img = Image.from_numpy(data)
            >>> img.write("test.jpg")

        """
        data = self.data
        if self.channels_order == ChannelsOrder.CHANNELS_LAST:
            data = data.permute(2, 0, 1)
        write_image(file_path, data)

    def print(self, max_width: int = 256) -> None:
        """Print the image torch.tensor to the console.

        Args:
            max_width: the maximum width of the image to print.

        .. code-block:: python

            img = Image.from_file("panda.png")
            img.print()

        .. image:: https://github.com/kornia/data/blob/main/print_image.png?raw=true

        """
        print(image_to_string(self.data, max_width))
