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
from typing import Any, Optional, Union

import torch
from torch.utils.dlpack import from_dlpack, to_dlpack

import kornia.color
from kornia.core.check import are_checks_enabled
from kornia.core.exceptions import ValueCheckError
from kornia.image.base import (
    KORNIA_CHECK_IMAGE_LAYOUT,
    ChannelsOrder,
    ColorSpace,
    ImageLayout,
    ImageSize,
    PixelFormat,
)
from kornia.image.image_print import image_to_string
from kornia.io.io import ImageLoadType, load_image, write_image

# placeholder for numpy
np_ndarray = Any
DLPack = Any


# the channel count each colour space implies; UNKNOWN accepts any count
_CHANNELS_OF_COLOR_SPACE = {ColorSpace.GRAY: 1, ColorSpace.RGB: 3, ColorSpace.BGR: 3}


def _default_color_space(channels: int) -> ColorSpace:
    """Label an image by its channel count: 1 is GRAY, 3 is RGB, anything else is UNKNOWN."""
    return {1: ColorSpace.GRAY, 3: ColorSpace.RGB}.get(channels, ColorSpace.UNKNOWN)


class Image:
    r"""Class that holds an Image torch.Tensor representation.

    .. warning::

        This API is experimental and might suffer changes in the future.

    Convention:
        - An ``Image`` holds one image, :math:`(C, H, W)` or :math:`(H, W, C)` as ``layout.channels_order`` says,
          never a batch: a 4-D tensor raises ``ShapeError``. The layout and the pixel format are checked against the
          data: ``bit_depth`` is the storage size, ``8 * data.element_size()``, and the color space must fit the
          channel count, 1 for ``GRAY``, 3 for ``RGB`` and ``BGR``, any for ``UNKNOWN``.
        - :meth:`from_numpy`, :meth:`to_numpy`, :meth:`from_dlpack` and :meth:`to_dlpack` share memory with the
          array or capsule of a CPU image. :meth:`to_numpy` and :meth:`to_dlpack` return the stored layout, so an
          image from :meth:`from_file` gives :math:`(C, H, W)`. Neither the array nor the capsule carries the layout
          or the color space: :meth:`from_numpy` reads :math:`(H, W, C)` and :meth:`from_dlpack` reads
          :math:`(C, H, W)` unless ``channels_order`` says otherwise, and both label the color space by channel count
          unless ``color_space`` is given. Pass ``img.pixel_format.color_space`` and ``img.channels_order`` to read an
          exported image back.
        - :meth:`to_gray`, :meth:`to_rgb` and :meth:`to_bgr` return the image itself when it is already in that color
          space, an image with new data when it is in another of the three, and raise ``ValueError`` for
          ``UNKNOWN``. Converting RGB or BGR to gray requires ``uint8`` or a floating dtype supported by
          :func:`~kornia.color.rgb_to_grayscale`; other integer dtypes, such as ``uint16``, are unsupported even
          though ``Image`` can hold them. An already-gray image is returned unchanged regardless of dtype.
          :meth:`to`, :meth:`float` and :meth:`clone` return a new ``Image`` and leave this one unchanged.

    Args:
        data: a torch.Tensor containing the image data.
        pixel_format: the pixel format of the image.
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

        Like the other kornia checks, the pixel-format checks are skipped after ``kornia.core.check.disable_checks()``.

        Args:
            data: a torch.Tensor containing the image data.
            pixel_format: the pixel format of the image.
            layout: a dataclass containing the image layout information.

        Raises:
            ShapeError: if ``data`` does not have the shape ``layout`` describes. It is also a ``ValueError``.
            ValueCheckError: if ``pixel_format.bit_depth`` is not ``8 * data.element_size()`` (also a ``ValueError``).
            ValueCheckError: if the color space does not match the channel count. It is also a ``ValueError``.

        """
        KORNIA_CHECK_IMAGE_LAYOUT(data, layout)
        if are_checks_enabled():
            storage_bits = 8 * data.element_size()
            if pixel_format.bit_depth != storage_bits:
                raise ValueCheckError(
                    f"Invalid bit depth: the pixel format says {pixel_format.bit_depth} bits but the data is stored "
                    f"in {storage_bits} bits ({data.dtype}).",
                    actual_value=pixel_format.bit_depth,
                    expected_range=(storage_bits, storage_bits),
                )
            expected_channels = _CHANNELS_OF_COLOR_SPACE.get(pixel_format.color_space)
            if expected_channels is not None and layout.channels != expected_channels:
                raise ValueCheckError(
                    f"Invalid color space: {pixel_format.color_space.name} needs {expected_channels} channel(s) but "
                    f"the layout has {layout.channels}. Use ColorSpace.UNKNOWN for other channel counts.",
                    actual_value=layout.channels,
                    expected_range=(expected_channels, expected_channels),
                )

        self._data = data
        self._pixel_format = pixel_format
        self._layout = layout

    def __repr__(self) -> str:
        return f"Image data: {self.data}\nPixel Format: {self.pixel_format}\n Layout: {self.layout}"

    # TODO: explore use TensorWrapper
    def to(self, device: Union[str, torch.device, None] = None, dtype: Union[torch.dtype, None] = None) -> Image:
        """Return the image on ``device`` and with ``dtype``, like :meth:`torch.Tensor.to`.

        The image itself is not changed; assign the result (``img = img.to(...)``). The pixel format's ``bit_depth``
        follows the new dtype.

        Args:
            device: the device to move the image to.
            dtype: the data type to cast the image to.

        Returns:
            Image: the image on the given device and with the given dtype.

        Raises:
            ValueCheckError: if the image was built with checks disabled and its color space does not fit its channels.

        """
        if device is not None and isinstance(device, torch.dtype):
            dtype, device = device, None
        data = self.data.to(device, dtype)
        pixel_format = PixelFormat(color_space=self.pixel_format.color_space, bit_depth=data.element_size() * 8)
        return Image(data, pixel_format, self.layout)

    # TODO: explore use TensorWrapper
    def clone(self) -> Image:
        """Return a copy of the image."""
        return Image(self.data.clone(), self.pixel_format, self.layout)

    @property
    def data(self) -> torch.Tensor:
        """Return the underlying torch.Tensor data."""
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
        """Return the image height (number of rows)."""
        return int(self.layout.image_size.height)

    @property
    def width(self) -> int:
        """Return the image width (number of columns)."""
        return int(self.layout.image_size.width)

    @property
    def channels_order(self) -> ChannelsOrder:
        """Return the channels order."""
        return self.layout.channels_order

    # TODO: figure out a better way map this function
    def float(self) -> Image:
        """Return the image as float32; the image itself is not changed."""
        return self.to(dtype=torch.float32)

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
        color_space: Optional[ColorSpace] = None,
        channels_order: ChannelsOrder = ChannelsOrder.CHANNELS_LAST,
    ) -> Image:
        """Construct an ``Image`` from a numpy array, sharing its memory.

        Args:
            data: a numpy array containing the image data.
            color_space: the color space of the image. If ``None``, a 1-channel array is ``GRAY``, a 3-channel
                array is ``RGB`` and any other channel count is ``UNKNOWN``. An explicit color space that does not
                match the channel count raises.
            channels_order: what dimension the channels are in the image torch.Tensor.

        Raises:
            ValueError: if ``channels_order`` is neither ``CHANNELS_LAST`` nor ``CHANNELS_FIRST``.
            ValueCheckError: if an explicit ``color_space`` does not match the channel count (also a ``ValueError``).
            ShapeError: if ``data`` has more than three dimensions. It is also a ``ValueError``.
            IndexError: if ``data`` has fewer than two dimensions.

        Example:
            >>> data = np.ones((4, 5, 3), dtype=np.uint8)  # HxWxC
            >>> img = Image.from_numpy(data, color_space=ColorSpace.RGB)
            >>> assert img.channels == 3
            >>> assert img.width == 5
            >>> assert img.height == 4

        """
        if data.ndim == 2:
            # a single-channel image without a channel axis
            data = data[..., None] if channels_order == ChannelsOrder.CHANNELS_LAST else data[None]
        if channels_order == ChannelsOrder.CHANNELS_LAST:
            image_size = ImageSize(height=data.shape[0], width=data.shape[1])
            channels = data.shape[2]
        elif channels_order == ChannelsOrder.CHANNELS_FIRST:
            image_size = ImageSize(height=data.shape[1], width=data.shape[2])
            channels = data.shape[0]
        else:
            raise ValueError("channels_order must be either `CHANNELS_LAST` or `CHANNELS_FIRST`")

        if color_space is None:
            color_space = _default_color_space(channels)
        # create the pixel format based on the input data
        pixel_format = PixelFormat(color_space=color_space, bit_depth=data.itemsize * 8)

        # create the image layout based on the input data
        layout = ImageLayout(image_size=image_size, channels=channels, channels_order=channels_order)

        # create the image torch.Tensor
        return cls(torch.from_numpy(data), pixel_format, layout)

    def to_numpy(self) -> np_ndarray:
        """Return the image data as a numpy array on the CPU, in the stored layout."""
        return self.data.cpu().detach().numpy()

    @classmethod
    def from_dlpack(
        cls,
        data: DLPack,
        color_space: Optional[ColorSpace] = None,
        channels_order: ChannelsOrder = ChannelsOrder.CHANNELS_FIRST,
    ) -> Image:
        """Construct an ``Image`` from a DLPack capsule, sharing its memory.

        A capsule carries neither the layout nor the color space, so read ``img.to_dlpack()`` back with
        ``Image.from_dlpack(capsule, img.pixel_format.color_space, img.channels_order)``.

        Args:
            data: a DLPack capsule from numpy, tvm or jax, holding a ``(C, H, W)`` image, or ``(H, W, C)`` with
                ``channels_order=CHANNELS_LAST``.
            color_space: the color space of the image. If ``None``, a 1-channel image is ``GRAY``, a 3-channel
                image is ``RGB`` and any other channel count is ``UNKNOWN``. An explicit color space that does not
                match the channel count raises.
            channels_order: the layout of the capsule.

        Raises:
            ValueError: if ``channels_order`` is neither ``CHANNELS_LAST`` nor ``CHANNELS_FIRST``.
            ValueCheckError: if an explicit ``color_space`` does not match the channel count (also a ``ValueError``).
            ShapeError: if the capsule has more than three dimensions. It is also a ``ValueError``.
            IndexError: if the capsule has fewer than three dimensions.

        Example:
            >>> x = np.ones((3, 4, 5))  # (C, H, W)
            >>> img = Image.from_dlpack(x.__dlpack__())
            >>> img.channels, img.height, img.width
            (3, 4, 5)
            >>> hwc = Image.from_numpy(np.ones((4, 5, 3), dtype=np.uint8))  # (H, W, C), CHANNELS_LAST
            >>> back = Image.from_dlpack(hwc.to_dlpack(), hwc.pixel_format.color_space, hwc.channels_order)
            >>> back.channels, back.height, back.width
            (3, 4, 5)

        """
        # check before consuming the capsule, which can be read only once
        if channels_order not in (ChannelsOrder.CHANNELS_FIRST, ChannelsOrder.CHANNELS_LAST):
            raise ValueError("channels_order must be either `CHANNELS_LAST` or `CHANNELS_FIRST`")
        _data: torch.Tensor = from_dlpack(data)

        if channels_order == ChannelsOrder.CHANNELS_FIRST:
            image_size = ImageSize(height=_data.shape[1], width=_data.shape[2])
            channels = _data.shape[0]
        else:
            image_size = ImageSize(height=_data.shape[0], width=_data.shape[1])
            channels = _data.shape[2]

        if color_space is None:
            color_space = _default_color_space(channels)
        pixel_format = PixelFormat(color_space=color_space, bit_depth=_data.element_size() * 8)

        # create the image layout based on the input data
        layout = ImageLayout(image_size=image_size, channels=channels, channels_order=channels_order)

        return cls(_data, pixel_format, layout)

    def to_dlpack(self) -> DLPack:
        """Return a DLPack capsule from the image torch.Tensor, sharing its memory.

        The capsule holds the data in the image's own :attr:`channels_order`, ``(C, H, W)`` or ``(H, W, C)``, and
        carries no color space; read it back with
        ``Image.from_dlpack(capsule, img.pixel_format.color_space, img.channels_order)``.
        """
        return to_dlpack(self.data)

    @classmethod
    def from_file(cls, file_path: str | Path) -> Image:
        """Construct an ``Image`` from a file.

        The file is loaded with :func:`~kornia.io.load_image` as ``ImageLoadType.RGB8``: the image is
        :math:`(3, H, W)` ``uint8``, ``RGB`` and ``CHANNELS_FIRST``, on the CPU.

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

        The format follows the file extension; see :func:`kornia.io.write_image` for the supported formats. A ``BGR``
        image is converted to RGB first, so the file holds the true colours.

        Args:
            file_path: the path to the file to write the image to.

        Example:
            >>> data = np.ones((4, 5, 3), dtype=np.uint8)  # HxWxC
            >>> img = Image.from_numpy(data)
            >>> import tempfile, os
            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     img.write(os.path.join(tmp, "test.jpg"))

        """
        data = self.to_rgb().data if self.pixel_format.color_space == ColorSpace.BGR else self.data
        if self.channels_order == ChannelsOrder.CHANNELS_LAST:
            data = data.permute(2, 0, 1)
        write_image(file_path, data)

    def print(self, max_width: int = 256) -> None:
        """Print the image torch.Tensor to the console.

        Args:
            max_width: the widest image, in pixels, that is printed without resizing; see
                :func:`~kornia.image.image_to_string`.

        A ``BGR`` image is converted to RGB first, so the terminal shows the true colours.

        .. code-block:: python

            img = Image.from_file("panda.png")
            img.print()

        .. image:: https://github.com/kornia/data/blob/main/print_image.png?raw=true

        """
        data = self.to_rgb().data if self.pixel_format.color_space == ColorSpace.BGR else self.data
        if self.channels_order == ChannelsOrder.CHANNELS_LAST:
            data = data.permute(2, 0, 1)
        print(image_to_string(data, max_width))  # noqa: T201 - Image.print prints by design
