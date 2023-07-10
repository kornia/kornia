from __future__ import annotations

import torch
from torch.utils.dlpack import from_dlpack, to_dlpack

from kornia.core import Tensor
from kornia.image.base import ChannelsOrder, ImageLayout, ImageSize, PixelFormat

# TODO: move this utils to kornia.image.conversions
from kornia.utils.image import image_to_tensor, tensor_to_image


class Image:
    r"""Class that holds an Image Tensor representation.

    .. note::

        Disclaimer: This class provides the minimum functionality for image manipulation. However, as soon
        as you start to experiment with advanced tensor manipulation, you might expect fancy
        polymorphic behaviours.

    .. warning::

        This API is experimental and might suffer changes in the future.

    Args:
        data: a torch tensor containing the image data.
        layout: a dataclass containing the image layout information.

    Examples:
        >>> import kornia
        >>> import torch
        >>> import numpy as np
        >>>
        >>> # from a torch.tensor
        >>> data = torch.randint(0, 255, (3, 4, 5))  # CxHxW
        >>> layout = ImageLayout(
        ...     image_size=ImageSize(4, 5),
        ...     channels=3,
        ...     pixel_format=PixelFormat.RGB,
        ...     channels_order=ChannelsOrder.CHANNEL_FIRST,
        ... )
        >>> img = Image(data, layout)
        >>> assert img.channels == 3

        >>> # from a numpy array (like opencv)
        >>> data = np.ones((4, 5, 3), dtype=np.uint8)  # HxWxC
        >>> img = Image.from_numpy(data, pixel_format=PixelFormat.BGR)
        >>> assert img.channels == 3
        >>> assert img.width == 5
        >>> assert img.height == 4
    """

    def __init__(self, data: Tensor, layout: ImageLayout) -> None:
        """Image constructor.

        Args:
            data: a torch tensor containing the image data.
            layout: a dataclass containing the image layout information.
        """
        self._data = data
        self._layout = layout

    def __repr__(self) -> str:
        return f"Image data: {self.data}\nLayout: {self.layout}"

    # TODO: explore use TensorWrapper
    def to(self, device=None, dtype=None) -> Image:
        if device is not None and isinstance(device, torch.dtype):
            dtype, device = device, None
        return Image(self.data.to(device, dtype), self.layout)

    # TODO: explore use TensorWrapper
    def clone(self) -> Image:
        return Image(self.data.clone(), self.layout)

    @property
    def data(self) -> Tensor:
        """Return the underlying tensor data."""
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
        return self.layout.image_size.height

    @property
    def width(self) -> int:
        """Return the image width (rows)."""
        return self.layout.image_size.width

    @property
    def pixel_format(self) -> PixelFormat:
        """Return the pixel format."""
        return self.layout.pixel_format

    @property
    def channels_order(self) -> ChannelsOrder:
        """Return the channels order."""
        return self.layout.channels_order

    # TODO: figure out a better way map this function
    def float(self) -> Image:
        return Image(self.data.float(), self.layout)

    @classmethod
    def from_numpy(
        cls,
        data,
        channels_order: ChannelsOrder = ChannelsOrder.CHANNEL_LAST,
        pixel_format: PixelFormat = PixelFormat.RGB,
    ) -> Image:
        """Construct an image tensor from a numpy array.

        Args:
            data: a numpy array with the shape of :math:`(H,W,C).
            channels_order: the channel order of the image.
            pixel_format: the pixel format of the image.

        Example:
            >>> import numpy as np
            >>> data = np.ones((4, 5, 3), dtype=np.uint8)  # HxWxC
            >>> img = Image.from_numpy(data)
            >>> assert img.channels == 3
            >>> assert img.shape == (3, 4, 5)
        """
        if channels_order == ChannelsOrder.CHANNEL_LAST:
            _data = image_to_tensor(data)
            image_size = ImageSize(height=data.shape[0], width=data.shape[1])
            channels = data.shape[2]
        elif channels_order == ChannelsOrder.CHANNEL_FIRST:
            _data = torch.from_numpy(data)
            image_size = ImageSize(height=data.shape[1], width=data.shape[2])
            channels = data.shape[0]
        else:
            raise ValueError("channels_order must be either CHANNEL_LAST or CHANNEL")

        # create the image layout based on the input data
        layout = ImageLayout(
            image_size=image_size, channels=channels, pixel_format=pixel_format, channels_order=channels_order
        )

        # create the image tensor
        return cls(_data, layout)

    def to_numpy(self):
        """Return a numpy array with the shape of :math:`(H,W,C)`."""
        return tensor_to_image(self.data, keepdim=True)

    @classmethod
    def from_dlpack(cls, data) -> Image:
        """Construct an image tensor from a DLPack capsule.

        Args:
            data: a DLPack capsule from numpy, tvm or jax.

        Example:
            >>> import numpy as np
            >>> x = np.ones((4, 5, 3))
            >>> img = Image.from_dlpack(x.__dlpack__())
        """
        _data = from_dlpack(data)

        # create the image layout based on the input data
        layout = ImageLayout(
            image_size=ImageSize(height=_data.shape[1], width=_data.shape[2]),
            channels=_data.shape[0],
            pixel_format=PixelFormat.RGB,
            channels_order=ChannelsOrder.CHANNEL_FIRST,
        )

        return cls(_data, layout)

    def to_dlpack(self):
        """Return a DLPack capsule from the image tensor."""
        return to_dlpack(self.data)

    @classmethod
    def from_file(cls, file_path: str) -> Image:
        """Construct an image tensor from a given file.

        .. warning::

            COMING SOON !
        """
        raise NotImplementedError("not implemented yet.")
