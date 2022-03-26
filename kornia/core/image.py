from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from kornia.core import Tensor
from kornia.utils import image_to_tensor, tensor_to_image


class ImageColor(Enum):
    r"""Enum that represents an image color space."""
    GRAY8 = 0
    GRAY32 = 1
    RGB8 = 2
    BGR8 = 3


class Image:
    r"""Class that holds an Image Tensor representation.

    The image tensor expects a data layout :math:`(B)(C, H, W)` and subclasses :class:`torch.Tensor`.
    That means that shares the same underlying functionality as if you were using a barebone tensor.
    Additionally, this class holds meta information to avoid common bugs during image manipulation.

    .. note::

        Disclaimer: This class provides the minimum functionality for image manipulation. However, as soon
        as you start to experiment with advanced tensor manipulation, you might expect fancy
        polymorphic behaviours.

    .. warning::

        This API is experimental and might suffer changes in the future.

    Args:
        data (torch.Tensor): a torch.Tensor with the shape of :math:`(B)(C, H, W)`.
        color (ImageColor): the color space representation of the input data.
        mean (List[float]): a list containing the mean values of the input data.
        std (List[float]): a list containing the standard deviation of the input data.

    Examples:
        >>> import kornia
        >>> import torch
        >>> import numpy as np
        >>>
        >>> # from a torch.tensor
        >>> data = torch.randint(0, 255, (3, 4, 5))  # CxHxW
        >>> img = Image(data, ImageColor.RGB)
        >>> assert img.channels == 3
        >>> assert img.resolution == (4, 5)
        >>> assert (img.data == data).all()
        >>> out = (img + 2 * img - 3 * img.clone()).sum()
        >>> assert isinstance(out, Image)

        >>> # from a bached torch.tensor
        >>> data = torch.rand(2, 3, 4, 5)  # BxCxHxW
        >>> img = Image(data, ImageColor.RGB)
        >>> assert img.channels == 3
        >>> assert img.resolution == (4, 5)
        >>> assert img.is_batch
        >>> r, g, b = img[:, 0], img[:, 1], img[:, 2]
        >>> mean_ch = img.mean(dim=1, keepdim=True)
        >>> mean_ch.shape
        torch.Size([2, 1, 4, 5])

        >>> # from a numpy array (like opencv)
        >>> data = np.ones((4, 5, 3), dtype=np.uint8)  # HxWxC
        >>> img = Image.from_numpy(data, color=ImageColor.BGR)
        >>> assert img.channels == 3
        >>> assert img.resolution == (4, 5)

        >>> # direct conversion from/to dlpack
        >>> data = torch.rand(3, 4, 5)
        >>> dlpack = Image(data, ImageColor.RGB).to_dlpack()
        >>> img = Image.from_dlpack(dlpack, ImageColor.RGB)

        >>> # apply any kornia function
        >>> data = torch.rand(3, 4, 5)
        >>> img = Image(data, color=ImageColor.RGB)
        >>> img_gray = img.apply(kornia.color.rgb_to_grayscale)
        >>> assert isinstance(img_gray, Image)
        >>> assert img_gray.color == ImageColor.GRAY

        >>> # denormalize for visualization
        >>> data = torch.rand(3, 4, 5)
        >>> img = Image(data, color=ImageColor.RGB, mean=[0.,0.,0], std=[255.,255.,255.])
        >>> img_denorm = img.denormalize()
        >>> assert img_denorm.mean().item() > 1.0
    """
    # defaults
    _meta: Dict[str, Any] = {}
    _meta['color'] = ImageColor.RGB8
    _meta['mean'] = None
    _meta['std'] = None

    def __init__(
        self, data: Tensor, color: ImageColor, mean: Optional[List[float]] = None, std: Optional[List[float]] = None
    ) -> None:
        self._data = data
        self._meta['color'] = color
        self._meta['mean'] = mean
        self._meta['std'] = std

    def __getattr__(self, name: str):
        """Direct access to the backend methods."""
        if name in ['apply']:
            return
        return getattr(self.data, name)

    def __getitem__(self, idx):
        return Image(self.data[idx], self.color, self._mean(), self._std())

    def __repr__(self) -> str:
        return f"Image data: {self.data}\nMeta: {self._meta}"

    def __len__(self) -> int:
        return len(self.data)

    def __floordiv__(self, x):
        if hasattr(x, 'data'):
            x = x.data
        return Image(self.data / x, self.color, self._mean(), self._std())

    def __truediv__(self, x):
        if hasattr(x, 'data'):
            x = x.data
        return Image(self.data // x, self.color, self._mean(), self._std())

    def __mul__(self, x):
        if hasattr(x, 'data'):
            x = x.data
        return Image(self.data * x, self.color, self._mean(), self._std())

    def __add__(self, x):
        if hasattr(x, 'data'):
            x = x.data
        return Image(self.data + x, self.color, self._mean(), self._std())

    def __sub__(self, x):
        if hasattr(x, 'data'):
            x = x.data
        return Image(self.data - x, self.color, self._mean(), self._std())

    def __mod__(self, x):
        if hasattr(x, 'data'):
            x = x.data
        return Image(self.data % x, self.color, self._mean(), self._std())

    def __pow__(self, x):
        if hasattr(x, 'data'):
            x = x.data
        return Image(self.data**x, self.color, self._mean(), self._std())

    def to(self, device=None, dtype=None) -> 'Image':
        if device is not None and isinstance(device, torch.dtype):
            dtype, device = device, None
        return Image(self.data.to(device, dtype), self.color, self._mean(), self._std())

    def clone(self) -> 'Image':
        return Image(self.data.clone(), self.color, self._mean(), self._std())

    @property
    def data(self) -> Tensor:
        """Return the underlying tensor data."""
        return self._data

    @property
    def is_batch(self) -> bool:
        """Return whether the image data is a batch of images."""
        return len(self.data.shape) > 3

    @property
    def channels(self) -> int:
        """Return the number channels of the image."""
        return self.data.shape[-3]

    @property
    def height(self) -> int:
        """Return the image height (columns)."""
        return self.data.shape[-2]

    @property
    def resolution(self) -> Tuple[int, int]:
        """Return a tuple with the image (height, width)."""
        return (self.height, self.width)

    @property
    def width(self) -> int:
        """Return the image width (rows)."""
        return self.data.shape[-1]

    @property
    def color(self) -> ImageColor:
        """Return the color space representation of the image."""
        return self._meta['color']

    @color.setter
    def color(self, x: ImageColor) -> None:
        """Setter for the color space representation of the image."""
        self._meta['color'] = x

    # TODO: figure out a better way map this function
    def float(self) -> 'Image':
        return Image(self.data.float(), self.color, self._mean(), self._std())

    def _mean(self) -> List[float]:
        return self._meta['mean']

    def _std(self) -> List[float]:
        return self._meta['std']

    @classmethod
    def from_numpy(
        cls, data, color: ImageColor, mean: Optional[List[float]] = None, std: Optional[List[float]] = None
    ) -> 'Image':
        """Construct an image tensor from a numpy array.

        Args:
            data: a numpy array with the shape of :math:`(H,W,C).
            color: the color space representation of the input data.
            mean: a list containing the mean values of the input data.
            std: a list containing the standard deviation of the input data.

        Example:
            >>> import numpy as np
            >>> data = np.ones((4, 5, 3), dtype=np.uint8)  # HxWxC
            >>> img = Image.from_numpy(data, color=ImageColor.BGR)
            >>> assert img.channels == 3
            >>> assert img.resolution == (4, 5)
            >>> assert img.shape == (3, 4, 5)
        """
        return cls(image_to_tensor(data), color, mean, std)

    def to_numpy(self):
        """Return a numpy array with the shape of :math:`(H,W,C)`."""
        return tensor_to_image(self.data, keepdim=True)

    @classmethod
    def from_dlpack(
        cls, data, color: ImageColor, mean: Optional[List[float]] = None, std: Optional[List[float]] = None
    ):
        """Construct an image tensor from a DLPack capsule.

        Args:
            data: a DLPack capsule from numpy, tvm or jax.
            color: the color space representation of the input data.
            mean: a list containing the mean values of the input data.
            std: a list containing the standard deviation of the input data.

        Example:
            >>> import numpy as np
            >>> x = np.ones((4, 5, 3))
            >>> img = Image.from_dlpack(x.__dlpack__(), ImageColor.RGB)
        """
        from torch.utils.dlpack import from_dlpack

        return cls(from_dlpack(data), color, mean, std)

    def to_dlpack(self):
        """Return a DLPack capsule from the image tensor."""
        from torch.utils.dlpack import to_dlpack

        return to_dlpack(self.data)

    @classmethod
    def from_file(cls, file_path: str) -> 'Image':
        """Construct an image tensor from a given file.

        .. warning::

            COMING SOON !
        """
        raise NotImplementedError("not implemented yet.")

    @classmethod
    def show(self) -> None:
        """Shows an image with the opengl vviz.

        .. warning::

            COMING SOON !
        """
        raise NotImplementedError("not implemented yet.")

    def apply(self, handle: Callable, *args, **kwargs) -> 'Image':
        """Apply a given kornia function to the image data.

        Args:
            handle: the callable function to apply to the image.
            args: the function arguments.
            kwargs: the function keyword arguments.

        Example:
            >>> import kornia
            >>> import torch
            >>> data = torch.rand(3, 4, 5)
            >>> img = Image(data, color=ImageColor.RGB)
            >>> img = img.apply(kornia.geometry.resize, (2, 2))
            >>> assert img.resolution == (2, 2)
        """
        return handle(self, *args, **kwargs)

    def denormalize(self) -> 'Image':
        """Denormalize the image data with given mean and std.

        The preconditions of this function are:

        1) The image data must be in floating point precision.

        2) The mean and std values have to be specified in the constructor.

        .. tip::

            This function is specially useful for visualization purposes.

        Example:
            >>> data = torch.rand(3, 4, 5)
            >>> img = Image(data, color=ImageColor.RGB, mean=[0.,0.,0], std=[255.,255.,255.])
            >>> img_denorm = img.denormalize()
            >>> assert img_denorm.mean().item() > 1.0
        """
        if not self.is_floating_point():
            raise TypeError("Image must be in floating point.")

        if self._mean() is None or self._std() is None:
            return self

        def _make_tensor(data):
            data = Tensor(data, device=self.device).to(self.dtype)
            data = data.view(-1, 1, 1)
            return data[None] if self.is_batch else data

        # convert to tensor the mean and std
        mean = _make_tensor(self._mean())
        std = _make_tensor(self._std())

        data_denorm = (self.data * std) + mean
        return Image(data_denorm, self.color, self._mean(), self._std())
