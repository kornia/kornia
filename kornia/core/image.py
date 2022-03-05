from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from kornia.core import Tensor
from kornia.utils import image_to_tensor, tensor_to_image


# TODO: add more types
class ImageColor(Enum):
    GRAY = 0
    RGB = 1
    BGR = 2


class Image(Tensor):
    # defaults
    _meta: Dict[str, Any] = {}
    _meta['color'] = ImageColor.RGB
    _meta['mean'] = None
    _meta['std'] = None

    @staticmethod
    def __new__(
        cls,
        data: Tensor,
        color: ImageColor,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        *args,
        **kwargs
    ):
        return Tensor._make_subclass(cls, data, *args, **kwargs)

    def __init__(
        self, data: Tensor, color: ImageColor, mean: Optional[List[float]] = None, std: Optional[List[float]] = None
    ) -> None:
        self._meta['color'] = color
        self._meta['mean'] = mean
        self._meta['std'] = std

    @property
    def is_batch(self) -> bool:
        return len(self.data.shape) > 3

    @property
    def channels(self) -> int:
        return self.data.shape[-3]

    @property
    def height(self) -> int:
        return self.data.shape[-2]

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.height, self.width)

    @property
    def width(self) -> int:
        return self.data.shape[-1]

    @property
    def color(self) -> ImageColor:
        return self._meta['color']

    @color.setter
    def color(self, x: ImageColor) -> None:
        self._meta['color'] = x

    def _get_mean(self) -> List[float]:
        return self._meta['mean']

    def _get_std(self) -> List[float]:
        return self._meta['std']

    @classmethod
    def from_numpy(
        cls, data, color: ImageColor, mean: Optional[List[float]] = None, std: Optional[List[float]] = None
    ) -> 'Image':
        return cls(image_to_tensor(data), color, mean, std)

    def to_numpy(self):
        return tensor_to_image(self.data, keepdim=True)

    @classmethod
    def from_dlpack(
        cls, data, color: ImageColor, mean: Optional[List[float]] = None, std: Optional[List[float]] = None
    ):
        from torch.utils.dlpack import from_dlpack

        return cls(from_dlpack(data), color, mean, std)

    def to_dlpack(self):
        from torch.utils.dlpack import to_dlpack

        return to_dlpack(self)

    @classmethod
    def from_file(cls, file_path: str) -> 'Image':
        raise NotImplementedError("not implemented yet.")

    def apply(self, handle: Callable, *args, **kwargs) -> 'Image':
        return handle(self, *args, **kwargs)

    def denormalize(self) -> 'Image':
        if not self.is_floating_point():
            raise TypeError("Image must be in floating point.")

        if self._get_mean() is None or self._get_std() is None:
            return self

        def _make_tensor(data):
            data = Tensor(data, device=self.device).to(self.dtype)
            data = data.view(-1, 1, 1)
            return data[None] if self.is_batch else data

        # convert to tensor the mean and std
        mean = _make_tensor(self._get_mean())
        std = _make_tensor(self._get_std())

        data_denorm = (self.data * std) + mean
        return Image(data_denorm, self.color, self._get_mean(), self._get_std())
