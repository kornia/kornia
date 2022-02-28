from enum import Enum
from typing import Any, Callable, Dict, List, Tuple, Union

from kornia.core import Tensor
from kornia.utils import image_to_tensor, tensor_to_image


class ImageColor(Enum):
    GRAY = 0
    RGB = 1
    BGR = 2


class Image(Tensor):
    # defaults
    _meta: Dict[str, Any] = {}
    _meta['color'] = ImageColor.RGB
    _meta['is_normalized'] = False
    _meta['mean'] = 0.0
    _meta['std'] = 255.0

    @staticmethod
    def __new__(cls, data: Tensor, color: ImageColor, is_normalized: bool, *args, **kwargs):
        return Tensor._make_subclass(cls, data, *args, **kwargs)

    def __init__(self, data: Tensor, color: ImageColor, is_normalized: bool) -> None:
        self._meta['color'] = color
        self._meta['is_normalized'] = is_normalized

    @property
    def is_normalized(self) -> bool:
        return self._meta['is_normalized']

    @is_normalized.setter
    def is_normalized(self, x: bool) -> None:
        self._meta['is_normalized'] = x

    def get_mean(self) -> Union[float, Tensor]:
        return self._meta['mean']

    def set_mean(self, x: Union[float, Tensor]) -> None:
        if not isinstance(x, (float, Tensor)):
            raise TypeError(f"Unsupported type {type(x)}.")
        self._meta['mean'] = x

    def get_std(self) -> Union[float, Tensor]:
        return self._meta['std']

    def set_std(self, x: Union[float, Tensor]) -> None:
        if not isinstance(x, (float, Tensor)):
            raise TypeError(f"Unsupported type {type(x)}.")
        self._meta['std'] = x

    @property
    def valid(self) -> bool:
        return self.data.data_ptr is not None

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

    @classmethod
    def from_tensor(cls, data: Tensor, color: ImageColor, is_normalized: bool) -> 'Image':
        return cls(data, color, is_normalized)

    @classmethod
    def from_numpy(cls, data, color: ImageColor, is_normalized: bool) -> 'Image':
        return cls(image_to_tensor(data), color, is_normalized)

    def to_numpy(self):
        return tensor_to_image(self.data, keepdim=True)

    @classmethod
    def from_list(cls, data: List[List[Union[float, int]]], color: ImageColor, is_normalized: bool) -> 'Image':
        return cls(Tensor(data), color, is_normalized)

    @classmethod
    def from_file(cls, file_path: str) -> 'Image':
        raise NotImplementedError("not implemented yet.")

    def apply(self, handle: Callable, *args, **kwargs) -> 'Image':
        return handle(self, *args, **kwargs)

    def normalize(self) -> 'Image':
        if self.is_normalized:
            return self
        data_norm = (self.data.float() - self.get_mean()) / self.get_std()
        return Image(data_norm, self.color, is_normalized=True)

    def denormalize(self) -> 'Image':
        if not self.is_normalized:
            return self
        data_denorm = (self.data * self.get_std()) + self.get_mean()
        return Image(data_denorm, self.color, is_normalized=False)
