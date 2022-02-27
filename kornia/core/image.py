from enum import Enum
from typing import Callable, List, Tuple, Union

from kornia.core import Tensor
from kornia.utils import image_to_tensor, tensor_to_image


class ImageColor(Enum):
    GRAY = 0
    RGB = 1
    BGR = 2


class Image(Tensor):
    _color = ImageColor.RGB
    _is_normalized = False

    @staticmethod
    def __new__(cls, data, color, *args, **kwargs):
        return Tensor._make_subclass(cls, data)

    def __init__(self, data: Tensor, color: ImageColor) -> None:
        self._color = color

        # TODO: need to propagate metadata
        self._is_normalized: bool = False
        self._mean: Union[float, Tensor] = 0.0  # or tensor
        self._std: Union[float, Tensor] = 255.0  # or tensor

    @property
    def is_normalized(self) -> bool:
        return self._is_normalized

    @is_normalized.setter
    def is_normalized(self, x: bool) -> None:
        self._is_normalized = x

    def get_mean(self) -> Union[float, Tensor]:
        return self._mean

    def set_mean(self, x: Union[float, Tensor]) -> None:
        if not isinstance(x, (float, Tensor)):
            raise TypeError(f"Unsupported type {type(x)}.")
        self._mean = x

    def get_std(self) -> Union[float, Tensor]:
        return self._std

    def set_std(self, x: Union[float, Tensor]) -> None:
        if not isinstance(x, (float, Tensor)):
            raise TypeError(f"Unsupported type {type(x)}.")
        self._std = x

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
        return tuple(self.data.shape[-2:])

    @property
    def width(self) -> int:
        return self.data.shape[-1]

    @property
    def color(self) -> ImageColor:
        return self._color

    @color.setter
    def color(self, x: ImageColor) -> None:
        self._color = x

    @classmethod
    def from_tensor(cls, data: Tensor, color: ImageColor) -> 'Image':
        return cls(data, color)

    @classmethod
    def from_numpy(cls, data, color: ImageColor = ImageColor.RGB) -> 'Image':
        return cls(image_to_tensor(data), color)

    def to_numpy(self):
        return tensor_to_image(self.data, keepdim=True)

    @classmethod
    def from_list(cls, data: List[List[Union[float, int]]], color: ImageColor) -> 'Image':
        return cls(Tensor(data), color)

    @classmethod
    def from_file(cls, file_path: str) -> 'Image':
        raise NotImplementedError("not implemented yet.")

    def apply(self, handle: Callable, *args, **kwargs) -> 'Image':
        return handle(self, *args, **kwargs)

    def normalize(self) -> 'Image':
        if self._is_normalized:
            return self
        data_norm = (self.data.float() - self._mean) / self._std
        img_new = Image(data_norm, self.color)
        img_new._is_normalized = True
        return img_new

    def denormalize(self) -> 'Image':
        import pdb

        pdb.set_trace()
        if not self._is_normalized:
            return self

        data_norm = (self.data * self._std) + self._mean
        img_new = Image(data_norm, self.color)
        img_new._is_normalized = False
        return img_new
