from enum import Enum
from typing import List, Tuple, Union

from kornia.color import bgr_to_grayscale, bgr_to_rgb, rgb_to_bgr, rgb_to_grayscale
from kornia.core import Tensor
from kornia.geometry.transform import hflip, vflip
from kornia.utils import image_to_tensor, tensor_to_image


class ImageColor(Enum):
    GRAY = 0
    RGB = 1
    BGR = 2


class Image(Tensor):
    _color = ImageColor.RGB

    @staticmethod
    def __new__(cls, data, color, *args, **kwargs):
        return Tensor._make_subclass(cls, data)

    def __init__(self, data: Tensor, color: ImageColor) -> None:
        self._color = color

    @property
    def valid(self) -> bool:
        # TODO: find a better way to do this
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
    def resolution(self) -> Tuple[int, ...]:
        return tuple(self.data.shape[-2:])

    @property
    def width(self) -> int:
        return self.data.shape[-1]

    @property
    def color(self) -> ImageColor:
        return self._color

    @classmethod
    def from_tensor(cls, data: Tensor, color: ImageColor) -> 'Image':
        return cls(data, color)

    # TODO: possibly call torch.as_tensor
    @classmethod
    def from_numpy(cls, data, color: ImageColor = ImageColor.RGB) -> 'Image':
        data_t: Tensor = image_to_tensor(data)
        return cls(data_t, color)

    def to_numpy(self):
        return tensor_to_image(self.data, keepdim=True)

    # TODO: possibly call torch.as_tensor
    @classmethod
    def from_list(cls, data: List[List[Union[float, int]]], color: ImageColor) -> 'Image':
        return cls(Tensor(data), color)

    @classmethod
    def from_file(cls, file_path: str) -> 'Image':
        raise NotImplementedError("not implemented yet.")

    # TODO: implement with another logic
    def _to_grayscale(self, data: Tensor) -> Tensor:
        if self.color == ImageColor.GRAY:
            out = data
        elif self.color == ImageColor.RGB:
            out = rgb_to_grayscale(data)
        elif self.color == ImageColor.BGR:
            out = bgr_to_grayscale(data)
        else:
            raise NotImplementedError(f"Unsupported color: {self.color}.")
        return out

    def grayscale(self) -> 'Image':
        gray = self._to_grayscale(self.data)
        return Image(gray, ImageColor.GRAY)

    def grayscale_(self) -> 'Image':
        self.data = self._to_grayscale(self.data)
        self._color = ImageColor.GRAY
        return self

    def _convert(self, color: ImageColor) -> Tuple[Tensor, ImageColor]:
        if color == ImageColor.RGB:
            if self.color == ImageColor.BGR:
                data_out = bgr_to_rgb(self.data)
                color_out = ImageColor.RGB
        elif color == ImageColor.BGR:
            if self.color == ImageColor.RGB:
                data_out = rgb_to_bgr(self.data)
                color_out = ImageColor.BGR
        elif color == ImageColor.GRAY:
            if self.color == ImageColor.BGR:
                data_out = bgr_to_grayscale(self.data)
                color_out = ImageColor.GRAY
            elif self.color == ImageColor.RGB:
                data_out = rgb_to_grayscale(self.data)
                color_out = ImageColor.GRAY
        else:
            raise NotImplementedError(f"Unsupported color: {self.color}.")
        return data_out, color_out

    def convert(self, color: ImageColor) -> 'Image':
        data, color = self._convert(color)
        return Image(data, color)

    def convert_(self, color: ImageColor) -> 'Image':
        self.data, self._color = self._convert(color)
        return self

    def hflip(self) -> 'Image':
        data = hflip(self.data)
        return Image(data, self.color)

    def vflip(self) -> 'Image':
        data = vflip(self.data)
        return Image(data, self.color)

    # TODO: add the methods we need
    # - lab, hsv, ...
    # - erode, dilate, ...
