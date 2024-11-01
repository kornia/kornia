from typing import Union

from kornia.core import Tensor

ImagePaths = list[str]
ImageTensors = list[Tensor]
Images = Union[ImagePaths, ImageTensors]
