from typing import List, Union

from torch import Tensor, device

Device = Union[str, device]

ImagePaths = List[str]
ImageTensors = List[Tensor]
Images = Union[ImagePaths, ImageTensors]
