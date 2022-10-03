from typing import List, Union

from kornia.core import Tensor

ImagePaths = List[str]
ImageTensors = List[Tensor]
Images = Union[ImagePaths, ImageTensors]
