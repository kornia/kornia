from dataclasses import dataclass
from typing import Union

from kornia.core import Tensor


@dataclass
class ImageSize:
    height: Union[int, Tensor]
    width: Union[int, Tensor]
