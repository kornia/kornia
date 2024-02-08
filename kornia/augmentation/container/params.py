from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Union

from kornia.core import Tensor

__all__ = ["ParamItem", "PatchParamItem"]


class ParamItem(NamedTuple):
    name: str
    data: Optional[Union[Dict[str, Tensor], List["ParamItem"]]]


class PatchParamItem(NamedTuple):
    indices: List[int]
    param: ParamItem
