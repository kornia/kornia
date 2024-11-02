from typing import NamedTuple, Optional, Union

from kornia.core import Tensor

__all__ = ["ParamItem", "PatchParamItem"]


class ParamItem(NamedTuple):
    name: str
    data: Optional[Union[dict[str, Tensor], list["ParamItem"]]]


class PatchParamItem(NamedTuple):
    indices: list[int]
    param: ParamItem
