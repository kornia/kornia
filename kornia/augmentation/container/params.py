from typing import Dict, List, NamedTuple, Optional, Union

from kornia.core import Tensor

__all__ = ["ParamItem", "PatchParamItem"]


class ParamItem(NamedTuple):  # noqa: D101
    name: str
    data: Optional[Union[Dict[str, Tensor], List["ParamItem"]]]


class PatchParamItem(NamedTuple):  # noqa: D101
    indices: List[int]
    param: ParamItem
