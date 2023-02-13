from typing import Dict, List, NamedTuple, Optional, Union

from kornia.core import Tensor

__all__ = ["ParamItem", "PatchParamItem"]


class ParamItem(NamedTuple):
    name: str
    # TODO: add type List['ParamItem'] when mypy > 0.991 be available (see python/mypy#14200)
    data: Optional[Union[Dict[str, Tensor], List]]  # type: ignore [type-arg]


class PatchParamItem(NamedTuple):
    indices: List[int]
    param: ParamItem
