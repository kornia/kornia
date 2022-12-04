from typing import Any, Generator, List, Optional, Sequence, Tuple, Union, overload

from torch import layout

from ._backend import Device, Dtype, Size, Tensor

_size = Union[Size, List[int], Tuple[int, ...]]

# The idea here is to overwrite the typing for old torch versions that don't have dtype and device types with None
# TODO: Keep this update based on the last version of pytorch without break compatibility
# From last version of pytorch 1.13.0: "torch/_C/_VariableFunctions.pyi"
__all__ = ['zeros', 'ones', 'as_tensor', 'rand', 'zeros_like']

class SymInt: ...

# Defined in torch/csrc/MemoryFormat.cpp
class memory_format: ...

@overload
def zeros(
    size: Sequence[Union[int, SymInt]],
    *,
    out: Optional[Tensor] = None,
    dtype: Dtype = None,
    layout: Optional[layout] = None,
    device: Device = None,
    pin_memory: Optional[bool] = False,
    requires_grad: Optional[bool] = False
) -> Tensor: ...
@overload
def zeros(
    *size: int,
    out: Optional[Tensor] = None,
    dtype: Dtype = None,
    layout: Optional[layout] = None,
    device: Device = None,
    pin_memory: Optional[bool] = False,
    requires_grad: Optional[bool] = False
) -> Tensor: ...
def zeros_like(
    input: Tensor,
    *,
    memory_format: Optional[memory_format] = None,
    dtype: Dtype = None,
    layout: Optional[layout] = None,
    device: Device = None,
    pin_memory: Optional[bool] = False,
    requires_grad: Optional[bool] = False
) -> Tensor: ...
@overload
def ones(
    size: Sequence[Union[int, SymInt]],
    *,
    out: Optional[Tensor] = None,
    dtype: Dtype = None,
    layout: Optional[layout] = None,
    device: Device = None,
    pin_memory: Optional[bool] = False,
    requires_grad: Optional[bool] = False
) -> Tensor: ...
@overload
def ones(
    *size: int,
    out: Optional[Tensor] = None,
    dtype: Dtype = None,
    layout: Optional[layout] = None,
    device: Device = None,
    pin_memory: Optional[bool] = False,
    requires_grad: Optional[bool] = False
) -> Tensor: ...
@overload
def rand(
    size: _size,
    *,
    out: Optional[Tensor] = None,
    dtype: Dtype = None,
    layout: Optional[layout] = None,
    device: Device = None,
    pin_memory: Optional[bool] = False,
    requires_grad: Optional[bool] = False
) -> Tensor: ...
@overload
def rand(
    *size: int,
    out: Optional[Tensor] = None,
    dtype: Dtype = None,
    layout: Optional[layout] = None,
    device: Device = None,
    pin_memory: Optional[bool] = False,
    requires_grad: Optional[bool] = False
) -> Tensor: ...
def as_tensor(data: Any, dtype: Dtype = None, device: Device = None) -> Tensor: ...
