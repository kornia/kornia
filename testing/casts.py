from __future__ import annotations

from typing import Any, TypeVar

from kornia.core import Device, Dtype, Tensor

T = TypeVar("T")


def dict_to(data: dict[T, Any], device: Device, dtype: Dtype) -> dict[T, Any]:
    out: dict[T, Any] = {}
    for key, val in data.items():
        out[key] = val.to(device, dtype) if isinstance(val, Tensor) else val
    return out
