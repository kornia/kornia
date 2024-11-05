from typing import TYPE_CHECKING, Any, Callable, ContextManager, List, Optional, Tuple, TypeVar

import torch
from packaging import version
from torch import Tensor


def torch_version() -> str:
    """Parse the `torch.__version__` variable and removes +cu*/cpu."""
    return torch.__version__.split("+")[0]


def torch_version_lt(major: int, minor: int, patch: int) -> bool:
    _version = version.parse(torch_version())
    return _version < version.parse(f"{major}.{minor}.{patch}")


def torch_version_le(major: int, minor: int, patch: int) -> bool:
    _version = version.parse(torch_version())
    return _version <= version.parse(f"{major}.{minor}.{patch}")


def torch_version_ge(major: int, minor: int, patch: Optional[int] = None) -> bool:
    _version = version.parse(torch_version())
    if patch is None:
        return _version >= version.parse(f"{major}.{minor}")
    else:
        return _version >= version.parse(f"{major}.{minor}.{patch}")


if TYPE_CHECKING:
    # TODO: remove this branch when kornia relies on torch >= 1.10.0
    def torch_meshgrid(tensors: List[Tensor], indexing: Optional[str] = None) -> Tuple[Tensor, ...]: ...

elif torch_version_ge(1, 10, 0):

    def torch_meshgrid(tensors: List[Tensor], indexing: str):
        return torch.meshgrid(tensors, indexing=indexing)

else:
    # TODO: remove this branch when kornia relies on torch >= 1.10.0
    def torch_meshgrid(tensors: List[Tensor], indexing: str):
        return torch.meshgrid(tensors)


if TYPE_CHECKING:
    # TODO: remove this branch when kornia relies on torch >= 1.10.0
    _T = TypeVar("_T")
    torch_inference_mode: Callable[..., ContextManager[_T]]
elif torch_version_ge(1, 10, 0):
    torch_inference_mode = torch.inference_mode
else:
    # TODO: remove this branch when kornia relies on torch >= 1.10.0
    torch_inference_mode = torch.no_grad

if TYPE_CHECKING:  # TODO (@johnnv1): remove this branch when bump the pytorch CI to support torch 2.4
    custom_fwd: Callable[..., Any]
    autocast: Callable[..., Any]
elif torch_version_ge(2, 4):
    from functools import partial

    from torch.amp import autocast as _autocast
    from torch.amp import custom_fwd as _custom_fwd

    custom_fwd = partial(_custom_fwd, device_type="cuda")
    autocast = partial(_autocast, "cuda")

else:
    custom_fwd = torch.cuda.amp.custom_fwd
    autocast = torch.cuda.amp.autocast
