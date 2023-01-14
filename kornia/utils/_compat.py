from typing import TYPE_CHECKING, Callable, ContextManager, List, Optional, Tuple, TypeVar

import torch
from torch import Tensor

from packaging import version


def torch_version() -> str:
    """Parse the `torch.__version__` variable and removes +cu*/cpu."""
    return torch.__version__.split('+')[0]


# TODO: replace by torch_version_ge``
def torch_version_geq(major, minor) -> bool:
    _version = version.parse(torch_version())
    return _version >= version.parse(f"{major}.{minor}")


def torch_version_lt(major: int, minor: int, patch: int) -> bool:
    _version = version.parse(torch_version())
    return _version < version.parse(f"{major}.{minor}.{patch}")


def torch_version_le(major: int, minor: int, patch: int) -> bool:
    _version = version.parse(torch_version())
    return _version <= version.parse(f"{major}.{minor}.{patch}")


def torch_version_ge(major: int, minor: int, patch: int) -> bool:
    _version = version.parse(torch_version())
    return _version >= version.parse(f"{major}.{minor}.{patch}")


if TYPE_CHECKING:
    # TODO: remove this branch when kornia relies on torch >= 1.10.0
    def torch_meshgrid(tensors: List[Tensor], indexing: Optional[str] = None) -> Tuple[Tensor, ...]:
        ...

else:
    if torch_version_ge(1, 10, 0):

        def torch_meshgrid(tensors: List[Tensor], indexing: str):
            return torch.meshgrid(tensors, indexing=indexing)

    else:
        # TODO: remove this branch when kornia relies on torch >= 1.10.0
        def torch_meshgrid(tensors: List[Tensor], indexing: str):
            return torch.meshgrid(tensors)


if TYPE_CHECKING:
    # TODO: remove this branch when kornia relies on torch >= 1.10.0
    _T = TypeVar('_T')
    torch_inference_mode: Callable[..., ContextManager[_T]]
else:
    if torch_version_ge(1, 10, 0):
        torch_inference_mode = torch.inference_mode
    else:
        # TODO: remove this branch when kornia relies on torch >= 1.10.0
        torch_inference_mode = torch.no_grad
