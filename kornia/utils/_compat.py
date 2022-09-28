from __future__ import annotations

from typing import TYPE_CHECKING

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


from torch import qr as linalg_qr


if torch_version_ge(1, 10, 0):

    if not TYPE_CHECKING:

        def torch_meshgrid(tensors: list[Tensor], indexing: str):
            return torch.meshgrid(tensors, indexing=indexing)

else:

    if TYPE_CHECKING:

        def torch_meshgrid(tensors: list[Tensor], indexing: str | None = None) -> tuple[Tensor, ...]:
            return torch.meshgrid(tensors)

    else:

        def torch_meshgrid(tensors: list[Tensor], indexing: str):
            return torch.meshgrid(tensors)
