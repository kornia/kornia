from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from torch import Tensor

from packaging import version


def torch_version() -> str:
    """Parse the `torch.__version__` variable and removes +cu*/cpu."""
    return torch.__version__.split('+')[0]


def torch_version_geq(major, minor) -> bool:
    _version = version.parse(torch_version())
    return _version >= version.parse(f"{major}.{minor}")


if version.parse(torch_version()) > version.parse("1.7.1"):
    # TODO: remove the type: ignore once Python 3.6 is deprecated.
    # It turns out that Pytorch has no attribute `torch.linalg` for
    # Python 3.6 / PyTorch 1.7.0, 1.7.1
    from torch.linalg import solve  # type: ignore
else:
    from torch import solve as _solve

    # NOTE: in previous versions `torch.solve` accepted arguments in another order.
    def solve(A: Tensor, B: Tensor) -> Tensor:
        return _solve(B, A).solution


if version.parse(torch_version()) > version.parse("1.7.1"):
    # TODO: remove the type: ignore once Python 3.6 is deprecated.
    # It turns out that Pytorch has no attribute `torch.linalg` for
    # Python 3.6 / PyTorch 1.7.0, 1.7.1
    from torch.linalg import qr as linalg_qr  # type: ignore
else:
    from torch import qr as linalg_qr  # type: ignore # noqa: F401


# TODO: remove after deprecating < 1.9.1
if version.parse(torch_version()) > version.parse("1.9.1"):

    if not TYPE_CHECKING:

        def torch_meshgrid(tensors: List[Tensor], indexing: str):
            return torch.meshgrid(tensors, indexing=indexing)

else:

    if TYPE_CHECKING:

        def torch_meshgrid(tensors: List[Tensor], indexing: Optional[str] = None) -> Tuple[Tensor, ...]:
            return torch.meshgrid(tensors)

    else:

        def torch_meshgrid(tensors: List[Tensor], indexing: str):
            return torch.meshgrid(tensors)
