from typing import TYPE_CHECKING, List, Optional, Tuple, Union

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


if version.parse(torch_version()) > version.parse("1.9.1"):
    from torch import meshgrid as torch_meshgrid  # type: ignore
else:
    from torch import meshgrid as _meshgrid

    if TYPE_CHECKING:
        # The JIT doesn't understand Union, so only add type annotation for mypy
        def torch_meshgrid(*tensors: Union[Tensor, List[Tensor]]) -> Tuple[Tensor, ...]:
            return _meshgrid(*tensors)

    else:
        # NOTE: the typing below has been modified to make happy torchscript
        def torch_meshgrid(tensors: List[Tensor], indexing: Optional[str] = None) -> List[Tensor]:
            return _meshgrid(tensors)
