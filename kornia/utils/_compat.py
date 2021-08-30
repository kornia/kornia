import torch

from packaging import version

if version.parse(torch.__version__) > version.parse("1.7.1"):
    # TODO: remove the type: ignore once Python 3.6 is deprecated.
    # It turns out that Pytorch has no attribute `torch.linalg` for
    # Python 3.6 / PyTorch 1.7.0, 1.7.1
    from torch.linalg import solve  # type: ignore
else:
    from torch import solve as _solve

    # NOTE: in previous versions `torch.solve` accepted arguments in another order.
    def solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _solve(B, A).solution
