import torch

from packaging import version

if version.parse(torch.__version__) > version.parse("1.7.1"):
    from torch.linalg import solve
else:
    from torch import solve as _solve

    # NOTE: in previous versions `torch.solve` accepted arguments in another order.
    def solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _solve(B, A).solution
