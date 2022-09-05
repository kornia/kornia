import torch
from torch import Tensor


def randperm(n: int, ensure_perm: bool = True, **kwargs) -> Tensor:
    """`randomperm` with the ability to ensure the different arrangement generated."""
    perm = torch.randperm(n, **kwargs)
    if ensure_perm:
        while torch.all(torch.eq(perm, torch.arange(n, device=perm.device))):
            perm = torch.randperm(n, **kwargs)
    return perm
