import torch
import torch.nn.functional as F
import numpy as np

# classes
Tensor = torch.Tensor
Module = torch.nn.Module
Parameter = torch.nn.Parameter

# functions
# concatenate = torch.cat
stack = torch.stack
normalize = F.normalize
# zeros_like = torch.zeros_like
zeros = torch.zeros
where = torch.where

# constructors
as_tensor = torch.as_tensor

# random
rand = torch.rand

def concatenate(tensors, dim=0, out=None):
    """Concatenates the given sequence of seq tensors or ndarrays alongside the specified axis.

    Args:
        tensors: any python sequence of torch.tensors or np.ndarrays.
    """
    if all(isinstance(x, np.ndarray) for x in tensors):
        return np.concatenate(tensors, axis=dim, out=out)
    else:
        return torch.cat(tensors, dim=dim, out=out)


def zeros_like(input, dtype=None, device=None):
    """Returns a tensor filled with the scalar value 0, with the same size as input.

    Args:
        input: torch.Tensor or np.ndarray
    """
    if (isinstance(input, np.ndarray)):
        return np.zeros_like(input, dtype=dtype)
    else:
        return torch.zeros_like(input, dtype=dtype, device=device)
