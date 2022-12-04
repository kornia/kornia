from typing import TYPE_CHECKING, Union

import torch
import torch.nn.functional as F
from torch import device, dtype

# Types
Device = Union[device, str, None]
Tensor = torch.Tensor
Dtype = Union[dtype, None]

# classes
tensor = torch.tensor
Size = torch.Size
Module = torch.nn.Module
Parameter = torch.nn.Parameter

# functions
concatenate = torch.cat
stack = torch.stack
normalize = F.normalize
pad = F.pad
eye = torch.eye
where = torch.where
complex = torch.complex

if TYPE_CHECKING:
    from ._backend_typing import as_tensor, ones, rand, zeros, zeros_like
else:
    # constructors
    as_tensor = torch.as_tensor

    # functions
    zeros = torch.zeros
    ones = torch.ones
    zeros_like = torch.zeros_like

    # random
    rand = torch.rand
