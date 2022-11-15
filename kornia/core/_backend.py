from typing import Union

import torch
import torch.nn.functional as F
from torch import device

# classes
Tensor = torch.Tensor
tensor = torch.tensor
Module = torch.nn.Module
Parameter = torch.nn.Parameter

# functions
concatenate = torch.cat
stack = torch.stack
normalize = F.normalize
pad = F.pad
eye = torch.eye
zeros = torch.zeros
zeros_like = torch.zeros_like
where = torch.where
complex = torch.complex

# constructors
as_tensor = torch.as_tensor

# random
rand = torch.rand

# device
Device = Union[str, device]
