from __future__ import annotations

import torch

# classes
Tensor = torch.Tensor
tensor = torch.tensor
Module = torch.nn.Module
Parameter = torch.nn.Parameter

# functions
# NOTE: ideally we expose what we find in numpy
concatenate = torch.cat
stack = torch.stack
linspace = torch.linspace
normalize = torch.nn.functional.normalize
pad = torch.nn.functional.pad
eye = torch.eye
zeros = torch.zeros
zeros_like = torch.zeros_like
where = torch.where
complex = torch.complex
diag = torch.diag

# constructors
as_tensor = torch.as_tensor

# random
rand = torch.rand

# type alias
Device = str | torch.device | None
Dtype = torch.dtype | None
