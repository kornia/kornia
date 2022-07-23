import torch
import torch.nn.functional as F

# classes
Tensor = torch.Tensor
Module = torch.nn.Module
Parameter = torch.nn.Parameter

# functions
concatenate = torch.cat
stack = torch.stack
normalize = F.normalize

# constructors
as_tensor = torch.as_tensor

# math
cos = torch.cos
sin = torch.sin
sqrt = torch.sqrt

# random
rand = torch.rand
