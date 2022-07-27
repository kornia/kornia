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
