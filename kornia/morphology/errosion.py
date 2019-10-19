import torch
import torch.nn as nn

class Errosion(nn.Module):
    def __init__(self):
        super(Dilation, self).__init__()

    def forward(self):
        return errosion()

def errosion(img: torch.Tensor, filter: torch.tensor):
    if not torch.is_tensor(img) or not torch.is_tensor(filter):
        raise TypeError 
    conv1 = nn.Conv2d(img, filter)
    clipped_res = torch.clamp(conv1, min = 0, max = 1)
    return clipped_res