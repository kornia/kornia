import torch
import torch.nn as nn

class Dilation(nn.Module):
    def __init__(self) -> None:
        super(Dilation, self).__init__()

    def forward(self):
        return dilation(img, filter)
        
 

def dilation(img: torch.Tensor, filter: torch.Tensor):
    if not torch.is_tensor(img) or not torch.is_tensor(filter):
        raise TypeError 
    conv1 = nn.Conv2d(img, filter, padding=1)
    clipped_res = torch.clamp(conv1, min = 0, max = 1)

    return clipped_res