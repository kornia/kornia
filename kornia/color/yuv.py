
import torch
import torch.nn as nn
import torch.nn.functional as F


class RGBToYUV(nn.Module):
    r"""
        TODO
    """

    def __init__(self) -> None:
        super(RGBToYUV, self).__init__()

    def forward(  # type: ignore
            self, input: torch.Tensor) -> torch.Tensor:
        return rgb_to_yuv(input)

def rgb_to_yuv(input: torch.Tensor):
    r, g, b = torch.chunk(input, chunks=3, dim=-3)
    y = 0.299*r + 0.587*g + 0.114*b
    u = -0.14713*r -0.28886*g + 0.436*b
    v = 0.615*r -0.51499*g - 0.10001*b
    return torch.cat((y,u,v), 1)




class YUVToRGB(nn.Module):
    r"""
        TODO
    """

    def __init__(self) -> None:
        super(YUVtoRGB, self).__init__()

    def forward(  # type: ignore
            self, input: torch.Tensor) -> torch.Tensor:
        return rgb_to_yuv(input)

def yuv_to_rgb(input: torch.Tensor):
    y, u, v = torch.chunk(input, chunks=3, dim=-3)
    r = r + 1.14*b # coefficient for g is 0
    g = r + -0.396*g - 0.581*b
    b = r + 2.029*g # coefficient for b is 0
    return torch.cat((r,g,b), 1)
