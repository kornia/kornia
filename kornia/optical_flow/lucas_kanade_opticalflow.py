import torch
import torch.nn as nn

from kornia.filters import spatial_gradient


def optical_flow_lk(image_prev: torch.Tensor, image_next: torch.Tensor, window_size: int) -> torch.Tensor:

    r"""Calculate an optical flow for an input pair of images.

    Args:
        image_prev: the input tensor with shape :math:`(B,C,H,W)`.
        image_next: the size of the kernel.
        window_size: the size of the window.



    Returns:
        stacked tensor with shape :math:`(B,2,H,W)`.

    Examples:


    """
    if(window_size%2==0):
        return "Expected an odd number for window size, got an even number"
    filter_prev=spatial_gradient(input=image_prev, mode='oflk', order = 1)
    filter_next=spatial_gradient(input=image_next, mode='oflk', order = 1)

    fx= 0.5 * (filter_prev[0][0] + filter_next[0][0])
    fy = 0.5 * (filter_prev[0][1] + filter_next[0][1])
    ft = filter_prev[0][2] + filter_next[0][3]
    n = window_size//2
    pad = nn.ZeroPad2d(n)
    resx = pad(fx)
    resy = pad(fy)
    resdt = pad(ft)
    u = torch.zeros(image_prev[0][0].shape)
    v = torch.zeros(image_prev[0][0].shape)
    for i in range(n, len(resx)-n):
        for j in range(n, len(resx[0])-n):
            px = resx[i - n:i+n+1, j - n:j+n+1]
            px = torch.flatten(px)
            py = resy[i - n:i+n+1, j - n:j+n+1]
            py = torch.flatten(py)
            pn = resdt[i - n:i+n+1, j - n:j+n+1]
            a = torch.stack([px, py])
            b = torch.flatten(pn)
            tr = a.t()
            if torch.det(torch.matmul(a, tr)) != 0:
                rightprod = torch.matmul(b, tr)
                leftprod = torch.matmul(a, tr)
                inv = torch.inverse(leftprod)
                res = torch.matmul(inv, rightprod)
                u[i-n][j-n] = res[0]
                v[i-n][j-n] = res[1]
    return torch.stack([u, v]).unsqueeze(0)

class OpticalFlowLK(nn.Module):
    r"""Calculate an optical flow for an input pair of images.

    Args:
        image_prev: the input tensor with shape :math:`(B,C,H,W)`.
        image_next: the size of the kernel.
        window_size: the size of the window.




    Returns:
        stacked tensor with shape :math:`(B,2,H,W)`.

    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, image_prev: torch.Tensor, image_next: torch.Tensor, window_size: int) -> torch.Tensor:
        return optical_flow_lk(image_prev, image_next, window_size)
