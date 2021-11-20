import torch
import torch.nn as nn
from kornia.filters import filter2d


def optical_flow_lk(image_prev: torch.Tensor, image_next: torch.Tensor) -> torch.Tensor:
    r"""Calculate an optical flow for an input pair of images.

    Args:
        image_prev: the input tensor with shape :math:`(B,C,H,W)`.
        image_next: the size of the kernel.



    Returns:
        stacked tensor with shape :math:`(B,2,H,W)`.

    """
    fx1 = torch.Tensor([[-1, 1], [-1, 1]]).unsqueeze(0)
    fx2 = torch.Tensor([[-1, 1], [-1, 1]]).unsqueeze(0)
    fy1 = torch.Tensor([[-1, -1], [1, 1]]).unsqueeze(0)
    fy2 = torch.Tensor([[-1, -1], [1, 1]]).unsqueeze(0)
    ft1 = torch.Tensor([[-1, -1], [-1, -1]]).unsqueeze(0)
    ft2 = torch.Tensor([[1, 1], [1, 1]]).unsqueeze(0)
    tx1 = filter2d(image_prev, fx1, border_type='reflect', normalized=False, padding='same')[0][0]
    tx2 = filter2d(image_next, fx2, border_type='reflect', normalized=False, padding='same')[0][0]

    resx = 0.5 * (tx1 + tx2)

    ty1 = filter2d(image_prev, fy1, border_type='reflect', normalized=False, padding='same')[0][0]
    ty2 = filter2d(image_next, fy2, border_type='reflect', normalized=False, padding='same')[0][0]
    resy = 0.5 * (ty1 + ty2)

    tdt1 = filter2d(image_prev, ft1, border_type='reflect', normalized=False, padding='same')[0][0]
    tdt2 = filter2d(image_next, ft2, border_type='reflect', normalized=False, padding='same')[0][0]
    resdt = 0.5 * (tdt1 + tdt2)

    u = torch.zeros(image_prev[0][0].shape)
    v = torch.zeros(image_prev[0][0].shape)

    for i in range(1, len(u)):
        for j in range(1, len(u[0])):
            px = resx[i - 1:i + 2, j - 1:j + 2]
            px = torch.flatten(px)
            py = resy[i - 1:i + 2, j - 1:j + 2]
            py = torch.flatten(py)
            pn = resdt[i - 1:i + 2, j - 1:j + 2]
            a = torch.stack([px, py])
            b = torch.flatten(pn)
            tr = a.t()
            if torch.det(torch.matmul(a, tr)) != 0:
                rightprod = torch.matmul(b, tr)
                leftprod = torch.matmul(a, tr)
                inv = torch.inverse(leftprod)
                res = torch.matmul(inv, rightprod)
                u[i][j] = res[0]
                v[i][j] = res[1]

    return torch.stack([u, v]).unsqueeze(0)


class OpticalFlowLK(nn.Module):
    r"""Calculate an optical flow for an input pair of images.

    Args:
        image_prev: the input tensor with shape :math:`(B,C,H,W)`.
        image_next: the size of the kernel.



    Returns:
        stacked tensor with shape :math:`(B,2,H,W)`.

    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, image_prev: torch.Tensor, image_next: torch.Tensor) -> torch.Tensor:
        return optical_flow_lk(image_prev, image_next)
