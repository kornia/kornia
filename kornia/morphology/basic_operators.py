# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


# _se_to_mask
def _se_to_mask(se: torch.Tensor) -> torch.Tensor:
    se_h, se_w = se.size()
    se_flat = se.view(-1)
    num_feats = se_h * se_w
    out = torch.zeros(num_feats, 1, se_h, se_w, dtype=se.dtype, device=se.device)
    for i in range(num_feats):
        y = i % se_h
        x = i // se_h
        out[i, 0, x, y] = (se_flat[i] >= 0).float()
    return out


# dilation
class Dilate(nn.Module):

    def __init__(self, se: torch.Tensor) -> None:
        super().__init__()
        self.se = se - 1
        self.se_h, self.se_w = se.shape
        self.pad = (self.se_h // 2, self.se_w // 2)
        self.kernel = _se_to_mask(self.se)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input.view(input.shape[0] * input.shape[1], 1, input.shape[2], input.shape[3])
        output = (F.conv2d(output, self.kernel, padding=self.pad) + self.se.view(1, -1, 1, 1)).max(dim=1)[0]

        return output.view(*input.shape)


def dilation(tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:

    r"""

    Returns the dilated image applying the same kernel in each channel.
    The kernel must have 2 dimensions, each one defined by an odd number.

    Args

       tensor (torch.Tensor): Image with shape :math:`(B, C, H, W)`.
       kernel (torch.Tensor): Structuring element with shape :math:`(H, W)`.

    Returns:

       torch.Tensor: Dilated image with shape :math:`(B, C, H, W)`.

    Example:

        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> dilated_img = dilation(tensor, kernel)

    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(tensor)))

    if len(tensor.shape) != 4:
        raise ValueError("Input size must have 4 dimensions. Got {}".format(
            tensor.dim()))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Kernel type is not a torch.Tensor. Got {}".format(
            type(kernel)))

    if len(kernel.shape) != 2:
        raise ValueError("Kernel size must have 2 dimensions. Got {}".format(
            kernel.dim()))

    return Dilate(kernel)(tensor)


# erosion
class Erode(nn.Module):

    def __init__(self, se: torch.Tensor) -> None:
        super().__init__()
        self.se = se - 1
        self.se_h, self.se_w = se.shape
        self.pad = (self.se_h // 2, self.se_w // 2, self.se_h // 2, self.se_w // 2)
        self.kernel = _se_to_mask(self.se)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input.view(input.shape[0] * input.shape[1], 1, input.shape[2], input.shape[3])
        output = F.pad(output, self.pad, mode='constant', value=1)
        output = (F.conv2d(output, self.kernel) - self.se.view(1, -1, 1, 1)).min(dim=1)[0]

        return output.view(*input.shape)


def erosion(tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:

    r"""

    Returns the eroded image applying the same kernel in each channel.
    The kernel must have 2 dimensions, each one defined by an odd number.

    Args

       tensor (torch.Tensor): Image with shape :math:`(B, C, H, W)`.
       kernel (torch.Tensor): Structuring element with shape :math:`(H, W)`.

    Returns:

       torch.Tensor: Eroded image with shape :math:`(B, C, H, W)`.

    Example:

        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(5, 5)
        >>> output = erosion(tensor, kernel)

    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(tensor)))

    if len(tensor.shape) != 4:
        raise ValueError("Input size must have 4 dimensions. Got {}".format(
            tensor.dim()))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Kernel type is not a torch.Tensor. Got {}".format(
            type(kernel)))

    if len(kernel.shape) != 2:
        raise ValueError("Kernel size must have 2 dimensions. Got {}".format(
            kernel.dim()))

    return Erode(kernel)(tensor)
