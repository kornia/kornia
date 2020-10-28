import torch
import torch.nn as nn
import torch.nn.functional as F


class Erode(nn.Module):

    def __init__(self, se: torch.Tensor) -> None:
        super().__init__()
        self.se = se - 1
        self.se_h, self.se_w = se.shape
        self.pad = (self.se_h // 2, self.se_w // 2)

        def se_to_mask(se: torch.Tensor) -> torch.Tensor:
            se_h, se_w = se.size()
            se_flat = se.view(-1)
            num_feats = se_h * se_w
            out = torch.zeros(num_feats, 1, se_h, se_w, dtype=se.dtype, device=se.device)
            for i in range(num_feats):
                y = i % se_h
                x = i // se_h
                out[i, 0, x, y] = (se_flat[i] >= 0).float()
            return out

        self.kernel = se_to_mask(self.se)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input.view(input.shape[0] * input.shape[1], 1, input.shape[2], input.shape[3])
        output = (F.conv2d(output, self.kernel, padding=self.pad) - self.se.view(1, -1, 1, 1)).min(dim=1)[0]

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
        >>> output = kornia.morphology.erosion(tensor, kernel)

    """

    assert torch.is_tensor(tensor), "Invalid type for image. Expected torch.Tensor"

    assert torch.is_tensor(kernel), "Invalid type for kernel. Expected torch.Tensor"

    assert tensor.dim() == 4, f"Invalid number of dimensions for image. Expected 4. Got {tensor.dim()}"

    assert kernel.dim() == 2, f"Invalid number of dimensions for kernel. Expected 2. Got {kernel.dim()}"

    return Erode(kernel)(tensor)
