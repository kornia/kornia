import torch
import kornia.morphology


def open(tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    r"""
        Returns the opened image, (that means, erosion after a dilation) applying the same kernel in each channel.
        The kernel must have 2 dimensions, each one defined by an odd number.

        Args
           tensor (torch.Tensor): Image with shape :math:`(B, C, H, W)`.
           kernel (torch.Tensor): Structuring element with shape :math:`(H, W)`.

        Returns:
           torch.Tensor: Dilated image with shape :math:`(B, C, H, W)`.

        Example:
            >>> tensor = torch.rand(1, 3, 5, 5)
            >>> kernel = torch.ones(3, 3)
            >>> opened_img = open(tensor, kernel)
        """

    return kornia.morphology.Dilate(kernel)((kornia.morphology.Erode(kernel)(tensor)))
