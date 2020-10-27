import torch
import kornia.morphology


def black_hat(tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    r"""
        Returns the black hat tranformation of an image,
        (that means, closed_image - image) applying the same kernel in each channel.

        The kernel must have 2 dimensions, each one defined by an odd number.

        See :class:`~kornia.morphology.close` for details.

        Args
           tensor (torch.Tensor): Image with shape :math:`(B, C, H, W)`.
           kernel (torch.Tensor): Structuring element with shape :math:`(H, W)`.

        Returns:
           torch.Tensor: Top hat transformated image with shape :math:`(B, C, H, W)`.

        Example:
            >>> tensor = torch.rand(1, 3, 5, 5)
            >>> kernel = torch.ones(3, 3)
            >>> black_hat_img = black_hat(tensor, kernel)
        """

    return kornia.morphology.close(tensor, kernel) - tensor
