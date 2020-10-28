import torch
import kornia.morphology


def top_hat(tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    r"""
        Returns the top hat tranformation of an image,
        (that means, image - opened_image) applying the same kernel in each channel.

        The kernel must have 2 dimensions, each one defined by an odd number.

        See :class:`~kornia.morphology.open` for details.

        Args
           tensor (torch.Tensor): Image with shape :math:`(B, C, H, W)`.
           kernel (torch.Tensor): Structuring element with shape :math:`(H, W)`.

        Returns:
           torch.Tensor: Top hat transformated image with shape :math:`(B, C, H, W)`.

        Example:
            >>> tensor = torch.rand(1, 3, 5, 5)
            >>> kernel = torch.ones(3, 3)
            >>> top_hat_img = top_hat(tensor, kernel)
        """

    assert torch.is_tensor(tensor), "Invalid type for image. Expected torch.Tensor"

    assert torch.is_tensor(kernel), "Invalid type for kernel. Expected torch.Tensor"

    assert tensor.dim() == 4, f"Invalid number of dimensions for image. Expected 4. Got {tensor.dim()}"

    assert kernel.dim() == 2, f"Invalid number of dimensions for kernel. Expected 2. Got {kernel.dim()}"

    return tensor - kornia.morphology.open(tensor, kernel)
