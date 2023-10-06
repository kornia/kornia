import torch
import torch.nn.functional as F

from kornia.core import Tensor


def connected_components(image: Tensor, num_iterations: int = 100) -> Tensor:
    r"""Computes the Connected-component labelling (CCL) algorithm.

    .. image:: https://github.com/kornia/data/raw/main/cells_segmented.png

    The implementation is an adaptation of the following repository:

    https://gist.github.com/efirdc/5d8bd66859e574c683a504a4690ae8bc

    .. warning::
        This is an experimental API subject to changes and optimization improvements.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/connected_components.html>`__.

    Args:
        image: the binarized input image with shape :math:`(*, 1, H, W)`.
          The image must be in floating point with range [0, 1].
        num_iterations: the number of iterations to make the algorithm to converge.

    Return:
        The labels image with the same shape of the input image.

    Example:
        >>> img = torch.rand(2, 1, 4, 5)
        >>> img_labels = connected_components(img, num_iterations=100)
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input imagetype is not a Tensor. Got: {type(image)}")

    if not isinstance(num_iterations, int) or num_iterations < 1:
        raise TypeError("Input num_iterations must be a positive integer.")

    if len(image.shape) < 3 or image.shape[-3] != 1:
        raise ValueError(f"Input image shape must be (*,1,H,W). Got: {image.shape}")

    H, W = image.shape[-2:]
    image_view = image.view(-1, 1, H, W)

    # precompute a mask with the valid values
    mask = image_view == 1

    # allocate the output tensors for labels
    B, _, _, _ = image_view.shape
    out = torch.arange(B * H * W, device=image.device, dtype=image.dtype).view((-1, 1, H, W))
    out[~mask] = 0

    for _ in range(num_iterations):
        out = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)
        out = torch.mul(out, mask)  # mask using element-wise multiplication

    return out.view_as(image)
