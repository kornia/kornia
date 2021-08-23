import torch
import torch.nn.functional as F


def connected_components(input: torch.Tensor, num_iterations: int = 100) -> torch.Tensor:
    r"""Computes the Connected-component labelling (CCL) algorithm.

    .. image:: https://github.com/kornia/data/raw/main/cells_segmented.png

    The implementation is an adaptation of the following repository:

    https://gist.github.com/efirdc/5d8bd66859e574c683a504a4690ae8bc

    .. note::
        This is an experimental API subject to changes and optimization improvements.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       connected_components.html>`__.

    Args:
        input: the binarized input image with shape :math:`(B, 1, H, W)`.
          The image must be in floating point with range [0, 1].
        num_iterations: the number of iterations to make the algorithm to converge.

    Return:
        The labels image with the same shape of the input image.

    Example:
        >>> img = torch.rand(2, 1, 4, 5)
        >>> img_labels = connected_components(img, num_iterations=100)
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input input type is not a torch.Tensor. Got: {type(input)}")

    if not isinstance(num_iterations, int) or num_iterations < 1:
        raise TypeError("Input num_iterations must be a positive integer.")

    if len(input.shape) != 4 or input.shape[-3] != 1:
        raise ValueError(f"Input image shape must be Bx1xHxW. Got: {input.shape}")

    # precomput a mask with the valid values
    mask = input == 1

    # allocate the output tensors for labels
    B, _, H, W = input.shape
    out = torch.arange(B * W * H, device=input.device, dtype=input.dtype).reshape((B, 1, H, W))
    out[~mask] = 0

    for _ in range(num_iterations):
        out[mask] = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)[mask]

    return out
