from functools import wraps
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    import numpy as np


def image_to_tensor(image: "np.ndarray", keepdim: bool = True) -> torch.Tensor:
    """Convert a numpy image to a PyTorch 4d tensor image.

    Args:
        image: image of the form :math:`(H, W, C)`, :math:`(H, W)` or
            :math:`(B, H, W, C)`.
        keepdim: If ``False`` unsqueeze the input image to match the shape
            :math:`(B, H, W, C)`.

    Returns:
        tensor of the form :math:`(B, C, H, W)` if keepdim is ``False``,
            :math:`(C, H, W)` otherwise.
    """
    if len(image.shape) > 4 or len(image.shape) < 2:
        raise ValueError("Input size must be a two, three or four dimensional array")

    input_shape = image.shape
    tensor: torch.Tensor = torch.from_numpy(image)

    if len(input_shape) == 2:
        # (H, W) -> (1, H, W)
        tensor = tensor.unsqueeze(0)
    elif len(input_shape) == 3:
        # (H, W, C) -> (C, H, W)
        tensor = tensor.permute(2, 0, 1)
    elif len(input_shape) == 4:
        # (B, H, W, C) -> (B, C, H, W)
        tensor = tensor.permute(0, 3, 1, 2)
        keepdim = True  # no need to unsqueeze
    else:
        raise ValueError("Cannot process image with shape {}".format(input_shape))

    return tensor.unsqueeze(0) if not keepdim else tensor


def _to_bchw(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a PyTorch tensor image to BCHW format.

    Args:
        tensor: image of the form :math:`(H, W)`, :math:`(C, H, W)`, :math:`(H, W, C)` or
            :math:`(B, C, H, W)`.

    Returns:
        input tensor of the form :math:`(B, C, H, W)`.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) > 4 or len(tensor.shape) < 2:
        raise ValueError(f"Input size must be a two, three or four dimensional tensor. Got {tensor.shape}")

    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)

    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)

    return tensor


def _to_bcdhw(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a PyTorch tensor image to BCDHW format.

    Args:
        tensor: image of the form :math:`(D, H, W)`, :math:`(C, D, H, W)`, :math:`(D, H, W, C)` or
            :math:`(B, C, D, H, W)`.

    Returns:
        input tensor of the form :math:`(B, C, D, H, W)`.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) > 5 or len(tensor.shape) < 3:
        raise ValueError(f"Input size must be a three, four or five dimensional tensor. Got {tensor.shape}")

    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)

    if len(tensor.shape) == 4:
        tensor = tensor.unsqueeze(0)

    return tensor


def tensor_to_image(tensor: torch.Tensor) -> "np.ndarray":
    """Converts a PyTorch tensor image to a numpy image.

    In case the tensor is in the GPU, it will be copied back to CPU.

    Args:
        tensor: image of the form :math:`(H, W)`, :math:`(C, H, W)` or
            :math:`(B, C, H, W)`.

    Returns:
        image of the form :math:`(H, W)`, :math:`(H, W, C)` or :math:`(B, H, W, C)`.

    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(tensor)))

    if len(tensor.shape) > 4 or len(tensor.shape) < 2:
        raise ValueError("Input size must be a two, three or four dimensional tensor")

    input_shape = tensor.shape
    image: "np.ndarray" = tensor.cpu().detach().numpy()

    if len(input_shape) == 2:
        # (H, W) -> (H, W)
        pass
    elif len(input_shape) == 3:
        # (C, H, W) -> (H, W, C)
        if input_shape[0] == 1:
            # Grayscale for proper plt.imshow needs to be (H,W)
            image = image.squeeze()
        else:
            image = image.transpose(1, 2, 0)
    elif len(input_shape) == 4:
        # (B, C, H, W) -> (B, H, W, C)
        image = image.transpose(0, 2, 3, 1)
        if input_shape[0] == 1:
            image = image.squeeze(0)
        if input_shape[1] == 1:
            image = image.squeeze(-1)
    else:
        raise ValueError("Cannot process tensor with shape {}".format(input_shape))

    return image


class ImageToTensor(nn.Module):
    """Converts a numpy image to a PyTorch 4d tensor image.

    Args:
        keepdim: If ``False`` unsqueeze the input image to match the shape :math:`(B, H, W, C)`.
    """

    def __init__(self, keepdim: bool = False):
        super().__init__()
        self.keepdim = keepdim

    def forward(self, x: "np.ndarray") -> torch.Tensor:
        return image_to_tensor(x, keepdim=self.keepdim)


def perform_keep_shape(f):
    """TODO: where can we put this?"""

    @wraps(f)
    def _wrapper(input, *args, **kwargs):
        input_shape = input.shape
        if len(input_shape) == 2:
            input = input[None]

        dont_care_shape = input.shape[:-3]
        input = input.view(-1, input.shape[-3], input.shape[-2], input.shape[-1])

        output = f(input, *args, **kwargs)
        output = output.view(*(dont_care_shape + output.shape[-3:]))
        if len(input_shape) == 2:
            output = output[0]
        return output

    return _wrapper
