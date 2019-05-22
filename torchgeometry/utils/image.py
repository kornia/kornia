from typing import Optional

import numpy as np
import torch


def image_to_tensor(image: np.array) -> torch.Tensor:
    """Converts a numpy image to a PyTorch tensor image.

    Args:
        image (numpy.ndarray): image of the form :math:`(H, W, C)`.

    Returns:
        torch.Tensor: tensor of the form :math:`(C, H, W)`.

    """
    if not type(image) == np.ndarray:
        raise TypeError("Input type is not a numpy.ndarray. Got {}".format(
            type(image)))

    if len(image.shape) > 3 or len(image.shape) < 2:
        raise ValueError("Input size must be a two or three dimensional array")

    tensor: torch.Tensor = torch.from_numpy(image)

    if len(tensor.shape) == 2:
        tensor = torch.unsqueeze(tensor, dim=-1)

    return tensor.permute(2, 0, 1).squeeze_()  # CxHxW


def tensor_to_image(tensor: torch.Tensor) -> np.array:
    """Converts a PyTorch tensor image to a numpy image. In case the tensor is
    in the GPU, it will be copied back to CPU.

    Args:
        tensor (torch.Tensor): image of the form :math:`(C, H, W)`.

    Returns:
        numpy.ndarray: image of the form :math:`(H, W, C)`.

    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(tensor)))

    if len(tensor.shape) > 3 or len(tensor.shape) < 2:
        raise ValueError(
            "Input size must be a two or three dimensional tensor")

    input_shape = tensor.shape
    if len(input_shape) == 2:
        tensor = torch.unsqueeze(tensor, dim=0)

    tensor = tensor.permute(1, 2, 0)

    if len(input_shape) == 2:
        tensor = torch.squeeze(tensor, dim=-1)

    return tensor.cpu().detach().numpy()
