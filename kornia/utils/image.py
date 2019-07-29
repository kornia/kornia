from typing import Optional

import numpy as np
import torch


def image_to_tensor(image: np.array) -> torch.Tensor:
    """Converts a numpy image to a PyTorch 4d tensor image.

    Args:
        image (numpy.ndarray): image of the form :math:`(H, W)`,
        math:`(H, W, C)`, or math:`(B, H, W, C)`.

    Returns:
        torch.Tensor: tensor of the form :math:`(H, W)`,
        math:`(B, C, H, W)`.

    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input type is not a numpy.ndarray. Got {}".format(
            type(image)))

    if len(image.shape) > 4 or len(image.shape) < 2:
        raise ValueError(
            "Input size must be a two, three or four dimensional array")

    input_shape = image.shape
    tensor: torch.Tensor = torch.from_numpy(image)
    if len(input_shape) == 2:
        # (H, W) -> (1, 1, H, W)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif len(input_shape) == 3:
        # (H, W, C) -> (C, H, W)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    elif len(input_shape) == 4:
        # (B, H, W, C) -> (B, C, H, W)
        tensor = tensor.permute(0, 3, 1, 2)
    else:
        raise ValueError(
            "Cannot process image with shape {}".format(input_shape))
    return tensor


def tensor_to_image(tensor: torch.Tensor) -> np.array:
    """Converts a PyTorch tensor image to a numpy image. In case the tensor is
    in the GPU, it will be copied back to CPU.

    Args:
        tensor (torch.Tensor): image of the form :math:`(H, W)`,
        math:`(C, H, W)`, or math:`(B, C, H, W)`.

    Returns:
        numpy.ndarray: image of the form :math:`(H, W)`,
        math:`(H, W)`, math:`(H, W, C)`, or math:`(B, H, W, C)`.

    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(tensor)))

    if len(tensor.shape) > 4 or len(tensor.shape) < 2:
        raise ValueError(
            "Input size must be a two, three or four dimensional tensor")

    input_shape = tensor.shape
    image: np.array = tensor.cpu().detach().numpy()
    if len(input_shape) == 2:
        # (H, W) -> (H, W)
        image = image
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
        raise ValueError(
            "Cannot process tensor with shape {}".format(input_shape))

    return image
