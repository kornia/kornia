from typing import Optional

import numpy as np
import torch


def hw_image_to_hw_tensor(image: np.array) -> torch.Tensor:
    if len(image.shape) != 2:
        raise ValueError("Input size must be a two dimensional array")
    tensor: torch.Tensor = torch.from_numpy(image)
    return tensor


def hwc_image_to_chw_tensor(image: np.array) -> torch.Tensor:
    if len(image.shape) != 3:
        raise ValueError("Input size must be a three dimensional array")
    tensor: torch.Tensor = torch.from_numpy(image)
    tensor = tensor.permute(2, 0, 1)
    return tensor


def bhwc_image_to_bchw_tensor(image: np.array) -> torch.Tensor:
    if len(image.shape) != 4:
        raise ValueError("Input size must be a four dimensional array")
    tensor: torch.Tensor = torch.from_numpy(image)
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor


def image_to_tensor(image: np.array) -> torch.Tensor:
    """Converts a numpy image to a PyTorch tensor image.

    Args:
        image (numpy.ndarray): image of the form :math:`(H, W)`, math:`(H, W, C)`, 
        or math:`(B, H, W, C)`.

    Returns:
        torch.Tensor: tensor of the form :math:`(H, W)`, math:`(C, H, W)`, 
        or math:`(B, C, H, W)`.

    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input type is not a numpy.ndarray. Got {}".format(
            type(image)))

    if len(image.shape) > 4 or len(image.shape) < 2:
        raise ValueError(
            "Input size must be a two, three or four dimensional array")

    input_shape = image.shape
    if len(input_shape) == 2:
        tensor: torch.Tensor = hw_image_to_hw_tensor(image)
    elif len(input_shape) == 3:
        tensor: torch.Tensor = hwc_image_to_chw_tensor(image)
    elif len(input_shape) == 4:
        tensor: torch.Tensor = bhwc_image_to_bchw_tensor(image)
    else:
        raise ValueError("Cannot process image with shape {}".format(input_shape))
    return tensor


def hw_tensor_to_hw_image(tensor: torch.Tensor) -> np.array:
    if len(tensor.shape) != 2:
        raise ValueError("Input size must be a two dimensional array")
    image: np.array = tensor.cpu().detach().numpy()
    return image


def chw_tensor_to_hwc_image(tensor: torch.Tensor) -> np.array:
    if len(tensor.shape) != 3:
        raise ValueError("Input size must be a three dimensional array")
    image: np.array = tensor.permute(1, 2, 0).cpu().detach().numpy()
    return image


def bchw_tensor_to_bhwc_image(tensor: torch.Tensor) -> np.array:
    if len(tensor.shape) != 4:
        raise ValueError("Input size must be a four dimensional array")
    image: np.array = tensor.permute(0, 2, 3, 1).cpu().detach().numpy()
    return image


def tensor_to_image(tensor: torch.Tensor) -> np.array:
    """Converts a PyTorch tensor image to a numpy image. In case the tensor is
    in the GPU, it will be copied back to CPU.

    Args:
        tensor (torch.Tensor): image of the form :math:`(H, W)`, math:`(C, H, W)`, 
        or math:`(B, C, H, W)`.

    Returns:
        numpy.ndarray: image of the form :math:`(H, W)`, math:`(H, W, C)`, 
        or math:`(B, H, W, C)`.

    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(tensor)))

    if len(tensor.shape) > 4 or len(tensor.shape) < 2:
        raise ValueError(
            "Input size must be a two, three or four dimensional tensor")

    input_shape = tensor.shape
    if len(input_shape) == 2:
        image: np.array = hw_tensor_to_hw_image(tensor)
    elif len(input_shape) == 3:
        image: np.array = chw_tensor_to_hwc_image(tensor)
    elif len(input_shape) == 4:
        image: np.array = bchw_tensor_to_bhwc_image(tensor)
    else:
        raise ValueError("Cannot process tensor with shape {}".format(input_shape))

    return image
