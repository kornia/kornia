import torch
import numpy as np

__all__ = [
    "tensor_to_image",
    "image_to_tensor",
]


def image_to_tensor(image):
    """Converts a numpy image to a torch.Tensor image.

    Args: 
        image (numpy.ndarray): image of the form (H, W, C).

    Returns:
        numpy.ndarray: image of the form (H, W, C).

    """
    if not type(image) == np.array:
        raise TypeError("Input type is not a numpy.array. Got {}".format(
            type(image)))
    if len(image.shape) > 3 or len(image.shape) < 2:
        raise ValueError("Input size must be a two or three dimensional array")
    tensor = torch.from_numpy(image)
    if len(tensor.shape) == 2:
        tensor = torch.unsqueeze(tensor, dim=0)
    return tensor.permute(2, 0, 1)  # CxHxW


def tensor_to_image(tensor):
    """Converts a torch.Tensor image to a numpy image. In case the tensor is
       in the GPU, it will be copied back to CPU.

    Args:
        tensor (Tensor): image of the form (C, H, W).

    Returns:
        numpy.ndarray: image of the form (H, W, C).

    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(tensor)))
    if len(tensor.shape) > 3 or len(tensor.shape) < 2:
        raise ValueError(
            "Input size must be a two or three dimensional tensor")
    tensor = torch.squeeze(tensor)
    if len(tensor.shape) == 2:
        tensor = torch.unsqueeze(tensor, dim=0)
    return tensor.permute(1, 2, 0).contiguous().cpu().detach().numpy()
