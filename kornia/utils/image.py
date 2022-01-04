from functools import wraps
from typing import TYPE_CHECKING, Callable, List

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

    Example:
        >>> img = np.ones((3, 3))
        >>> image_to_tensor(img).shape
        torch.Size([1, 3, 3])

        >>> img = np.ones((4, 4, 1))
        >>> image_to_tensor(img).shape
        torch.Size([1, 4, 4])

        >>> img = np.ones((4, 4, 3))
        >>> image_to_tensor(img, keepdim=False).shape
        torch.Size([1, 3, 4, 4])
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
        raise ValueError(f"Cannot process image with shape {input_shape}")

    return tensor.unsqueeze(0) if not keepdim else tensor


def image_list_to_tensor(images: List["np.ndarray"]) -> torch.Tensor:
    """Converts a list of numpy images to a PyTorch 4d tensor image.

    Args:
        images: list of images, each of the form :math:`(H, W, C)`.
        Image shapes must be consistent

    Returns:
        tensor of the form :math:`(B, C, H, W)`.

    Example:
        >>> imgs = [np.ones((4, 4, 1)), np.zeros((4, 4, 1))]
        >>> image_list_to_tensor(imgs).shape
        torch.Size([2, 1, 4, 4])
    """
    if not images:
        raise ValueError("Input list of numpy images is empty")
    if len(images[0].shape) != 3:
        raise ValueError("Input images must be three dimensional arrays")

    list_of_tensors: List[torch.Tensor] = []
    for image in images:
        list_of_tensors.append(image_to_tensor(image))
    tensor: torch.Tensor = torch.stack(list_of_tensors)
    return tensor


def _to_bchw(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a PyTorch tensor image to BCHW format.

    Args:
        tensor (torch.Tensor): image of the form :math:`(*, H, W)`.

    Returns:
        input tensor of the form :math:`(B, C, H, W)`.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) < 2:
        raise ValueError(f"Input size must be a two, three or four dimensional tensor. Got {tensor.shape}")

    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)

    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)

    if len(tensor.shape) > 4:
        tensor = tensor.view(-1, tensor.shape[-3], tensor.shape[-2], tensor.shape[-1])

    return tensor


def _to_bcdhw(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a PyTorch tensor image to BCDHW format.

    Args:
        tensor (torch.Tensor): image of the form :math:`(*, D, H, W)`.

    Returns:
        input tensor of the form :math:`(B, C, D, H, W)`.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) < 3:
        raise ValueError(f"Input size must be a three, four or five dimensional tensor. Got {tensor.shape}")

    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)

    if len(tensor.shape) == 4:
        tensor = tensor.unsqueeze(0)

    if len(tensor.shape) > 5:
        tensor = tensor.view(-1, tensor.shape[-4], tensor.shape[-3], tensor.shape[-2], tensor.shape[-1])

    return tensor


def tensor_to_image(tensor: torch.Tensor, keepdim: bool = False) -> "np.ndarray":
    """Converts a PyTorch tensor image to a numpy image.

    In case the tensor is in the GPU, it will be copied back to CPU.

    Args:
        tensor: image of the form :math:`(H, W)`, :math:`(C, H, W)` or
            :math:`(B, C, H, W)`.
        keepdim: If ``False`` squeeze the input image to match the shape
            :math:`(H, W, C)` or :math:`(H, W)`.

    Returns:
        image of the form :math:`(H, W)`, :math:`(H, W, C)` or :math:`(B, H, W, C)`.

    Example:
        >>> img = torch.ones(1, 3, 3)
        >>> tensor_to_image(img).shape
        (3, 3)

        >>> img = torch.ones(3, 4, 4)
        >>> tensor_to_image(img).shape
        (4, 4, 3)
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

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
        if input_shape[0] == 1 and not keepdim:
            image = image.squeeze(0)
        if input_shape[1] == 1:
            image = image.squeeze(-1)
    else:
        raise ValueError(f"Cannot process tensor with shape {input_shape}")

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


def perform_keep_shape_image(f: Callable) -> Callable:
    """A decorator that enable `f` to be applied to an image of arbitrary leading dimensions `(*, C, H, W)`.

    It works by first viewing the image as `(B, C, H, W)`, applying the function and re-viewing the image as original
    shape.
    """

    @wraps(f)
    def _wrapper(input: torch.Tensor, *args, **kwargs):
        if not isinstance(input, torch.Tensor):
            raise TypeError(f"Input input type is not a torch.Tensor. Got {type(input)}")

        if input.numel() == 0:
            raise ValueError("Invalid input tensor, it is empty.")

        input_shape = input.shape
        input = _to_bchw(input)  # view input as (B, C, H, W)
        output: torch.Tensor = f(input, *args, **kwargs)
        if len(input_shape) == 3:
            output = output[0]

        if len(input_shape) == 2:
            output = output[0, 0]

        if len(input_shape) > 4:
            output = output.view(*(input_shape[:-3] + output.shape[-3:]))

        return output

    return _wrapper


def perform_keep_shape_video(f: Callable) -> Callable:
    """A decorator that enable `f` to be applied to an image of arbitrary leading dimensions `(*, C, D, H, W)`.

    It works by first viewing the image as `(B, C, D, H, W)`, applying the function and re-viewing the image as original
    shape.
    """

    @wraps(f)
    def _wrapper(input: torch.Tensor, *args, **kwargs):
        if not isinstance(input, torch.Tensor):
            raise TypeError(f"Input input type is not a torch.Tensor. Got {type(input)}")

        if input.numel() == 0:
            raise ValueError("Invalid input tensor, it is empty.")

        input_shape = input.shape
        input = _to_bcdhw(input)  # view input as (B, C, D, H, W)
        output: torch.Tensor = f(input, *args, **kwargs)
        if len(input_shape) == 4:
            output = output[0]

        if len(input_shape) == 3:
            output = output[0, 0]

        if len(input_shape) > 5:
            output = output.view(*(input_shape[:-4] + output.shape[-4:]))

        return output

    return _wrapper
