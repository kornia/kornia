from typing import Optional

import numpy as np
import torch


def image_to_tensor(image: np.ndarray, keepdim: bool = True) -> torch.Tensor:
    """Converts a numpy image to a PyTorch 4d tensor image.

    Args:
        image (numpy.ndarray): image of the form :math:`(H, W, C)`, :math:`(H, W)` or
            :math:`(B, H, W, C)`.
        keepdim (bool): If ``False`` unsqueeze the input image to match the shape
            :math:`(B, H, W, C)`. Default: ``True``

    Returns:
        torch.Tensor: tensor of the form :math:`(B, C, H, W)` if keepdim is ``False``,
            :math:`(C, H, W)` otherwise.
    """
    if not isinstance(image, (np.ndarray,)):
        raise TypeError("Input type must be a numpy.ndarray. Got {}".format(
            type(image)))

    if len(image.shape) > 4 or len(image.shape) < 2:
        raise ValueError(
            "Input size must be a two, three or four dimensional array")

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
        raise ValueError(
            "Cannot process image with shape {}".format(input_shape))

    return tensor.unsqueeze(0) if not keepdim else tensor


def _to_bchw(tensor: torch.Tensor, color_channel_num: Optional[int] = None) -> torch.Tensor:
    """Converts a PyTorch tensor image to BCHW format.

    Args:
        tensor (torch.Tensor): image of the form :math:`(H, W)`, :math:`(C, H, W)`, :math:`(H, W, C)` or
            :math:`(B, C, H, W)`.
        color_channel_num (Optional[int]): Color channel of the input tensor.
            If None, it will not alter the input channel.

    Returns:
        torch.Tensor: input tensor of the form :math:`(B, H, W, C)`.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) > 4 or len(tensor.shape) < 2:
        raise ValueError(f"Input size must be a two, three or four dimensional tensor. Got {tensor.shape}")

    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)

    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)

    # TODO(jian): this function is never used. Besides is not feasible for torchscript.
    # In addition, the docs must be updated. I don't understand what is doing.
    # if color_channel_num is not None and color_channel_num != 1:
    #    channel_list = [0, 1, 2, 3]
    #    channel_list.insert(1, channel_list.pop(color_channel_num))
    #    tensor = tensor.permute(*channel_list)
    return tensor


def _to_bcdhw(tensor: torch.Tensor, color_channel_num: Optional[int] = None) -> torch.Tensor:
    """Converts a PyTorch tensor image to BCHW format.
    Args:
        tensor (torch.Tensor): image of the form :math:`(D, H, W)`, :math:`(C, D, H, W)`, :math:`(D, H, W, C)` or
            :math:`(B, C, D, H, W)`.
        color_channel_num (Optional[int]): Color channel of the input tensor.
            If None, it will not alter the input channel.

    Returns:
        torch.Tensor: input tensor of the form :math:`(B, C, D, H, W)`.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) > 5 or len(tensor.shape) < 3:
        raise ValueError(f"Input size must be a three, four or five dimensional tensor. Got {tensor.shape}")

    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)

    if len(tensor.shape) == 4:
        tensor = tensor.unsqueeze(0)

    # TODO(jian): this function is never used. Besides is not feasible for torchscript.
    # In addition, the docs must be updated. I don't understand what is doing.
    # if color_channel_num is not None and color_channel_num != 1:
    #    channel_list = [0, 1, 2, 3, 4]
    #    channel_list.insert(1, channel_list.pop(color_channel_num))
    #    tensor = tensor.permute(*channel_list)
    return tensor


def tensor_to_image(tensor: torch.Tensor) -> np.array:
    """Converts a PyTorch tensor image to a numpy image.

    In case the tensor is in the GPU, it will be copied back to CPU.

    Args:
        tensor (torch.Tensor): image of the form :math:`(H, W)`, :math:`(C, H, W)` or
            :math:`(B, C, H, W)`.

    Returns:
        numpy.ndarray: image of the form :math:`(H, W)`, :math:`(H, W, C)` or :math:`(B, H, W, C)`.

    """
    if not isinstance(tensor, torch.Tensor):
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
