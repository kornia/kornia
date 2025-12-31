# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from functools import wraps
from typing import Any, Callable, List, Optional

import torch
from torch import nn
from torch.nn import functional as F


def image_to_tensor(image: Any, keepdim: bool = True) -> torch.Tensor:
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


def image_list_to_tensor(images: List[Any]) -> torch.Tensor:
    """Convert a list of numpy images to a PyTorch 4d tensor image.

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
        raise ValueError("Input list of images is empty")

    images_t = []
    for img in images:
        if not torch.is_tensor(img):
            img = torch.as_tensor(img)
        images_t.append(img)

    shape = images_t[0].shape
    if len(shape) != 3:
        raise ValueError("Each image must have shape (H, W, C)")
    if any(img.shape != shape for img in images_t):
        raise ValueError("All images must have the same shape")

    # Stack into (N, H, W, C) then permute to (N, C, H, W)
    return torch.stack(images_t, dim=0).permute(0, 3, 1, 2)


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


def tensor_to_image(tensor: torch.Tensor, keepdim: bool = False, force_contiguous: bool = False) -> Any:
    """Convert a PyTorch tensor image to a numpy image.

    In case the tensor is in the GPU, it will be copied back to CPU.

    Args:
        tensor: image of the form :math:`(H, W)`, :math:`(C, H, W)` or
            :math:`(B, C, H, W)`.
        keepdim: If ``False`` squeeze the input image to match the shape
            :math:`(H, W, C)` or :math:`(H, W)`.
        force_contiguous: If ``True`` call `contiguous` to the tensor before

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
    image = tensor.cpu().detach()

    if len(input_shape) == 2:
        # (H, W) -> (H, W)
        pass
    elif len(input_shape) == 3:
        # (C, H, W) -> (H, W, C)
        if input_shape[0] == 1:
            # Grayscale for proper plt.imshow needs to be (H,W)
            image = image.squeeze()
        else:
            image = image.permute(1, 2, 0)
    elif len(input_shape) == 4:
        # (B, C, H, W) -> (B, H, W, C)
        image = image.permute(0, 2, 3, 1)
        if input_shape[0] == 1 and not keepdim:
            image = image.squeeze(0)
        if input_shape[1] == 1:
            image = image.squeeze(-1)
    else:
        raise ValueError(f"Cannot process tensor with shape {input_shape}")

    # make sure the image is contiguous
    if force_contiguous:
        image = image.contiguous()

    return image.numpy()


class ImageToTensor(nn.Module):
    """Converts a numpy image to a PyTorch 4d tensor image.

    Args:
        keepdim: If ``False`` unsqueeze the input image to match the shape :math:`(B, H, W, C)`.

    """

    def __init__(self, keepdim: bool = False) -> None:
        super().__init__()
        self.keepdim = keepdim

    def forward(self, x: Any) -> torch.Tensor:
        return image_to_tensor(x, keepdim=self.keepdim)


def make_grid(tensor: torch.Tensor, n_row: Optional[int] = None, padding: int = 2) -> torch.Tensor:
    """Convert a batched tensor to one image with padding in between.

    Args:
        tensor: A batched tensor of shape (B, C, H, W).
        n_row: Number of images displayed in each row of the grid.
        padding: The amount of padding to add between images.

    Returns:
        torch.Tensor: The combined image grid.

    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input tensor must be a PyTorch tensor.")

    B, C, H, W = tensor.shape
    if n_row is None:
        n_row = int(torch.sqrt(tensor(B, dtype=torch.float32)).ceil().item())
    n_col = (B + n_row - 1) // n_row

    padded_H = H + padding
    padded_W = W + padding

    # pad each image on right and bottom with `padding` zeros
    tensor_padded = F.pad(tensor, (0, padding, 0, padding))

    total = n_row * n_col
    if total > B:
        pad_tiles = torch.zeros((total - B, C, padded_H, padded_W), dtype=tensor.dtype, device=tensor.device)
        tensor_padded = torch.cat((tensor_padded, pad_tiles), dim=0)

    # ensure contiguous memory layout before reshaping / permuting
    tensor_padded = tensor_padded.contiguous()

    # reshape into (n_row, n_col, C, padded_H, padded_W)
    grid = tensor_padded.view(n_row, n_col, C, padded_H, padded_W)

    # permute to (C, n_row, padded_H, n_col, padded_W) then collapse
    grid = grid.permute(2, 0, 3, 1, 4).contiguous()
    combined = grid.view(C, n_row * padded_H, n_col * padded_W)

    # crop trailing right/bottom padding to match original
    combined_H = n_row * padded_H - padding
    combined_W = n_col * padded_W - padding
    combined = combined[:, :combined_H, :combined_W]

    return combined


def perform_keep_shape_image(f: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Apply `f` to an image of arbitrary leading dimensions `(*, C, H, W)`.

    It works by first viewing the image as `(B, C, H, W)`, applying the function and re-viewing the image as original
    shape.
    """

    @wraps(f)
    def _wrapper(input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if not isinstance(input, torch.Tensor):
            raise TypeError(f"Input input type is not a torch.Tensor. Got {type(input)}")

        if input.shape.numel() == 0:
            raise ValueError("Invalid input tensor, it is empty.")

        input_shape = input.shape
        input = _to_bchw(input)  # view input as (B, C, H, W)
        output = f(input, *args, **kwargs)
        if len(input_shape) == 3:
            output = output[0]

        if len(input_shape) == 2:
            output = output[0, 0]

        if len(input_shape) > 4:
            output = output.view(*(input_shape[:-3] + output.shape[-3:]))

        return output

    return _wrapper


def perform_keep_shape_video(f: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Apply `f` to an image of arbitrary leading dimensions `(*, C, D, H, W)`.

    It works by first viewing the image as `(B, C, D, H, W)`, applying the function and re-viewing the image as original
    shape.
    """

    @wraps(f)
    def _wrapper(input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if not isinstance(input, torch.Tensor):
            raise TypeError(f"Input input type is not a torch.Tensor. Got {type(input)}")

        if input.numel() == 0:
            raise ValueError("Invalid input tensor, it is empty.")

        input_shape = input.shape
        input = _to_bcdhw(input)  # view input as (B, C, D, H, W)
        output = f(input, *args, **kwargs)
        if len(input_shape) == 4:
            output = output[0]

        if len(input_shape) == 3:
            output = output[0, 0]

        if len(input_shape) > 5:
            output = output.view(*(input_shape[:-4] + output.shape[-4:]))

        return output

    return _wrapper
