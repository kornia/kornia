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

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def image_to_tensor(image: Any, keepdim: bool = True) -> torch.Tensor:
    """Convert a NumPy image to a PyTorch tensor image.

    Convention:
        - The last axis of a 3-D or 4-D array is always the channels: :math:`(H, W, C)` becomes :math:`(C, H, W)` and
          :math:`(B, H, W, C)` becomes :math:`(B, C, H, W)`. A channels-first :math:`(C, H, W)` array is not detected:
          it is read as :math:`(H, W, C)`.
        - The dtype and the values are kept, so ``uint8`` stays in [0, 255], and the tensor is a view of the array
          that shares its memory. An array with a negative stride, such as OpenCV's ``img[..., ::-1]``, raises
          ``ValueError``; pass ``img[..., ::-1].copy()``.
        - ``keepdim`` defaults to ``True`` here and to ``False`` in :class:`~kornia.image.ImageToTensor`.
        - :ref:`Files and arrays <image-io-conventions>` relates it to torchvision's ``ToTensor``.

    Args:
        image: image of the form :math:`(H, W, C)`, :math:`(H, W)` or
            :math:`(B, H, W, C)`.
        keepdim: If ``False``, a 2-D or 3-D input gets a leading batch axis, :math:`(1, C, H, W)`. A 4-D input is
            always :math:`(B, C, H, W)`.

    Returns:
        tensor of the form :math:`(C, H, W)` for a 2-D or 3-D input with ``keepdim=True`` (a 2-D input has
        :math:`C = 1`), and :math:`(B, C, H, W)` otherwise.

    Raises:
        TypeError: if ``image`` is not a ``numpy.ndarray``.
        ValueError: if ``image`` is not two, three or four dimensional.

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
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input type is not a numpy.ndarray. Got {type(image)}")

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

    Convention:
        See the Convention block of :func:`~kornia.image.image_to_tensor`. Unlike it, this function copies the images
        into a new tensor and takes :math:`(H, W, C)` images only.

    Args:
        images: list of images, each of the form :math:`(H, W, C)`, all of the same shape.

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
        tensor = tensor.reshape(-1, tensor.shape[-3], tensor.shape[-2], tensor.shape[-1])

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
        tensor = tensor.reshape(-1, tensor.shape[-4], tensor.shape[-3], tensor.shape[-2], tensor.shape[-1])

    return tensor


def tensor_to_image(tensor: torch.Tensor, keepdim: bool = False, force_contiguous: bool = False) -> Any:
    """Convert a PyTorch tensor image to a numpy image.

    Convention:
        - The channels move last, and a single channel is always dropped: :math:`(1, H, W)` gives :math:`(H, W)` and
          :math:`(B, 1, H, W)` gives :math:`(B, H, W)`. ``keepdim`` keeps only the batch axis of a
          :math:`(1, C, H, W)` input.
        - The array shares memory with a CPU tensor. A tensor on another device is copied to the CPU, and
          ``force_contiguous=True`` copies a result that is not contiguous, such as the channels-last view of a
          contiguous tensor with more than one channel.
        - The dtype is kept, except ``bfloat16``, which NumPy lacks: it becomes ``float32``.

    Args:
        tensor: image of the form :math:`(H, W)`, :math:`(C, H, W)` or
            :math:`(B, C, H, W)`.
        keepdim: If ``True``, keep the batch axis of a :math:`(1, C, H, W)` input.
        force_contiguous: If ``True``, call ``contiguous`` on the result before converting it to NumPy.

    Returns:
        image of the form :math:`(H, W)`, :math:`(H, W, C)`, :math:`(B, H, W)` or :math:`(B, H, W, C)`.

    Raises:
        TypeError: if ``tensor`` is not a ``torch.Tensor``.
        ValueError: if ``tensor`` is not two, three or four dimensional.

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
    if image.dtype == torch.bfloat16:
        image = image.float()  # NumPy has no bfloat16

    if len(input_shape) == 2:
        # (H, W) -> (H, W)
        pass
    elif len(input_shape) == 3:
        # (C, H, W) -> (H, W, C)
        # Grayscale for proper plt.imshow needs to be (H,W); squeeze only the channel axis so that a unit H or W
        # of a 1-row or 1-column image survives.
        image = image.squeeze(0) if input_shape[0] == 1 else image.permute(1, 2, 0)
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
    """Convert a NumPy image to a PyTorch tensor image with :func:`image_to_tensor`.

    Convention:
        See the Convention block of :func:`~kornia.image.image_to_tensor`. Unlike it, ``keepdim`` defaults to
        ``False``, so an :math:`(H, W, C)` array becomes :math:`(1, C, H, W)`.

    Args:
        keepdim: If ``False``, a 2-D or 3-D input gets a leading batch axis, :math:`(1, C, H, W)`. A 4-D input is
            always :math:`(B, C, H, W)`.

    """

    def __init__(self, keepdim: bool = False) -> None:
        super().__init__()
        self.keepdim = keepdim

    def forward(self, x: Any) -> torch.Tensor:
        """Convert a NumPy image array into a PyTorch image tensor.

        Args:
            x: Input image array accepted by :func:`image_to_tensor`, of
                layout ``(H, W)``, ``(H, W, C)`` or ``(B, H, W, C)``,
                where :math:`B` is batch size, :math:`H` is height,
                :math:`W` is width, and :math:`C` is channel count.

        Returns:
            Tensor returned by :func:`image_to_tensor` with ``keepdim=self.keepdim``.
        """
        return image_to_tensor(x, keepdim=self.keepdim)


def make_grid(tensor: torch.Tensor, n_row: Optional[int] = None, padding: int = 2) -> torch.Tensor:
    r"""Convert a batched tensor to one image with padding in between.

    Convention:
        - ``n_row`` is the number of images in each row, that is the number of columns, as torchvision's ``nrow``;
          it defaults to :math:`\lceil \sqrt{B} \rceil`. The images fill each row in batch order, zero images complete
          the last row, and ``n_row > B`` still gives ``n_row`` columns.
        - ``padding`` zeros go between the images only, with no outer border, and the channels are kept: ``R`` rows
          of ``n_row`` images give :math:`(C, R (H + p) - p, n_{row} (W + p) - p)` for :math:`p` = ``padding``.
        - The grid is a new tensor in the input's dtype and on its device; :math:`B = 0` gives a zero-height grid.
        - :ref:`Files and arrays <image-io-conventions>` compares it with torchvision's ``make_grid``.

    Args:
        tensor: A batched tensor of shape (B, C, H, W).
        n_row: Number of images displayed in each row of the grid. If ``None``, use the square root
            of the batch size rounded up. Images are placed in batch order, filling each row before the next.
        padding: The amount of padding to add between images; must be non-negative.

    Returns:
        torch.Tensor: The combined image grid.

    Raises:
        TypeError: if ``tensor`` is not a ``torch.Tensor``.
        ValueError: if ``tensor`` is not four dimensional, ``n_row`` is not positive or ``padding`` is negative.

    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input tensor must be a PyTorch tensor.")
    if padding < 0:
        raise ValueError("padding must be non-negative.")

    B, C, H, W = tensor.shape
    if n_row is None:
        n_row = max(1, int(torch.sqrt(torch.tensor(B, dtype=torch.float32)).ceil().item()))
    elif n_row < 1:
        raise ValueError("n_row must be positive.")
    n_col = n_row
    n_row = (B + n_col - 1) // n_col

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

    return combined[:, :combined_H, :combined_W]


def perform_keep_shape_image(f: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Apply ``f`` to an image of arbitrary leading dimensions ``(*, C, H, W)``.

    Convention:
        - ``f`` always receives a ``(B, C, H, W)`` tensor: an ``(H, W)`` input becomes ``(1, 1, H, W)``, a
          ``(C, H, W)`` input ``(1, C, H, W)``, and the leading dimensions of a longer input are flattened into ``B``.
          The output gets the input's leading dimensions back around ``f``'s own ``C``, ``H`` and ``W``, so ``f`` must
          keep the batch size.
        - An empty input raises ``ValueError`` before ``f`` is called.
    """

    @wraps(f)
    def _wrapper(input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if not isinstance(input, torch.Tensor):
            raise TypeError(f"Input input type is not a torch.Tensor. Got {type(input)}")

        if input.shape.numel() == 0:
            raise ValueError("Invalid input tensor, it is empty.")

        input_shape = input.shape
        input = _to_bchw(input)  # reshape input as (B, C, H, W)
        output = f(input, *args, **kwargs)
        if len(input_shape) == 3:
            output = output[0]

        if len(input_shape) == 2:
            output = output[0, 0]

        if len(input_shape) > 4:
            output = output.reshape(*(input_shape[:-3] + output.shape[-3:]))

        return output

    return _wrapper


def perform_keep_shape_video(f: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Apply ``f`` to an image of arbitrary leading dimensions ``(*, C, D, H, W)``.

    Convention:
        See the Convention block of :func:`~kornia.image.perform_keep_shape_image`, with ``f`` receiving
        ``(B, C, D, H, W)``: a ``(D, H, W)`` input becomes ``(1, 1, D, H, W)`` and a ``(C, D, H, W)`` input
        ``(1, C, D, H, W)``.
    """

    @wraps(f)
    def _wrapper(input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if not isinstance(input, torch.Tensor):
            raise TypeError(f"Input input type is not a torch.Tensor. Got {type(input)}")

        if input.numel() == 0:
            raise ValueError("Invalid input tensor, it is empty.")

        input_shape = input.shape
        input = _to_bcdhw(input)  # reshape input as (B, C, D, H, W)
        output = f(input, *args, **kwargs)
        if len(input_shape) == 4:
            output = output[0]

        if len(input_shape) == 3:
            output = output[0, 0]

        if len(input_shape) > 5:
            output = output.reshape(*(input_shape[:-4] + output.shape[-4:]))

        return output

    return _wrapper
