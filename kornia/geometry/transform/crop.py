from typing import Tuple

import torch

from kornia.geometry.transform.imgwarp import (
    warp_perspective, get_perspective_transform
)

__all__ = [
    "crop_and_resize",
    "center_crop",
]


def crop_and_resize(tensor: torch.Tensor, boxes: torch.Tensor,
                    size: Tuple[int, int]) -> torch.Tensor:
    r"""Extracts crops from the input tensor and resizes them.

    Args:
        tensor (torch.Tensor): the reference tensor of shape BxCxHxW.
        boxes (torch.Tensor): a tensor containing the coordinates of the
          bounding boxes to be extracted. The tensor must have the shape
          of Bx4x2, where each box is defined in the following order: top-left,
          top-right, bottom-left and bottom-right. The coordinates order must
          be in y, x respectively.
        size (Tuple[int, int]): a tuple with the height and width that will be
          used to resize the extracted patches.

    Returns:
        torch.Tensor: tensor containing the patches with shape BxN1xN2

    Example:
        >>> input = torch.tensor([[
                [1., 2., 3., 4.],
                [5., 6., 7., 8.],
                [9., 10., 11., 12.],
                [13., 14., 15., 16.],
            ]])
        >>> boxes = torch.tensor([[
                [1., 1.],
                [1., 2.],
                [2., 1.],
                [2., 2.],
            ]])  # 1x4x2
        >>> kornia.crop_and_resize(input, boxes, (2, 2))
        tensor([[[ 6.0000,  7.0000],
                 [ 10.0000, 11.0000]]])
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))
    if not torch.is_tensor(boxes):
        raise TypeError("Input boxes type is not a torch.Tensor. Got {}"
                        .format(type(boxes)))
    if not len(tensor.shape) in (3, 4,):
        raise ValueError("Input tensor must be in the shape of CxHxW or "
                         "BxCxHxW. Got {}".format(tensor.shape))
    if not isinstance(size, (tuple, list,)) and len(size) == 2:
        raise ValueError("Input size must be a tuple/list of length 2. Got {}"
                         .format(size))
    # unpack input data
    dst_h: torch.Tensor = torch.tensor(size[0])
    dst_w: torch.Tensor = torch.tensor(size[1])

    # [y, x] origin
    # top-left, top-right, bottom-left, bottom-right
    points_src: torch.Tensor = boxes.to(
        tensor.device).to(tensor.dtype)

    # [y, x] destination
    # top-left, top-right, bottom-left, bottom-right
    points_dst: torch.Tensor = torch.tensor([[
        [0, 0],
        [0, dst_w - 1],
        [dst_h - 1, 0],
        [dst_h - 1, dst_w - 1],
    ]]).repeat(points_src.shape[0], 1, 1).to(
        tensor.device).to(tensor.dtype)

    # warping needs data in the shape of BCHW
    is_unbatched: bool = tensor.ndimension() == 3
    if is_unbatched:
        tensor = torch.unsqueeze(tensor, dim=0)

    # compute transformation between points and warp
    dst_trans_src: torch.Tensor = get_perspective_transform(
        points_src, points_dst)

    # simulate broadcasting
    dst_trans_src = dst_trans_src.expand(tensor.shape[0], -1, -1)

    patches: torch.Tensor = warp_perspective(
        tensor, dst_trans_src, (dst_h, dst_w))

    # return in the original shape
    if is_unbatched:
        patches = torch.squeeze(patches, dim=0)

    return patches


def center_crop(tensor: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    r"""Crops the given tensor at the center.

    Args:
        tensor (torch.Tensor): the input tensor with shape (C, H, W) or
          (B, C, H, W).
        size (Tuple[int, int]): a tuple with the expected height and width
          of the output patch.

    Returns:
        torch.Tensor: the output tensor with patches.

    Examples:
        >>> input = torch.tensor([[
                [1., 2., 3., 4.],
                [5., 6., 7., 8.],
                [9., 10., 11., 12.],
                [13., 14., 15., 16.],
             ]])
        >>> kornia.center_crop(input, (2, 4))
        tensor([[[ 5.0000,  6.0000,  7.0000,  8.0000],
                 [ 9.0000, 10.0000, 11.0000, 12.0000]]])
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))
    if not len(tensor.shape) in (3, 4,):
        raise ValueError("Input tensor must be in the shape of CxHxW or "
                         "BxCxHxW. Got {}".format(tensor.shape))
    if not isinstance(size, (tuple, list,)) and len(size) == 2:
        raise ValueError("Input size must be a tuple/list of length 2. Got {}"
                         .format(size))
    # unpack input sizes
    dst_h: torch.Tensor = torch.tensor(size[0])
    dst_w: torch.Tensor = torch.tensor(size[1])
    src_h: torch.Tensor = torch.tensor(tensor.shape[-2])
    src_w: torch.Tensor = torch.tensor(tensor.shape[-1])

    # compute start/end offsets
    dst_h_half: torch.Tensor = dst_h / 2
    dst_w_half: torch.Tensor = dst_w / 2
    src_h_half: torch.Tensor = src_h / 2
    src_w_half: torch.Tensor = src_w / 2

    start_x: torch.Tensor = src_h_half - dst_h_half
    start_y: torch.Tensor = src_w_half - dst_w_half

    end_x: torch.Tensor = start_x + dst_w - 1
    end_y: torch.Tensor = start_y + dst_h - 1

    # [y, x] origin
    # top-left, top-right, bottom-left, bottom-right
    points_src: torch.Tensor = torch.tensor([[
        [start_y, start_x],
        [start_y, end_x],
        [end_y, start_x],
        [end_y, end_x],
    ]]).to(tensor.device).to(tensor.dtype)

    # [y, x] destination
    # top-left, top-right, bottom-left, bottom-right
    points_dst: torch.Tensor = torch.tensor([[
        [0, 0],
        [0, dst_w - 1],
        [dst_h - 1, 0],
        [dst_h - 1, dst_w - 1],
    ]]).to(tensor.device).to(tensor.dtype)

    # warping needs data in the shape of BCHW
    is_unbatched: bool = tensor.ndimension() == 3
    if is_unbatched:
        tensor = torch.unsqueeze(tensor, dim=0)

    # compute transformation between points and warp
    dst_trans_src: torch.Tensor = get_perspective_transform(
        points_src, points_dst)
    dst_trans_src = dst_trans_src.repeat(tensor.shape[0], 1, 1)

    patches: torch.Tensor = warp_perspective(
        tensor, dst_trans_src, (dst_h, dst_w))

    # return in the original shape
    if is_unbatched:
        patches = torch.squeeze(patches, dim=0)

    return patches
