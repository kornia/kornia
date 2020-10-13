"""This module is deprecated soon.

Please use kornia.geometry.crop instead.
"""

from typing import Tuple, Union
import warnings

import torch

from .. import crop as crop

__all__ = [
    "crop_and_resize",
    "crop_by_boxes",
    "center_crop",
    "bbox_to_mask",
    "infer_box_shape",
    "validate_bboxes",
    "bbox_generator"
]


def __deprecation_warning(name: str, replacement: str):
    warnings.warn(f"`{name}` is no longer maintained and will be removed from the future versions. "
                  f"Please use {replacement} instead.")


def crop_and_resize(tensor: torch.Tensor, boxes: torch.Tensor, size: Tuple[int, int],
                    interpolation: str = 'bilinear', align_corners: bool = False) -> torch.Tensor:
    r"""Extract crops from 2D images (4D tensor) and resize them.

    Args:
        tensor (torch.Tensor): the 2D image tensor with shape (C, H, W) or (B, C, H, W).
        boxes (torch.Tensor): a tensor containing the coordinates of the
          bounding boxes to be extracted. The tensor must have the shape
          of Bx4x2, where each box is defined in the following (clockwise)
          order: top-left, top-right, bottom-right and bottom-left. The
          coordinates must be in the x, y order.
        size (Tuple[int, int]): a tuple with the height and width that will be
          used to resize the extracted patches.
        interpolation (str): Interpolation flag. Default: 'bilinear'.
        align_corners (bool): mode for grid_generation. Default: False. See
          https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for details.

    Returns:
        torch.Tensor: tensor containing the patches with shape BxCxN1xN2.

    Example:
        >>> input = torch.tensor([[
                [1., 2., 3., 4.],
                [5., 6., 7., 8.],
                [9., 10., 11., 12.],
                [13., 14., 15., 16.],
            ]])
        >>> boxes = torch.tensor([[
                [1., 1.],
                [2., 1.],
                [2., 2.],
                [1., 2.],
            ]])  # 1x4x2
        >>> kornia.crop_and_resize(input, boxes, (2, 2))
        tensor([[[ 6.0000,  7.0000],
                 [ 10.0000, 11.0000]]])
    """
    __deprecation_warning("kornia.geometry.transform.crop_and_resize", "kornia.geometry.crop.crop_and_resize")
    return crop.crop_and_resize(tensor, boxes, size, interpolation, align_corners)


def center_crop(tensor: torch.Tensor, size: Tuple[int, int],
                interpolation: str = 'bilinear',
                align_corners: bool = True) -> torch.Tensor:
    r"""Crop the 2D images (4D tensor) at the center.

    Args:
        tensor (torch.Tensor): the 2D image tensor with shape (C, H, W) or (B, C, H, W).
        size (Tuple[int, int]): a tuple with the expected height and width
          of the output patch.
        interpolation (str): Interpolation flag. Default: 'bilinear'.
        align_corners (bool): mode for grid_generation. Default: False. See
          https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for details
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
    __deprecation_warning("kornia.geometry.transform.center_crop", "kornia.geometry.crop.center_crop")
    return crop.center_crop(tensor, size, interpolation, align_corners)


def crop_by_boxes(tensor: torch.Tensor, src_box: torch.Tensor, dst_box: torch.Tensor,
                  interpolation: str = 'bilinear', align_corners: bool = False) -> torch.Tensor:
    """Perform crop transform on 2D images (4D tensor) by bounding boxes.

    Given an input tensor, this function selected the interested areas by the provided bounding boxes (src_box).
    Then the selected areas would be fitted into the targeted bounding boxes (dst_box) by a perspective transformation.
    So far, the ragged tensor is not supported by PyTorch right now. This function hereby requires the bounding boxes
    in a batch must be rectangles with same width and height.

    Args:
        tensor (torch.Tensor): the 2D image tensor with shape (C, H, W) or (B, C, H, W).
        src_box (torch.Tensor): a tensor with shape (B, 4, 2) containing the coordinates of the bounding boxes
            to be extracted. The tensor must have the shape of Bx4x2, where each box is defined in the clockwise
            order: top-left, top-right, bottom-right and bottom-left. The coordinates must be in x, y order.
        dst_box (torch.Tensor): a tensor with shape (B, 4, 2) containing the coordinates of the bounding boxes
            to be placed. The tensor must have the shape of Bx4x2, where each box is defined in the clockwise
            order: top-left, top-right, bottom-right and bottom-left. The coordinates must be in x, y order.
        interpolation (str): Interpolation flag. Default: 'bilinear'.
        align_corners (bool): mode for grid_generation. Default: False. See
          https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for details

    Returns:
        torch.Tensor: the output tensor with patches.

    Examples:
        >>> input = torch.arange(16, dtype=torch.float32).reshape((1, 4, 4))
        >>> src_box = torch.tensor([[
        ...     [1., 1.],
        ...     [2., 1.],
        ...     [2., 2.],
        ...     [1., 2.],
        ... ]])  # 1x4x2
        >>> dst_box = torch.tensor([[
        ...     [0., 0.],
        ...     [1., 0.],
        ...     [1., 1.],
        ...     [0., 1.],
        ... ]])  # 1x4x2
        >>> crop_by_boxes(input, src_box, dst_box, align_corners=True)
        tensor([[[ 5.0000,  6.0000],
                 [ 9.0000, 10.0000]]])

    Note:
        If the src_box is smaller than dst_box, the following error will be thrown.
        RuntimeError: solve_cpu: For batch 0: U(2,2) is zero, singular U.
    """
    __deprecation_warning("kornia.geometry.transform.crop_by_boxes", "kornia.geometry.crop.crop_by_boxes")
    return crop.crop_by_boxes(tensor, src_box, dst_box, interpolation, align_corners)


def infer_box_shape(boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Auto-infer the output sizes for the given 2D bounding boxes.

    Args:
        boxes (torch.Tensor): a tensor containing the coordinates of the
          bounding boxes to be extracted. The tensor must have the shape
          of Bx4x2, where each box is defined in the following (clockwise)
          order: top-left, top-right, bottom-right, bottom-left. The
          coordinates must be in the x, y order.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
        - Bounding box heights, shape of :math:`(B,)`.
        - Boundingbox widths, shape of :math:`(B,)`.

    Example:
        >>> boxes = torch.tensor([[
        ...     [1., 1.],
        ...     [2., 1.],
        ...     [2., 2.],
        ...     [1., 2.],
        ... ], [
        ...     [1., 1.],
        ...     [3., 1.],
        ...     [3., 2.],
        ...     [1., 2.],
        ... ]])  # 2x4x2
        >>> infer_box_shape(boxes)
        (tensor([2., 2.]), tensor([2., 3.]))
    """
    __deprecation_warning("kornia.geometry.transform.infer_box_shape", "kornia.geometry.crop.infer_box_shape")
    return crop.infer_box_shape(boxes)


def validate_bboxes(boxes: torch.Tensor) -> None:
    """Validate if a 2D bounding box usable or not.

    This function checks if the boxes are rectangular or not.

    Args:
        boxes (torch.Tensor): a tensor containing the coordinates of the
          bounding boxes to be extracted. The tensor must have the shape
          of Bx4x2, where each box is defined in the following (clockwise)
          order: top-left, top-right, bottom-right, bottom-left. The
          coordinates must be in the x, y order.
    """
    __deprecation_warning("kornia.geometry.transform.validate_bboxes", "kornia.geometry.crop.validate_bboxes")
    return crop.validate_bboxes(boxes)


def bbox_to_mask(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Convert 2D bounding boxes to masks. Covered area is 1. and the remaining is 0.

    Args:
        boxes (torch.Tensor): a tensor containing the coordinates of the
          bounding boxes to be extracted. The tensor must have the shape
          of Bx4x2, where each box is defined in the following (clockwise)
          order: top-left, top-right, bottom-right, bottom-left. The
          coordinates must be in the x, y order.
        width (int): width of the masked image.
        height (int): height of the masked image.

    Returns:
        torch.Tensor: the output mask tensor.

    Examples:
        >>> boxes = torch.tensor([[
        ...        [1., 1.],
        ...        [3., 1.],
        ...        [3., 2.],
        ...        [1., 2.],
        ...   ]])  # 1x4x2
        >>> bbox_to_mask(boxes, 5, 5)
        tensor([[[0., 0., 0., 0., 0.],
                 [0., 1., 1., 1., 0.],
                 [0., 1., 1., 1., 0.],
                 [0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]]])
    """
    __deprecation_warning("kornia.geometry.transform.bbox_to_mask", "kornia.geometry.crop.bbox_to_mask")
    return crop.bbox_to_mask(boxes, width, height)


def bbox_generator(
    x_start: torch.Tensor, y_start: torch.Tensor, width: torch.Tensor, height: torch.Tensor
) -> torch.Tensor:
    """Generate 2D bounding boxes according to the provided start coords, width and height.

    Args:
        x_start (torch.Tensor): a tensor containing the x coordinates of the bounding boxes to be extracted.
            Shape must be a scalar tensor or :math:`(B,)`.
        y_start (torch.Tensor): a tensor containing the y coordinates of the bounding boxes to be extracted.
            Shape must be a scalar tensor or :math:`(B,)`.
        width (torch.Tensor): widths of the masked image.
            Shape must be a scalar tensor or :math:`(B,)`.
        height (torch.Tensor): heights of the masked image.
            Shape must be a scalar tensor or :math:`(B,)`.

    Returns:
        torch.Tensor: the bounding box tensor.

    Examples:
        >>> x_start = torch.tensor([0, 1])
        >>> y_start = torch.tensor([1, 0])
        >>> width = torch.tensor([5, 3])
        >>> height = torch.tensor([7, 4])
        >>> bbox_generator(x_start, y_start, width, height)
        tensor([[[0, 1],
                 [4, 1],
                 [4, 7],
                 [0, 7]],
        <BLANKLINE>
                [[1, 0],
                 [3, 0],
                 [3, 3],
                 [1, 3]]])
    """
    __deprecation_warning("kornia.geometry.transform.bbox_generator", "kornia.geometry.crop.bbox_generator")
    return crop.bbox_generator(x_start, y_start, width, height)
