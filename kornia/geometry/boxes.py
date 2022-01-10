from typing import List, Optional, Tuple, Union, cast

import torch

from kornia.geometry.bbox import validate_bbox
from kornia.geometry.linalg import transform_points

__all__ = ["Boxes", "Boxes3D"]


def _is_floating_point_dtype(dtype: torch.dtype) -> bool:
    return dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.half)


def _merge_box_list(
    boxes: List[torch.Tensor], method: str = "pad"
) -> Tuple[torch.Tensor, List[int]]:
    r"""Merge a list of boxes into one tensor.
    """
    if not all(box.shape[-2:] == torch.Size([4, 2]) and box.dim() == 3 for box in boxes):
        raise TypeError(
            f"Input boxes must be a list of (N, 4, 2) shaped. Got: {[box.shape for box in boxes]}.")

    if method == "pad":
        max_N = max(box.shape[0] for box in boxes)
        stats = [max_N - box.shape[0] for box in boxes]
        output = torch.nn.utils.rnn.pad_sequence(boxes, batch_first=True)
    else:
        raise NotImplementedError(f"`{method}` is not implemented.")

    return output, stats


def _transform_boxes(boxes: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Transforms 3D and 2D in kornia format by applying the transformation matrix M. Boxes and the transformation
    matrix could be batched or not.

    Args:
        boxes: 2D quadrilaterals or 3D hexahedrons in kornia format.
        M: the transformation matrix of shape :math:`(3, 3)` or :math:`(B, 3, 3)` for 2D and :math:`(4, 4)` or
            :math:`(B, 4, 4)` for 3D hexahedron.
    """
    M = M if M.is_floating_point() else M.float()

    # Work with batch as kornia.transform_points only supports a batch of points.
    boxes_per_batch, n_points_per_box, coordinates_dimension = boxes.shape[-3:]
    points = boxes.view(-1, n_points_per_box * boxes_per_batch, coordinates_dimension)
    M = M if M.ndim == 3 else M.unsqueeze(0)

    if points.shape[0] != M.shape[0]:
        raise ValueError(
            f"Batch size mismatch. Got {points.shape[0]} for boxes and {M.shape[0]} for the transformation matrix."
        )

    transformed_boxes: torch.Tensor = transform_points(M, points)
    transformed_boxes = transformed_boxes.view_as(boxes)
    return transformed_boxes


def _boxes_to_polygons(
    xmin: torch.Tensor, ymin: torch.Tensor, width: torch.Tensor, height: torch.Tensor
) -> torch.Tensor:
    if not xmin.ndim == ymin.ndim == width.ndim == height.ndim == 2:
        raise ValueError("We expect to create a batch of 2D boxes (quadrilaterals) in vertices format (B, N, 4, 2)")

    # Create (B,N,4,2) with all points in top left position of boxes
    polygons = torch.zeros((xmin.shape[0], xmin.shape[1], 4, 2), device=xmin.device, dtype=xmin.dtype)
    polygons[..., 0] = xmin.unsqueeze(-1)
    polygons[..., 1] = ymin.unsqueeze(-1)
    # Shift top-right, bottom-right, bottom-left points to the right coordinates
    polygons[..., 1, 0] += width - 1  # Top right
    polygons[..., 2, 0] += width - 1  # Bottom right
    polygons[..., 2, 1] += height - 1  # Bottom right
    polygons[..., 3, 1] += height - 1  # Bottom left
    return polygons


def _boxes_to_quadrilaterals(
    boxes: torch.Tensor, mode: str = "xyxy", validate_boxes: bool = True
) -> torch.Tensor:
    """Convert from boxes to quadrilaterals."""
    mode = mode.lower()

    if mode.startswith("vertices"):
        batched = boxes.ndim == 4
        if not (3 <= boxes.ndim <= 4 and boxes.shape[-2:] == torch.Size([4, 2])):
            raise ValueError(f"Boxes shape must be (N, 4, 2) or (B, N, 4, 2) when {mode} mode. Got {boxes.shape}.")
    elif mode.startswith("xy"):
        batched = boxes.ndim == 3
        if not (2 <= boxes.ndim <= 3 and boxes.shape[-1] == 4):
            raise ValueError(f"Boxes shape must be (N, 4) or (B, N, 4) when {mode} mode. Got {boxes.shape}.")
    else:
        raise ValueError(f"Unknown mode {mode}")

    boxes = boxes if boxes.is_floating_point() else boxes.float()
    boxes = boxes if batched else boxes.unsqueeze(0)

    if mode.startswith("vertices"):
        if mode == "vertices":
            # Avoid passing reference
            quadrilaterals = boxes.clone()
        elif mode == "vertices_plus":
            quadrilaterals = boxes.clone()  # TODO: perform +1
        else:
            raise ValueError(f"Unknown mode {mode}")
        validate_boxes or validate_bbox(quadrilaterals)
    elif mode.startswith("xy"):
        if mode == "xyxy":
            height, width = boxes[..., 3] - boxes[..., 1], boxes[..., 2] - boxes[..., 0]
        elif mode == "xyxy_plus":
            height, width = boxes[..., 3] - boxes[..., 1] + 1, boxes[..., 2] - boxes[..., 0] + 1
        elif mode == "xywh":
            height, width = boxes[..., 3], boxes[..., 2]
        else:
            raise ValueError(f"Unknown mode {mode}")

        if validate_boxes:
            if (width <= 0).any():
                raise ValueError("Some boxes have negative widths or 0.")
            if (height <= 0).any():
                raise ValueError("Some boxes have negative heights or 0.")

        xmin, ymin = boxes[..., 0], boxes[..., 1]
        quadrilaterals = _boxes_to_polygons(xmin, ymin, width, height)
    else:
        raise ValueError(f"Unknown mode {mode}")

    quadrilaterals = quadrilaterals if batched else quadrilaterals.squeeze(0)

    return quadrilaterals


def _boxes3d_to_polygons3d(
    xmin: torch.Tensor,
    ymin: torch.Tensor,
    zmin: torch.Tensor,
    width: torch.Tensor,
    height: torch.Tensor,
    depth: torch.Tensor,
) -> torch.Tensor:
    if not xmin.ndim == ymin.ndim == zmin.ndim == width.ndim == height.ndim == depth.ndim == 2:
        raise ValueError("We expect to create a batch of 3D boxes (hexahedrons) in vertices format (B, N, 8, 3)")

    # Front
    # Create (B,N,4,3) with all points in front top left position of boxes
    front_vertices = torch.zeros((xmin.shape[0], xmin.shape[1], 4, 3), device=xmin.device, dtype=xmin.dtype)
    front_vertices[..., 0] = xmin.unsqueeze(-1)
    front_vertices[..., 1] = ymin.unsqueeze(-1)
    front_vertices[..., 2] = zmin.unsqueeze(-1)
    # Shift front-top-right, front-bottom-right, front-bottom-left points to the right coordinates
    front_vertices[..., 1, 0] += width - 1  # Top right
    front_vertices[..., 2, 0] += width - 1  # Bottom right
    front_vertices[..., 2, 1] += height - 1  # Bottom right
    front_vertices[..., 3, 1] += height - 1  # Bottom left

    # Back
    back_vertices = front_vertices.clone()
    back_vertices[..., 2] += depth.unsqueeze(-1) - 1

    polygons3d = torch.cat([front_vertices, back_vertices], dim=-2)
    return polygons3d


# NOTE: Cannot jit with Union types with torch <= 0.10
# @torch.jit.script
class Boxes:
    r"""2D boxes containing N or BxN boxes.

    Args:
        boxes: 2D boxes, shape of :math:`(N, 4, 2)`, :math:`(B, N, 4, 2)` or a list of :math:`(N, 4, 2)`.
            See below for more details.
        raise_if_not_floating_point: flag to control floating point casting behaviour when `boxes` is not a floating
            point tensor. True to raise an error when `boxes` isn't a floating point tensor, False to cast to float.
        mode: the box format of the input boxes.

    Note:
        **2D boxes format** is defined as a floating data type tensor of shape ``Nx4x2`` or ``BxNx4x2``
        where each box is a `quadrilateral <https://en.wikipedia.org/wiki/Quadrilateral>`_ defined by it's 4 vertices
        coordinates (A, B, C, D). Coordinates must be in ``x, y`` order. The height and width of a box is defined as
        ``width = xmax - xmin + 1`` and ``height = ymax - ymin + 1``. Examples of
        `quadrilaterals <https://en.wikipedia.org/wiki/Quadrilateral>`_ are rectangles, rhombus and trapezoids.
    """
    def __init__(
        self, boxes: Union[torch.Tensor, List[torch.Tensor]], raise_if_not_floating_point: bool = True,
        mode: str = "vertices_plus"
    ) -> None:

        self._N: Optional[List[int]] = None

        if isinstance(boxes, list):
            boxes, self._N = _merge_box_list(boxes)

        if not isinstance(boxes, torch.Tensor):
            raise TypeError(f"Input boxes is not a Tensor. Got: {type(boxes)}.")

        if not boxes.is_floating_point():
            if raise_if_not_floating_point:
                raise ValueError(f"Coordinates must be in floating point. Got {boxes.dtype}")

            boxes = boxes.float()

        if len(boxes.shape) == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            boxes = boxes.reshape((-1, 4))

        if not (3 <= boxes.ndim <= 4 and boxes.shape[-2:] == (4, 2)):
            raise ValueError(f"Boxes shape must be (N, 4, 2) or (B, N, 4, 2). Got {boxes.shape}.")

        self._is_batched = False if boxes.ndim == 3 else True

        self._data = boxes
        self._mode = mode

    def get_boxes_shape(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Compute boxes heights and widths.

        Returns:
            - Boxes heights, shape of :math:`(N,)` or :math:`(B,N)`.
            - Boxes widths, shape of :math:`(N,)` or :math:`(B,N)`.

        Example:
            >>> boxes_xyxy = torch.tensor([[[1,1,2,2],[1,1,3,2]]])
            >>> boxes = Boxes.from_tensor(boxes_xyxy)
            >>> boxes.get_boxes_shape()
            (tensor([[1., 1.]]), tensor([[1., 2.]]))
        """
        boxes_xywh = cast(torch.Tensor, self.to_tensor("xywh", as_padded_sequence=True))
        widths, heights = boxes_xywh[..., 2], boxes_xywh[..., 3]
        return heights, widths

    @classmethod
    def from_tensor(
        cls, boxes: Union[torch.Tensor, List[torch.Tensor]], mode: str = "xyxy", validate_boxes: bool = True
    ) -> "Boxes":
        r"""Helper method to easily create :class:`Boxes` from boxes stored in another format.

        Args:
            boxes: 2D boxes, shape of :math:`(N, 4)`, :math:`(B, N, 4)`, :math:`(N, 4, 2)` or :math:`(B, N, 4, 2)`.
            mode: The format in which the boxes are provided.

                * 'xyxy': boxes are assumed to be in the format ``xmin, ymin, xmax, ymax`` where ``width = xmax - xmin``
                  and ``height = ymax - ymin``. With shape :math:`(N, 4)`, :math:`(B, N, 4)`.
                * 'xyxy_plus': similar to 'xyxy' mode but where box width and length are defined as
                  ``width = xmax - xmin + 1`` and ``height = ymax - ymin + 1``.
                  With shape :math:`(N, 4)`, :math:`(B, N, 4)`.
                * 'xywh': boxes are assumed to be in the format ``xmin, ymin, width, height`` where
                  ``width = xmax - xmin`` and ``height = ymax - ymin``. With shape :math:`(N, 4)`, :math:`(B, N, 4)`.
                * 'vertices': boxes are defined by their vertices points in the following ``clockwise`` order:
                  *top-left, top-right, bottom-right, bottom-left*. Vertices coordinates are in (x,y) order. Finally,
                  box width and height are defined as ``width = xmax - xmin`` and ``height = ymax - ymin``.
                  With shape :math:`(N, 4, 2)` or :math:`(B, N, 4, 2)`.
                * 'vertices_plus': similar to 'vertices' mode but where box width and length are defined as
                  ``width = xmax - xmin + 1`` and ``height = ymax - ymin + 1``. ymin + 1``.
                  With shape :math:`(N, 4, 2)` or :math:`(B, N, 4, 2)`.

            validate_boxes: check if boxes are valid rectangles or not. Valid rectangles are those with width
                and height >= 1 (>= 2 when mode ends with '_plus' suffix).

        Returns:
            :class:`Boxes` class containing the original `boxes` in the format specified by ``mode``.

        Examples:
            >>> boxes_xyxy = torch.as_tensor([[0, 3, 1, 4], [5, 1, 8, 4]])
            >>> boxes = Boxes.from_tensor(boxes_xyxy, mode='xyxy')
            >>> boxes.data  # (2, 4, 2)
            tensor([[[0., 3.],
                     [0., 3.],
                     [0., 3.],
                     [0., 3.]],
            <BLANKLINE>
                    [[5., 1.],
                     [7., 1.],
                     [7., 3.],
                     [5., 3.]]])
        """
        quadrilaterals: Union[torch.Tensor, List[torch.Tensor]]
        if isinstance(boxes, (torch.Tensor)):
            quadrilaterals = _boxes_to_quadrilaterals(boxes, mode=mode, validate_boxes=validate_boxes)
        else:
            quadrilaterals = [_boxes_to_quadrilaterals(box, mode, validate_boxes) for box in boxes]

        # Due to some torch.jit.script bug (at least <= 1.9), you need to pass all arguments to __init__ when
        # constructing the class from inside of a method.
        return cls(quadrilaterals, False, mode)

    def to_tensor(
        self, mode: Optional[str] = None, as_padded_sequence: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        r"""Cast :class:`Boxes` to a tensor. ``mode`` controls which 2D boxes format should be use to represent boxes
        in the tensor.

        Args:
            mode: the output box format. It could be:

                * 'xyxy': boxes are defined as ``xmin, ymin, xmax, ymax`` where ``width = xmax - xmin`` and
                  ``height = ymax - ymin``.
                * 'xyxy_plus': similar to 'xyxy' mode but where box width and length are defined as
                  ``width = xmax - xmin + 1`` and ``height = ymax - ymin + 1``.
                * 'xywh': boxes are defined as ``xmin, ymin, width, height`` where ``width = xmax - xmin``
                  and ``height = ymax - ymin``.
                * 'vertices': boxes are defined by their vertices points in the following ``clockwise`` order:
                  *top-left, top-right, bottom-right, bottom-left*. Vertices coordinates are in (x,y) order. Finally,
                  box width and height are defined as ``width = xmax - xmin`` and ``height = ymax - ymin``.
                * 'vertices_plus': similar to 'vertices' mode but where box width and length are defined as
                  ``width = xmax - xmin + 1`` and ``height = ymax - ymin + 1``. ymin + 1``.
            as_padded_sequence: whether to keep the pads for a list of boxes. This parameter is only valid
                if the boxes are from a box list.

        Returns:
            Boxes tensor in the ``mode`` format. The shape depends with the ``mode`` value:

                * 'vertices' or 'verticies_plus': :math:`(N, 4, 2)` or :math:`(B, N, 4, 2)`.
                * Any other value: :math:`(N, 4)` or :math:`(B, N, 4)`.

        Examples:
            >>> boxes_xyxy = torch.as_tensor([[0, 3, 1, 4], [5, 1, 8, 4]])
            >>> boxes = Boxes.from_tensor(boxes_xyxy)
            >>> assert (boxes_xyxy == boxes.to_tensor(mode='xyxy')).all()
        """
        batched_boxes = self._data if self._is_batched else self._data.unsqueeze(0)

        boxes: Union[torch.Tensor, List[torch.Tensor]]

        # Create boxes in xyxy_plus format.
        boxes = torch.stack([batched_boxes.amin(dim=-2), batched_boxes.amax(dim=-2)], dim=-2).view(
            batched_boxes.shape[0], batched_boxes.shape[1], 4
        )

        if mode is None:
            mode = self.mode

        mode = mode.lower()

        if mode in ("xyxy", "xyxy_plus"):
            pass
        elif mode in ("xywh", "vertices", "vertices_plus"):
            height, width = boxes[..., 3] - boxes[..., 1] + 1, boxes[..., 2] - boxes[..., 0] + 1
            boxes[..., 2] = width
            boxes[..., 3] = height
        else:
            raise ValueError(f"Unknown mode {mode}")

        if mode in ("xyxy", "vertices"):
            offset = torch.as_tensor([0, 0, 1, 1], device=boxes.device, dtype=boxes.dtype)
            boxes = boxes + offset

        if mode.startswith('vertices'):
            boxes = _boxes_to_polygons(boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3])

        if self._N is not None and not as_padded_sequence:
            boxes = list(torch.nn.functional.pad(
                o, (len(o.shape) - 1) * [0, 0] + [0, - n]) for o, n in zip(boxes, self._N))
        else:
            boxes = boxes if self._is_batched else boxes.squeeze(0)
        return boxes

    def to_mask(self, height: int, width: int) -> torch.Tensor:
        """Convert 2D boxes to masks. Covered area is 1 and the remaining is 0.

        Args:
            height: height of the masked image/images.
            width: width of the masked image/images.

        Returns:
            the output mask tensor, shape of :math:`(N, width, height)` or :math:`(B,N, width, height)` and dtype of
            :func:`Boxes.dtype` (it can be any floating point dtype).

        Note:
            It is currently non-differentiable.

        Examples:
            >>> boxes = Boxes(torch.tensor([[  # Equivalent to boxes = Boxes.from_tensor([[1,1,4,3]])
            ...        [1., 1.],
            ...        [4., 1.],
            ...        [4., 3.],
            ...        [1., 3.],
            ...   ]]))  # 1x4x2
            >>> boxes.to_mask(5, 5)
            tensor([[[0., 0., 0., 0., 0.],
                     [0., 1., 1., 1., 1.],
                     [0., 1., 1., 1., 1.],
                     [0., 1., 1., 1., 1.],
                     [0., 0., 0., 0., 0.]]])
        """
        if self._data.requires_grad:
            raise RuntimeError(
                "Boxes.to_tensor isn't differentiable. Please, create boxes from tensors with `requires_grad=False`."
            )

        if self._is_batched:  # (B, N, 4, 2)
            mask = torch.zeros(
                (self._data.shape[0], self._data.shape[1], height, width), dtype=self.dtype, device=self.device
            )
        else:  # (N, 4, 2)
            mask = torch.zeros((self._data.shape[0], height, width), dtype=self.dtype, device=self.device)

        # Boxes coordinates can be outside the image size after transforms. Clamp values to the image size
        clipped_boxes_xyxy = cast(torch.Tensor, self.to_tensor("xyxy", as_padded_sequence=True))
        clipped_boxes_xyxy[..., ::2].clamp_(0, width)
        clipped_boxes_xyxy[..., 1::2].clamp_(0, height)

        # Reshape mask to (BxN, H, W) and boxes to (BxN, 4) to iterate over all of them.
        # Cast boxes coordinates to be integer to use them as indexes. Use round to handle decimal values.
        for mask_channel, box_xyxy in zip(mask.view(-1, height, width), clipped_boxes_xyxy.view(-1, 4).round().int()):
            # Mask channel dimensions: (height, width)
            mask_channel[box_xyxy[1] : box_xyxy[3], box_xyxy[0] : box_xyxy[2]] = 1

        return mask

    def transform_boxes(self, M: torch.Tensor, inplace: bool = False) -> "Boxes":
        r"""Apply a transformation matrix to the 2D boxes.

        Args:
            M: The transformation matrix to be applied, shape of :math:`(3, 3)` or :math:`(B, 3, 3)`.
            inplace: do transform in-place and return self.

        Returns:
            The transformed boxes.
        """
        if not 2 <= M.ndim <= 3 or M.shape[-2:] != (3, 3):
            raise ValueError(f"The transformation matrix shape must be (3, 3) or (B, 3, 3). Got {M.shape}.")

        # Due to some torch.jit.script bug (at least <= 1.9), you need to pass all arguments to __init__ when
        # constructing the class from inside of a method.
        transformed_boxes = _transform_boxes(self._data, M)
        if inplace:
            self._data = transformed_boxes
            return self

        return Boxes(transformed_boxes, False)

    def transform_boxes_(self, M: torch.Tensor) -> "Boxes":
        """Inplace version of :func:`Boxes.transform_boxes`"""
        return self.transform_boxes(M, inplace=True)

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def device(self) -> torch.device:
        """Returns boxes device."""
        return self._data.device

    @property
    def dtype(self) -> torch.dtype:
        """Returns boxes dtype."""
        return self._data.dtype

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> "Boxes":
        """Like :func:`torch.nn.Module.to()` method."""
        # In torchscript, dtype is a int and not a class. https://github.com/pytorch/pytorch/issues/51941
        if dtype is not None and not _is_floating_point_dtype(dtype):
            raise ValueError("Boxes must be in floating point")
        self._data = self._data.to(device=device, dtype=dtype)
        return self


@torch.jit.script
class Boxes3D:
    r"""3D boxes containing N or BxN boxes.

    Args:
        boxes: 3D boxes, shape of :math:`(N,8,3)` or :math:`(B,N,8,3)`. See below for more details.
        raise_if_not_floating_point: flag to control floating point casting behaviour when `boxes` is not a floating
            point tensor. True to raise an error when `boxes` isn't a floating point tensor, False to cast to float.

    Note:
        **3D boxes format** is defined as a floating data type tensor of shape ``Nx8x3`` or ``BxNx8x3`` where each box
        is a `hexahedron <https://en.wikipedia.org/wiki/Hexahedron>`_ defined by it's 8 vertices coordinates.
        Coordinates must be in ``x, y, z`` order. The height, width and depth of a box is defined as
        ``width = xmax - xmin + 1``, ``height = ymax - ymin + 1`` and ``depth = zmax - zmin + 1``. Examples of
        `hexahedrons <https://en.wikipedia.org/wiki/Hexahedron>`_ are cubes and rhombohedrons.
    """
    def __init__(
        self, boxes: torch.Tensor, raise_if_not_floating_point: bool = True,
        mode: str = "xyzxyz_plus"
    ) -> None:
        if not isinstance(boxes, torch.Tensor):
            raise TypeError(f"Input boxes is not a Tensor. Got: {type(boxes)}.")

        if not boxes.is_floating_point():
            if raise_if_not_floating_point:
                raise ValueError(f"Coordinates must be in floating point. Got {boxes.dtype}.")

            boxes = boxes.float()

        if len(boxes.shape) == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            boxes = boxes.reshape((-1, 6))

        if not (3 <= boxes.ndim <= 4 and boxes.shape[-2:] == (8, 3)):
            raise ValueError(f"3D bbox shape must be (N, 8, 3) or (B, N, 8, 3). Got {boxes.shape}.")

        self._is_batched = False if boxes.ndim == 3 else True

        self._data = boxes
        self._mode = mode

    def get_boxes_shape(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Compute boxes heights and widths.

        Returns:
            - Boxes depths, shape of :math:`(N,)` or :math:`(B,N)`.
            - Boxes heights, shape of :math:`(N,)` or :math:`(B,N)`.
            - Boxes widths, shape of :math:`(N,)` or :math:`(B,N)`.

        Example:
            >>> boxes_xyzxyz = torch.tensor([[ 0,  1,  2, 10, 21, 32], [3, 4, 5, 43, 54, 65]])
            >>> boxes3d = Boxes3D.from_tensor(boxes_xyzxyz)
            >>> boxes3d.get_boxes_shape()
            (tensor([30., 60.]), tensor([20., 50.]), tensor([10., 40.]))
        """
        boxes_xyzwhd = self.to_tensor(mode='xyzwhd')
        widths, heights, depths = boxes_xyzwhd[..., 3], boxes_xyzwhd[..., 4], boxes_xyzwhd[..., 5]
        return depths, heights, widths

    @classmethod
    def from_tensor(cls, boxes: torch.Tensor, mode: str = "xyzxyz", validate_boxes: bool = True) -> "Boxes3D":
        r"""Helper method to easily create :class:`Boxes3D` from 3D boxes stored in another format.

        Args:
            boxes: 3D boxes, shape of :math:`(N,6)` or :math:`(B,N,6)`.
            mode: The format in which the 3D boxes are provided.

                * 'xyzxyz': boxes are assumed to be in the format ``xmin, ymin, zmin, xmax, ymax, zmax`` where
                  ``width = xmax - xmin``, ``height = ymax - ymin`` and ``depth = zmax - zmin``.
                * 'xyzxyz_plus': similar to 'xyzxyz' mode but where box width, length and depth are defined as
                  ``width = xmax - xmin + 1``, ``height = ymax - ymin + 1`` and ``depth = zmax - zmin + 1``.
                * 'xyzwhd': boxes are assumed to be in the format ``xmin, ymin, zmin, width, height, depth`` where
                  ``width = xmax - xmin``, ``height = ymax - ymin`` and ``depth = zmax - zmin``.

            validate_boxes: check if boxes are valid rectangles or not. Valid rectangles are those with width, height
                and depth >= 1 (>= 2 when mode ends with '_plus' suffix).

        Returns:
            :class:`Boxes3D` class containing the original `boxes` in the format specified by ``mode``.

        Examples:
            >>> boxes_xyzxyz = torch.as_tensor([[0, 3, 6, 1, 4, 8], [5, 1, 3, 8, 4, 9]])
            >>> boxes = Boxes3D.from_tensor(boxes_xyzxyz, mode='xyzxyz')
            >>> boxes.data  # (2, 8, 3)
            tensor([[[0., 3., 6.],
                     [0., 3., 6.],
                     [0., 3., 6.],
                     [0., 3., 6.],
                     [0., 3., 7.],
                     [0., 3., 7.],
                     [0., 3., 7.],
                     [0., 3., 7.]],
            <BLANKLINE>
                    [[5., 1., 3.],
                     [7., 1., 3.],
                     [7., 3., 3.],
                     [5., 3., 3.],
                     [5., 1., 8.],
                     [7., 1., 8.],
                     [7., 3., 8.],
                     [5., 3., 8.]]])
        """
        if not (2 <= boxes.ndim <= 3 and boxes.shape[-1] == 6):
            raise ValueError(f"BBox shape must be (N, 6) or (B, N, 6). Got {boxes.shape}.")

        batched = boxes.ndim == 3
        boxes = boxes if batched else boxes.unsqueeze(0)
        boxes = boxes if boxes.is_floating_point() else boxes.float()

        xmin, ymin, zmin = boxes[..., 0], boxes[..., 1], boxes[..., 2]
        mode = mode.lower()
        if mode == "xyzxyz":
            width = boxes[..., 3] - boxes[..., 0]
            height = boxes[..., 4] - boxes[..., 1]
            depth = boxes[..., 5] - boxes[..., 2]
        elif mode == "xyzxyz_plus":
            width = boxes[..., 3] - boxes[..., 0] + 1
            height = boxes[..., 4] - boxes[..., 1] + 1
            depth = boxes[..., 5] - boxes[..., 2] + 1
        elif mode == "xyzwhd":
            depth, height, width = boxes[..., 4], boxes[..., 3], boxes[..., 5]
        else:
            raise ValueError(f"Unknown mode {mode}")

        if validate_boxes:
            if (width <= 0).any():
                raise ValueError("Some boxes have negative widths or 0.")
            if (height <= 0).any():
                raise ValueError("Some boxes have negative heights or 0.")
            if (depth <= 0).any():
                raise ValueError("Some boxes have negative depths or 0.")

        hexahedrons = _boxes3d_to_polygons3d(xmin, ymin, zmin, width, height, depth)
        hexahedrons = hexahedrons if batched else hexahedrons.squeeze(0)
        # Due to some torch.jit.script bug (at least <= 1.9), you need to pass all arguments to __init__ when
        # constructing the class from inside of a method.
        return cls(hexahedrons, raise_if_not_floating_point=False, mode=mode)

    def to_tensor(self, mode: str = "xyzxyz") -> torch.Tensor:
        r"""Cast :class:`Boxes3D` to a tensor. ``mode`` controls which 3D boxes format should be use to represent boxes
        in the tensor.

        Args:
            mode: The format in which the boxes are provided.

                * 'xyzxyz': boxes are assumed to be in the format ``xmin, ymin, zmin, xmax, ymax, zmax`` where
                  ``width = xmax - xmin``, ``height = ymax - ymin`` and ``depth = zmax - zmin``.
                * 'xyzxyz_plus': similar to 'xyzxyz' mode but where box width, length and depth are defined as
                   ``width = xmax - xmin + 1``, ``height = ymax - ymin + 1`` and ``depth = zmax - zmin + 1``.
                * 'xyzwhd': boxes are assumed to be in the format ``xmin, ymin, zmin, width, height, depth`` where
                  ``width = xmax - xmin``, ``height = ymax - ymin`` and ``depth = zmax - zmin``.
                * 'vertices': boxes are defined by their vertices points in the following ``clockwise`` order:
                  *front-top-left, front-top-right, front-bottom-right, front-bottom-left, back-top-left,
                  back-top-right, back-bottom-right,  back-bottom-left*. Vertices coordinates are in (x,y, z) order.
                  Finally, box width, height and depth are defined as ``width = xmax - xmin``, ``height = ymax - ymin``
                  and ``depth = zmax - zmin``.
                * 'vertices_plus': similar to 'vertices' mode but where box width, length and depth are defined as
                  ``width = xmax - xmin + 1`` and ``height = ymax - ymin + 1``.

        Returns:
            3D Boxes tensor in the ``mode`` format. The shape depends with the ``mode`` value:

                * 'vertices' or 'verticies_plus': :math:`(N, 8, 3)` or :math:`(B, N, 8, 3)`.
                * Any other value: :math:`(N, 6)` or :math:`(B, N, 6)`.

        Note:
            It is currently non-differentiable due to a bug. See github issue
            `#1304 <https://github.com/kornia/kornia/issues/1396>`_.

        Examples:
            >>> boxes_xyzxyz = torch.as_tensor([[0, 3, 6, 1, 4, 8], [5, 1, 3, 8, 4, 9]])
            >>> boxes = Boxes3D.from_tensor(boxes_xyzxyz, mode='xyzxyz')
            >>> assert (boxes.to_tensor(mode='xyzxyz') == boxes_xyzxyz).all()
        """
        if self._data.requires_grad:
            raise RuntimeError("Boxes3D.to_tensor doesn't support computing gradients since they aren't accurate. "
                               "Please, create boxes from tensors with `requires_grad=False`. "
                               "This is a known bug. Help is needed to fix it. For more information, "
                               "see https://github.com/kornia/kornia/issues/1396.")

        batched_boxes = self._data if self._is_batched else self._data.unsqueeze(0)

        # Create boxes in xyzxyz_plus format.
        boxes = torch.stack([batched_boxes.amin(dim=-2), batched_boxes.amax(dim=-2)], dim=-2).view(
            batched_boxes.shape[0], batched_boxes.shape[1], 6
        )

        mode = mode.lower()
        if mode in ("xyzxyz", "xyzxyz_plus"):
            pass
        elif mode in ("xyzwhd", "vertices", "vertices_plus"):
            width = boxes[..., 3] - boxes[..., 0] + 1
            height = boxes[..., 4] - boxes[..., 1] + 1
            depth = boxes[..., 5] - boxes[..., 2] + 1
            boxes[..., 3] = width
            boxes[..., 4] = height
            boxes[..., 5] = depth
        else:
            raise ValueError(f"Unknown mode {mode}")

        if mode in ("xyzxyz", "vertices"):
            offset = torch.as_tensor([0, 0, 0, 1, 1, 1], device=boxes.device, dtype=boxes.dtype)
            boxes = boxes + offset

        if mode.startswith('vertices'):
            xmin, ymin, zmin = boxes[..., 0], boxes[..., 1], boxes[..., 2]
            width, height, depth = boxes[..., 3], boxes[..., 4], boxes[..., 5]

            boxes = _boxes3d_to_polygons3d(xmin, ymin, zmin, width, height, depth)

        boxes = boxes if self._is_batched else boxes.squeeze(0)
        return boxes

    def to_mask(self, depth: int, height: int, width: int) -> torch.Tensor:
        """Convert ·D boxes to masks. Covered area is 1 and the remaining is 0.

        Args:
            depth: depth of the masked image/images.
            height: height of the masked image/images.
            width: width of the masked image/images.

        Returns:
            the output mask tensor, shape of :math:`(N, depth, width, height)` or :math:`(B,N, depth, width, height)`
             and dtype of :func:`Boxes3D.dtype` (it can be any floating point dtype).

        Note:
            It is currently non-differentiable.

        Examples:
            >>> boxes = Boxes3D(torch.tensor([[  # Equivalent to boxes = Boxes.3Dfrom_tensor([[1,1,1,3,3,2]])
            ...     [1., 1., 1.],
            ...     [3., 1., 1.],
            ...     [3., 3., 1.],
            ...     [1., 3., 1.],
            ...     [1., 1., 2.],
            ...     [3., 1., 2.],
            ...     [3., 3., 2.],
            ...     [1., 3., 2.],
            ... ]]))  # 1x8x3
            >>> boxes.to_mask(4, 5, 5)
            tensor([[[[0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.]],
            <BLANKLINE>
                     [[0., 0., 0., 0., 0.],
                      [0., 1., 1., 1., 0.],
                      [0., 1., 1., 1., 0.],
                      [0., 1., 1., 1., 0.],
                      [0., 0., 0., 0., 0.]],
            <BLANKLINE>
                     [[0., 0., 0., 0., 0.],
                      [0., 1., 1., 1., 0.],
                      [0., 1., 1., 1., 0.],
                      [0., 1., 1., 1., 0.],
                      [0., 0., 0., 0., 0.]],
            <BLANKLINE>
                     [[0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.]]]])
        """
        if self._data.requires_grad:
            raise RuntimeError(
                "Boxes.to_tensor isn't differentiable. Please, create boxes from tensors with `requires_grad=False`."
            )

        if self._is_batched:  # (B, N, 8, 3)
            mask = torch.zeros(
                (self._data.shape[0], self._data.shape[1], depth, height, width),
                dtype=self._data.dtype,
                device=self._data.device,
            )
        else:  # (N, 8, 3)
            mask = torch.zeros(
                (self._data.shape[0], depth, height, width), dtype=self._data.dtype, device=self._data.device
            )

        # Boxes coordinates can be outside the image size after transforms. Clamp values to the image size
        clipped_boxes_xyzxyz = self.to_tensor("xyzxyz")
        clipped_boxes_xyzxyz[..., ::3].clamp_(0, width)
        clipped_boxes_xyzxyz[..., 1::3].clamp_(0, height)
        clipped_boxes_xyzxyz[..., 2::3].clamp_(0, depth)

        # Reshape mask to (BxN, D, H, W) and boxes to (BxN, 6) to iterate over all of them.
        # Cast boxes coordinates to be integer to use them as indexes. Use round to handle decimal values.
        for mask_channel, box_xyzxyz in zip(
            mask.view(-1, depth, height, width), clipped_boxes_xyzxyz.view(-1, 6).round().int()
        ):
            # Mask channel dimensions: (depth, height, width)
            mask_channel[
                box_xyzxyz[2] : box_xyzxyz[5], box_xyzxyz[1] : box_xyzxyz[4], box_xyzxyz[0] : box_xyzxyz[3]
            ] = 1

        return mask

    def transform_boxes(self, M: torch.Tensor, inplace: bool = False) -> "Boxes3D":
        r"""Apply a transformation matrix to the 3D boxes.

        Args:
            M: The transformation matrix to be applied, shape of :math:`(4, 4)` or :math:`(B, 4, 4)`.
            inplace: do transform in-place and return self.

        Returns:
            The transformed boxes.
        """
        if not 2 <= M.ndim <= 3 or M.shape[-2:] != (4, 4):
            raise ValueError(f"The transformation matrix shape must be (4, 4) or (B, 4, 4). Got {M.shape}.")

        # Due to some torch.jit.script bug (at least <= 1.9), you need to pass all arguments to __init__ when
        # constructing the class from inside of a method.
        transformed_boxes = _transform_boxes(self._data, M)
        if inplace:
            self._data = transformed_boxes
            return self

        return Boxes3D(transformed_boxes, False, "xyzxyz_plus")

    def transform_boxes_(self, M: torch.Tensor) -> "Boxes3D":
        """Inplace version of :func:`Boxes3D.transform_boxes`"""
        return self.transform_boxes(M, inplace=True)

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def device(self) -> torch.device:
        """Returns boxes device."""
        return self._data.device

    @property
    def dtype(self) -> torch.dtype:
        """Returns boxes dtype."""
        return self._data.dtype

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> "Boxes3D":
        """Like :func:`torch.nn.Module.to()` method."""
        # In torchscript, dtype is a int and not a class. https://github.com/pytorch/pytorch/issues/51941
        if dtype is not None and not _is_floating_point_dtype(dtype):
            raise ValueError("Boxes must be in floating point")
        self._data = self._data.to(device=device, dtype=dtype)
        return self
