from typing import Tuple

import torch

import kornia


def _transform_boxes(boxes: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Transforms 3D and 2D in kornia format by applying the transformation matrix M. Boxes and the transformation
    matrix could be batched or not.

    Args:
        boxes: 2D or 3D hexahedron in kornia format.
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
            f"Batch size mismatch. Got {points.shape[0]} for hexahedron, {M.shape[0]} for the transformation matrix."
        )

    transformed_boxes: torch.Tensor = kornia.transform_points(M, points)
    transformed_boxes = transformed_boxes.view_as(boxes)
    return transformed_boxes


def _boxes_to_polygons(
    xmin: torch.Tensor, ymin: torch.Tensor, width: torch.Tensor, height: torch.Tensor
) -> torch.Tensor:
    if not xmin.ndim == ymin.ndim == width.ndim == height.ndim == 2:
        raise ValueError("We expect to create a batch of hexahedron in vertices format (B, N, 4, 2)")

    # Create (B,N,4,2) with all points in top left position of the bounding box
    polygons = torch.zeros((xmin.shape[0], xmin.shape[1], 4, 2), device=xmin.device, dtype=xmin.dtype)
    polygons[..., 0] = xmin.unsqueeze(-1)
    polygons[..., 1] = ymin.unsqueeze(-1)
    # Shift top-right, bottom-right, bottom-left points to the right coordinates
    polygons[..., 1, 0] += width  # Top right
    polygons[..., 2, 0] += width  # Bottom right
    polygons[..., 2, 1] += height  # Bottom right
    polygons[..., 3, 1] += height  # Bottom left
    return polygons


def _boxes3d_to_polygons3d(
    xmin: torch.Tensor,
    ymin: torch.Tensor,
    zmin: torch.Tensor,
    width: torch.Tensor,
    height: torch.Tensor,
    depth: torch.Tensor,
):
    if not xmin.ndim == ymin.ndim == zmin.ndim == width.ndim == height.ndim == depth.ndim == 2:
        raise ValueError("We expect to create a batch of 3D hexahedrons in vertices format (B, N, 8, 3)")

    # Front
    # Create (B,N,4,3) with all points in front top left position of the bounding box
    front_vertices = torch.zeros((xmin.shape[0], xmin.shape[1], 4, 3), device=xmin.device, dtype=xmin.dtype)
    front_vertices[..., 0] = xmin.unsqueeze(-1)
    front_vertices[..., 1] = ymin.unsqueeze(-1)
    front_vertices[..., 2] = zmin.unsqueeze(-1)
    # Shift front-top-right, front-bottom-right, front-bottom-left points to the right coordinates
    front_vertices[..., 1, 0] += width  # Top right
    front_vertices[..., 2, 0] += width  # Bottom right
    front_vertices[..., 2, 1] += height  # Bottom right
    front_vertices[..., 3, 1] += height  # Bottom left

    # Back
    back_vertices = front_vertices.clone()
    back_vertices[..., 2] += depth.unsqueeze(-1)

    polygons3d = torch.cat([front_vertices, back_vertices], dim=-2)
    return polygons3d


@torch.jit.script
class Boxes:
    def __init__(self, quadrilaterals: torch.Tensor, raise_if_not_floating_point: bool = True):
        if not quadrilaterals.is_floating_point():
            if raise_if_not_floating_point:
                raise ValueError(f"Coordinates must be in floating point. Got {quadrilaterals.dtype}")
            else:
                quadrilaterals = quadrilaterals.float()

        if quadrilaterals.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            quadrilaterals = quadrilaterals.reshape((-1, 4))

        if not (3 <= quadrilaterals.ndim <= 4 and quadrilaterals.shape[-2:] == (4, 2)):
            raise ValueError(f"Boxes shape must be (N, 4, 2) or (B, N, 4, 2). Got {quadrilaterals.shape}.")

        self._boxes = quadrilaterals

    def boxes_shape(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Compute bounding boxes heights and widths.

        Returns:
            - Bounding box heights, shape of :math:`(N,)` or :math:`(B,N)`.
            - Boundingbox widths, shape of :math:`(N,)` or :math:`(B,N)`.

        Example:
            >>> boxes_xyxy = torch.tensor([[[
            ...     [1., 1.],
            ...     [2., 1.],
            ...     [2., 2.],
            ...     [1., 2.],
            ... ], [
            ...     [1., 1.],
            ...     [3., 1.],
            ...     [3., 2.],
            ...     [1., 2.],
            ... ]]])  # 1x2x4x2
            >>> boxes = Boxes.from_tensor(boxes_xyxy)
            >>> boxes.boxes_shape()
            (tensor([[1., 1.]]), tensor([[1., 2.]]))
        """
        boxes_xywh = self.to_tensor(mode='xywh')
        widths, heights = boxes_xywh[..., 2], boxes_xywh[..., 3]
        return heights, widths

    @classmethod
    def from_tensor(cls, boxes: torch.Tensor, mode: str = "xyxy") -> "Boxes":
        r"""Convert 2D bounding boxes to kornia format according to the format in which the boxes are provided.

        Args:
            boxes: 2D boxes to be transformed, shape of :math:`(N,4)` or :math:`(B,N,4)`.
            mode: The format in which the boxes are provided.

                * 'xyxy': boxes are assumed to be in the format ``xmin, ymin, xmax, ymax`` where ``width = xmax - xmin``
                  and ``height = ymax - ymin``.
                * 'xyxy_plus_1': like 'xyxy' where ``width = xmax - xmin + 1`` and  ``height = ymax - ymin + 1``.
                * 'xywh': boxes are assumed to be in the format ``xmin, ymin, width, height`` where
                  ``width = xmax - xmin`` and ``height = ymax - ymin``.
                * 'xywh_plus_1': like 'xywh' where ``width = xmax - xmin + 1`` and  ``height = ymax - ymin + 1``.

        Returns:
            2D Bounding boxes tensor in kornia format, shape of :math:`(N, 4, 2)` or :math:`(B, N, 4, 2)` and dtype of
            ``boxes`` if it's a floating point data type and ``float`` if not.

        Examples:
            >>> boxes_xyxy = torch.as_tensor([[0, 3, 1, 4], [5, 1, 8, 4]])
            >>> Boxes.from_tensor(boxes_xyxy, mode='xyxy')
        """
        if not (2 <= boxes.ndim <= 3 and boxes.shape[-1] == 4):
            raise ValueError(f"Boxes shape must be (N, 4) or (B, N, 4). Got {boxes.shape}.")

        batched = boxes.ndim == 3
        boxes = boxes if batched else boxes.unsqueeze(0)
        boxes = boxes if boxes.is_floating_point() else boxes.float()

        xmin, ymin = boxes[..., 0], boxes[..., 1]
        mode = mode.lower()
        if mode in ("xyxy", "xyxy_plus_1"):
            height, width = boxes[..., 3] - boxes[..., 1], boxes[..., 2] - boxes[..., 0]
        elif mode in ("xywh", "xywh_plus_1"):
            height, width = boxes[..., 3], boxes[..., 2]
        else:
            raise ValueError(f"Unknown mode {mode}")

        if mode.endswith("plus_1"):
            height = height - 1
            width = width - 1

        quadrilaterals = _boxes_to_polygons(xmin, ymin, width, height)
        quadrilaterals = quadrilaterals if batched else quadrilaterals.squeeze(0)
        # Due to some torch.jit.script bug (at least <= 1.9), you need to pass all arguments to __init__ when
        # constructing the class from inside of a method.
        return Boxes(quadrilaterals, False)

    def to_tensor(self, mode: str = "xyxy") -> torch.Tensor:
        r"""Convert 2D bounding boxes in kornia format to the format specified by ``mode``.

        Args:
            kornia_boxes: boxes to be transformed, shape of :math:`(N, 4, 2)` or :math:`(B, N, 4, 2)`.
            mode: the output box format. It could be:

                * 'xyxy': boxes are defined as ``xmin, ymin, xmax, ymax`` where ``width = xmax - xmin`` and
                  ``height = ymax - ymin``.
                * 'xyxy_plus_1': like 'xyxy' where ``width = xmax - xmin + 1`` and  ``height = ymax - ymin + 1``.
                * 'xywh': boxes are defined as ``xmin, ymin, width, height`` where ``width = xmax - xmin``
                  and ``height = ymax - ymin``.
                * 'xywh_plus_1': like 'xywh' where ``width = xmax - xmin + 1`` and  ``height = ymax - ymin + 1``.
                * 'vertices': boxes are defined by their vertices points in the following ``clockwise`` order:
                  *top-left, top-right, bottom-right, bottom-left*. Vertices coordinates are in (x,y) order. Finally,
                  box width and height are defined as ``width = xmax - xmin`` and ``height = ymax - ymin``.
                * 'vertices_plus_1': like 'vertices' where ``width = xmax - xmin + 1`` and ``height = ymax - ymin + 1``.

        Returns:
            Bounding boxes tensor the ``mode`` format. The shape is :math:`(N, 4, 2)` or :math:`(B, N, 4, 2)` in
            'vertices' or 'vertices_plus_1' mode and :math:`(N, 4)` or :math:`(B, N, 4)` in all other cases.

        Examples:
            >>> boxes_xyxy = torch.as_tensor([[0, 3, 1, 4], [5, 1, 8, 4]])
            >>> boxes = Boxes.from_tensor(boxes_xyxy)
            >>> assert (boxes_xyxy == boxes.to_tensor(mode='xyxy')).all()
        """
        batched_boxes = self._boxes if self._is_batch else self._boxes.unsqueeze(0)

        # Create boxes in xyxy format.
        boxes = torch.stack([batched_boxes.amin(dim=-2), batched_boxes.amax(dim=-2)], dim=-2).view(
            batched_boxes.shape[0], batched_boxes.shape[1], 4
        )

        mode = mode.lower()
        if mode in ("xyxy", "xyxy_plus_1"):
            pass
        elif mode in ("xywh", "xywh_plus_1", "vertices", "vertices_plus_1"):
            height, width = boxes[..., 3] - boxes[..., 1], boxes[..., 2] - boxes[..., 0]
            boxes[..., 2] = width
            boxes[..., 3] = height
        else:
            raise ValueError(f"Unknown mode {mode}")

        if mode.endswith("plus_1"):
            offset = torch.as_tensor([0, 0, 1, 1], device=boxes.device, dtype=boxes.dtype)
            boxes = boxes + offset

        if mode.startswith('vertices'):
            boxes = _boxes_to_polygons(boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3])

        boxes = boxes if self._is_batch else boxes.squeeze(0)
        return boxes

    def to_mask(self, height: int, width: int) -> torch.Tensor:
        """Convert 2D bounding boxes to masks. Covered area is 1 and the remaining is 0.

        Args:
            height: height of the masked image/images.
            width: width of the masked image/images.

        Returns:
            the output mask tensor, shape of :math:`(N, width, height)` or :math:`(B,N, width, height)` and dtype of
            boxes.

        Note:
            It is currently non-differentiable.

        Examples:
            >>> boxes_xyxy = torch.tensor([[
            ...        [1., 1.],
            ...        [4., 1.],
            ...        [4., 3.],
            ...        [1., 3.],
            ...   ]])  # 1x4x2
            >>> boxes = Boxes.from_tensor(boxes)
            >>> boxes.to_mask(5, 5)
            tensor([[[0., 0., 0., 0., 0.],
                     [0., 1., 1., 1., 0.],
                     [0., 1., 1., 1., 0.],
                     [0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0.]]])
        """
        if self._is_batch:  # (B, N, 4, 2)
            mask = torch.zeros(
                (self._boxes.shape[0], self._boxes.shape[1], height, width), dtype=self.dtype, device=self.device
            )
        else:  # (N, 4, 2)
            mask = torch.zeros((self._boxes.shape[0], height, width), dtype=self.dtype, device=self.device)

        # Boxes coordinates can be outside the image size after transforms. Clamp values to the image size
        clipped_boxes_xyxy = self.to_tensor("xyxy")
        clipped_boxes_xyxy[..., ::2].clamp_(0, width)
        clipped_boxes_xyxy[..., 1::2].clamp_(0, height)

        # Reshape mask to (BxN, H, W) and boxes to (BxN, 4) to iterate over all of them.
        # Cast boxes coordinates to be integer to use them as indexes. Use round to handle decimal values.
        for mask_channel, box_xyxy in zip(mask.view(-1, height, width), clipped_boxes_xyxy.view(-1, 4).round().int()):
            # Mask channel dimensions: (height, width)
            mask_channel[box_xyxy[1] : box_xyxy[3], box_xyxy[0] : box_xyxy[2]] = 1

        return mask

    def transform_boxes(self, M: torch.Tensor) -> "Boxes":
        r"""Function that applies a transformation matrix to the boxes.

        Args:
            M: The transformation matrix to be applied, shape of :math:`(3, 3)` or :math:`(B, 3, 3)`.
        Returns:
            The transformed boxes.
        """
        if not 2 <= M.ndim <= 3 or M.shape[-2:] != (3, 3):
            raise ValueError(f"The transformation matrix shape must be (3, 3) or (B, 3, 3). Got {M.shape}.")

        # Due to some torch.jit.script bug (at least <= 1.9), you need to pass all arguments to __init__ when
        # constructing the class from inside of a method.
        return Boxes(_transform_boxes(self._boxes, M), False)

    @property
    def _is_batch(self) -> bool:
        return self._boxes.ndim == 4

    @property
    def device(self) -> torch.device:
        return self._boxes.device

    @property
    def dtype(self) -> torch.dtype:
        return self._boxes.dtype


@torch.jit.script
class Boxes3D:
    def __init__(self, hexahedrons: torch.Tensor, raise_if_not_floating_point: bool = True):
        if not hexahedrons.is_floating_point():
            if raise_if_not_floating_point:
                raise ValueError(f"Coordinates must be in floating point. Got {hexahedrons.dtype}")
            else:
                hexahedrons = hexahedrons.float()

        if hexahedrons.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            hexahedrons = hexahedrons.reshape((-1, 6))

        if not (3 <= hexahedrons.ndim <= 4 and hexahedrons.shape[-2:] == (8, 3)):
            raise ValueError(f"3D bbox shape must be (N, 8, 3) or (B, N, 8, 3). Got {hexahedrons.shape}.")

        self._boxes = hexahedrons

    def boxes_shape(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Auto-infer the output sizes for the given 3D boxes.

        Returns:
            - Bounding box depths, shape of :math:`(N,)` or :math:`(B, N)`.
            - Bounding box heights, shape of :math:`(N,)` or :math:`(B, N)`.
            - Bounding box widths, shape of :math:`(N,)` or :math:`(B, N)`.

        Example:
            >>> boxes_xyzxyz = torch.tensor([[[ 0,  1,  2],
            ...         [10,  1,  2],
            ...         [10, 21,  2],
            ...         [ 0, 21,  2],
            ...         [ 0,  1, 32],
            ...         [10,  1, 32],
            ...         [10, 21, 32],
            ...         [ 0, 21, 32]],
            ...        [[ 3,  4,  5],
            ...         [43,  4,  5],
            ...         [43, 54,  5],
            ...         [ 3, 54,  5],
            ...         [ 3,  4, 65],
            ...         [43,  4, 65],
            ...         [43, 54, 65],
            ...         [ 3, 54, 65]]]) # 2x8x3
            >>> boxes3d = Boxes3D.from_tensor(boxes_xyzxyz)
            >>> boxes3d.boxes_shape()
            (tensor([30, 60]), tensor([20, 50]), tensor([10, 40]))
        """
        boxes_xyzwhd = self.to_tensor(mode='xyzwhd')
        widths, heights, depths = boxes_xyzwhd[..., 3], boxes_xyzwhd[..., 4], boxes_xyzwhd[..., 5]
        return depths, heights, widths

    @classmethod
    def from_tensor(cls, boxes: torch.Tensor, mode: str = "xyzxyz") -> "Boxes3D":
        r"""Convert 3D bounding boxes to kornia format according to the format in which the boxes are provided.

        Args:
            boxes: 3D boxes to be transformed, shape of :math:`(N,6)` or :math:`(B,N,6)`.
            mode: The format in which the boxes are provided.

                * 'xyzxyz': boxes are assumed to be in the format ``xmin, ymin, zmin, xmax, ymax, zmax`` where
                  ``width = xmax - xmin``, ``height = ymax - ymin`` and ``depth = zmax - zmin``.
                * 'xyzxyz_plus_1': like 'xyzxyz' where ``width = xmax - xmin + 1``, ``height = ymax - ymin + 1`` and
                  ``depth = zmax - zmin + 1``.
                * 'xyzwhd': boxes are assumed to be in the format ``xmin, ymin, zmin, width, height, depth`` where
                  ``width = xmax - xmin``, ``height = ymax - ymin`` and ``depth = zmax - zmin``.
                * 'xyzwhd_plus_1': like 'xyzwhd' where ``width = xmax - xmin + 1``, ``height = ymax - ymin + 1`` and
                  ``depth = zmax - zmin + 1``.
        Returns:
            3D bounding boxes tensor in kornia format, shape of :math:`(N, 8, 3)` or :math:`(B, N, 8, 3)` and dtype of
            ``boxes`` if it's a floating point data type and ``float`` if not.

        Examples:
            >>> boxes_xyzxyz = torch.as_tensor([[0, 3, 6, 1, 4, 8], [5, 1, 3, 8, 4, 9]])
            >>> Boxes3D.from_tensor(boxes_xyzxyz, mode='xyzxyz')
        """
        if not (2 <= boxes.ndim <= 3 and boxes.shape[-1] == 6):
            raise ValueError(f"BBox shape must be (N, 6) or (B, N, 6). Got {boxes.shape}.")

        batched = boxes.ndim == 3
        boxes = boxes if batched else boxes.unsqueeze(0)
        boxes = boxes if boxes.is_floating_point() else boxes.float()

        xmin, ymin, zmin = boxes[..., 0], boxes[..., 1], boxes[..., 2]
        mode = mode.lower()
        if mode in ("xyzxyz", "xyzxyz_plus_1"):
            width = boxes[..., 3] - boxes[..., 0]
            height = boxes[..., 4] - boxes[..., 1]
            depth = boxes[..., 5] - boxes[..., 2]
        elif mode in ("xyzwhd", "xyzwhd_plus_1"):
            depth, height, width = boxes[..., 4], boxes[..., 3], boxes[..., 5]
        else:
            raise ValueError(f"Unknown mode {mode}")

        if mode.endswith("plus_1"):
            height = height - 1
            width = width - 1
            depth = depth - 1

        hexahedrons = _boxes3d_to_polygons3d(xmin, ymin, zmin, width, height, depth)
        hexahedrons = hexahedrons if batched else hexahedrons.squeeze(0)
        # Due to some torch.jit.script bug (at least <= 1.9), you need to pass all arguments to __init__ when
        # constructing the class from inside of a method.
        return Boxes3D(hexahedrons, raise_if_not_floating_point=False)

    def to_tensor(self, mode: str = "xyzxyz") -> torch.Tensor:
        r"""Convert 3D bounding boxes in kornia format according to the format specified by ``mode``.

        Args:
            kornia_boxes: 3D boxes to be transformed, shape of :math:`(N, 8, 3)` or :math:`(B, N, 8, 3)`.
            mode: The format in which the boxes are provided.

                * 'xyzxyz': boxes are assumed to be in the format ``xmin, ymin, zmin, xmax, ymax, zmax`` where
                  ``width = xmax - xmin``, ``height = ymax - ymin`` and ``depth = zmax - zmin``.
                * 'xyzxyz_plus_1': like 'xyzxyz' where ``width = xmax - xmin + 1``, ``height = ymax - ymin + 1`` and
                  ``depth = zmax - zmin + 1``.
                * 'xyzwhd': boxes are assumed to be in the format ``xmin, ymin, zmin, width, height, depth`` where
                  ``width = xmax - xmin``, ``height = ymax - ymin`` and ``depth = zmax - zmin``.
                * 'xyzwhd_plus_1': like 'xyzwhd' where ``width = xmax - xmin + 1``, ``height = ymax - ymin + 1`` and
                  ``depth = zmax - zmin + 1``.
                * 'vertices': boxes are defined by their vertices points in the following ``clockwise`` order:
                  *front-top-left, front-top-right, front-bottom-right, front-bottom-left, back-top-left,
                  back-top-right, back-bottom-right,  back-bottom-left*. Vertices coordinates are in (x,y, z) order.
                  Finally, box width, height and depth are defined as ``width = xmax - xmin``, ``height = ymax - ymin``
                  and ``depth = zmax - zmin``.
                * 'vertices_plus_1': like 'vertices' where ``width = xmax - xmin + 1`` and ``height = ymax - ymin + 1``.

        Returns:
            3D bounding boxes tensor the ``mode`` format. The shape is :math:`(N, 8, 3)` or :math:`(B, N, 8, 3)` in
            'vertices' or 'vertices_plus_1' mode and :math:`(N, 6)` or :math:`(B, N, 6)` in all other cases.

        Examples:
            >>> boxes_xyzxyz = torch.as_tensor([[0, 3, 6, 1, 4, 8], [5, 1, 3, 8, 4, 9]])
            >>> kornia_bbox = bbox3d_to_kornia_bbox3d(boxes_xyzxyz, mode='xyzxyz')
            >>> assert (kornia_bbox3d_to_bbox3d(kornia_bbox, mode='xyzxyz') == boxes_xyzxyz).all()
        """
        batched_boxes = self._boxes if self._is_batch else self._boxes.unsqueeze(0)

        # Create boxes in xyzxyz format.
        boxes = torch.stack([batched_boxes.amin(dim=-2), batched_boxes.amax(dim=-2)], dim=-2).view(
            batched_boxes.shape[0], batched_boxes.shape[1], 6
        )

        mode = mode.lower()
        if mode in ("xyzxyz", "xyzxyz_plus_1"):
            pass
        elif mode in ("xyzwhd", "xyzwhd_plus_1", "vertices", "vertices_plus_1"):
            width = boxes[..., 3] - boxes[..., 0]
            height = boxes[..., 4] - boxes[..., 1]
            depth = boxes[..., 5] - boxes[..., 2]
            boxes[..., 3] = width
            boxes[..., 4] = height
            boxes[..., 5] = depth
        else:
            raise ValueError(f"Unknown mode {mode}")

        if mode.endswith("plus_1"):
            offset = torch.as_tensor([0, 0, 0, 1, 1, 1], device=boxes.device, dtype=boxes.dtype)
            boxes = boxes + offset

        if mode.startswith('vertices'):
            xmin, ymin, zmin = boxes[..., 0], boxes[..., 1], boxes[..., 2]
            width, height, depth = boxes[..., 3], boxes[..., 4], boxes[..., 5]

            boxes = _boxes3d_to_polygons3d(xmin, ymin, zmin, width, height, depth)

        boxes = boxes if self._is_batch else boxes.squeeze(0)
        return boxes

    def to_mask(self, depth: int, height: int, width: int) -> torch.Tensor:
        """Convert 3D bounding boxes to masks. Covered area is 1. and the remaining is 0.

        Args:
            depth: depth of the masked image/images.
            height: height of the masked image/images.
            width: width of the masked image/images.

        Returns:
            the output mask tensor, shape of :math:`(N, depth, width, height)` or :math:`(B,N, depth, width, height)`
            and dtype of boxes.

        Note:
            It is currently non-differentiable.

        Examples:
            >>> boxes = torch.tensor([[
            ...     [1., 1., 1.],
            ...     [3., 1., 1.],
            ...     [3., 3., 1.],
            ...     [1., 3., 1.],
            ...     [1., 1., 2.],
            ...     [3., 1., 2.],
            ...     [3., 3., 2.],
            ...     [1., 3., 2.],
            ... ]])  # 1x8x3
            >>> bbox3d_to_mask3d(boxes, 4, 5, 5)
            tensor([[[0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0.]],
            <BLANKLINE>
                    [[0., 0., 0., 0., 0.],
                     [0., 1., 1., 0., 0.],
                     [0., 1., 1., 0., 0.],
                     [0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0.]],
            <BLANKLINE>
                    [[0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0.]],
            <BLANKLINE>
                    [[0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0.]]])
        """

        if self._is_batch:  # (B, N, 8, 3)
            mask = torch.zeros(
                (self._boxes.shape[0], self._boxes.shape[1], depth, height, width),
                dtype=self._boxes.dtype,
                device=self._boxes.device,
            )
        else:  # (N, 8, 3)
            mask = torch.zeros(
                (self._boxes.shape[0], depth, height, width), dtype=self._boxes.dtype, device=self._boxes.device
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

    def transform_boxes(self, M: torch.Tensor) -> "Boxes3D":
        r"""Function that applies a transformation matrix to 3D boxes or a batch of 3D boxes in kornia format.

        Args:
            boxes: 3D boxes in kornia format..
            M: The transformation matrix to be applied, shape of :math:`(4, 4)` or :math:`(B, 4, 4)`.
        Returns:
            A tensor of shape :math:`(N, 8, 3)` or :math:`(B, N, 8, 3)` with the transformed 3D boxes in kornia format.
        """
        if not 2 <= M.ndim <= 3 or M.shape[-2:] != (4, 4):
            raise ValueError(f"The transformation matrix shape must be (4, 4) or (B, 4, 4). Got {M.shape}.")

        # Due to some torch.jit.script bug (at least <= 1.9), you need to pass all arguments to __init__ when
        # constructing the class from inside of a method.
        return Boxes3D(_transform_boxes(self._boxes, M), False)

    @property
    def _is_batch(self) -> bool:
        return self._boxes.ndim == 4

    @property
    def device(self) -> torch.device:
        return self._boxes.device

    @property
    def dtype(self) -> torch.dtype:
        return self._boxes.dtype
