kornia.geometry.bbox_v2
=======================

.. currentmodule:: kornia.geometry.bbox_v2

Module with useful functionalities for 2D and 3D bounding boxes manipulation.

2D bounding boxes
-----------------
**Kornia 2D bounding boxes format** is defined as a floating data type tensor of shape ``Nx4x2`` or ``BxNx4x2`` where each box is a `quadrilateral <https://en.wikipedia.org/wiki/Quadrilateral>`_ defined by it's 4 vertices coordinates (A, B, C, D). Coordinates must be in ``x, y`` order. The height and width of a box is defined as ``width = xmax - xmin`` and ``height = ymax - ymin``. Examples of `quadrilaterals <https://en.wikipedia.org/wiki/Quadrilateral>`_ are rectangles, rhombus and trapezoids.

.. autofunction:: bbox_to_kornia_bbox
.. autofunction:: bbox_to_mask
.. autofunction:: infer_bbox_shape
.. autofunction:: kornia_bbox_to_bbox
.. autofunction:: transform_bbox
.. autofunction:: validate_bbox

3D bounding boxes
-----------------
**Kornia 3D bounding boxes format** is defined as a floating data type tensor of shape ``Nx8x3`` or ``BxNx8x3`` where each box is a `hexahedron <https://en.wikipedia.org/wiki/Hexahedron>`_ defined by it's 8 vertices coordinates. Coordinates must be in ``x, y, z`` order. The height, width and depth of a box is defined as ``width = xmax - xmin``, ``height = ymax - ymin`` and ``depth = zmax - zmin``. Examples of `hexahedrons <https://en.wikipedia.org/wiki/Hexahedron>`_ are cubes and rhombohedrons.

.. autofunction:: bbox3d_to_kornia_bbox3d
.. autofunction:: bbox3d_to_mask3d
.. autofunction:: infer_bbox3d_shape
.. autofunction:: kornia_bbox3d_to_bbox3d
.. autofunction:: transform_bbox3d
.. autofunction:: validate_bbox3d
