kornia.geometry.bbox
==========================

Module with useful functionalities for 2D and 3D bounding boxes manipulation.

Kornia bounding boxes format is defined as a a floating data type tensor of shape Nx4x2 or BxNx4x2 where each box is a `quadrilateral <https://en.wikipedia.org/wiki/Quadrilateral>`_ defined by it's 4 vertices coordinates (A, B, C, D). Coordinates must be in the x, y order. The height and width of a box is ``width = xmax - xmin`` and ``height = ymax - ymin``. Examples of `quadrilaterals <https://en.wikipedia.org/wiki/Quadrilateral>`_ are rectangles, rhombus and trapezoids.

Kornia 3D bounding boxes format is defined as a a floating data type tensor of shape Nx8x3 or BxNx8x3 where each box is a `hexahedron <https://en.wikipedia.org/wiki/Hexahedron>`_ defined by it's 8 vertices coordinates. Coordinates must be in the x, y, z order. The height, width and depth of a box is ``width = xmax - xmin``, ``height = ymax - ymin`` and ``depth = zmax - zmin``. Examples of `hexahedrons <https://en.wikipedia.org/wiki/Hexahedron>`_ are cubes and rhombohedrons.

.. automodule:: kornia.geometry.bbox
    :members:
