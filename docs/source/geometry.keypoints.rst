kornia.geometry.keypoints
=========================

.. meta::
   :description: The kornia.geometry.keypoints module provides the Keypoints container for 2D (x, y) keypoints, which kornia.augmentation transforms together with the images, and the Keypoints3D container for 3D (x, y, z) points.

Keypoint containers. :class:`~kornia.geometry.keypoints.Keypoints` wraps an ``(N, 2)`` or ``(B, N, 2)`` tensor of
``(x, y)`` pixel coordinates, transforms it by ``3 x 3`` matrices and pads it with its image, as
:class:`~kornia.augmentation.container.AugmentationSequential` does with its ``'keypoints'`` data.
:class:`~kornia.geometry.keypoints.Keypoints3D` wraps an ``(N, 3)`` or ``(B, N, 3)`` tensor of ``(x, y, z)`` points
and transforms it by ``4 x 4`` matrices. Pixel ``(0, 0)`` is centred at ``(0, 0)``; see
:ref:`Coordinates and sizes <coordinate-conventions>`.

.. autoclass:: kornia.geometry.keypoints.Keypoints
   :members:
   :undoc-members:

.. autoclass:: kornia.geometry.keypoints.Keypoints3D
   :members:
   :undoc-members:
