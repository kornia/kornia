kornia.geometry.keypoints
=========================

.. meta::
   :description: The kornia.geometry.keypoints module provides the Keypoints container for 2D (x, y) keypoints, which kornia.augmentation transforms together with the images, and the Keypoints3D container for 3D (x, y, z) points.

Object-oriented API for keypoints: :class:`~kornia.geometry.keypoints.Keypoints` wraps a tensor of 2D ``(x, y)``
points and transforms and pads it; :class:`~kornia.geometry.keypoints.Keypoints3D` validates and stores 3D
``(x, y, z)`` points.

.. autoclass:: kornia.geometry.keypoints.Keypoints
   :members:
   :undoc-members:

.. autoclass:: kornia.geometry.keypoints.Keypoints3D
   :members:
   :undoc-members:
