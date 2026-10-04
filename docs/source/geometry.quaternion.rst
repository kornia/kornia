kornia.geometry.quaternion
==========================

.. meta::
   :description: The kornia.geometry.quaternion module provides tools for working with quaternions, a mathematical concept widely used in 3D geometry and computer vision. The Quaternion class allows for quaternion manipulation, including conversion between different representations like axis-angle and rotation matrices. This module is essential for operations involving 3D rotations and transformations.

.. currentmodule:: kornia.geometry.quaternion

Checkpoint compatibility
------------------------

Plain tensor quaternion data is saved as a persistent buffer under ``_data``.
The corresponding rotation keys are ``_q._data`` for ``So3`` and
``_rotation._q._data`` for ``Se3``, prefixed by the enclosing module's attribute names when nested.
Explicit ``nn.Parameter`` inputs remain parameters at construction and keep their existing checkpoint keys.

Older checkpoints omitted rotations constructed from plain tensors. Loading those checkpoints with
``strict=True`` now raises a missing-key error. ``strict=False`` reports the missing rotation keys and
leaves the target's initialized rotation unchanged; it cannot recover a rotation that was never saved.
Supply the intended rotation separately before using such a checkpoint.

.. autoclass:: Quaternion
   :members:
   :special-members:

.. autofunction:: average_quaternions
