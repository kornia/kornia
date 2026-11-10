kornia.geometry.pose
====================

.. meta::
   :description: The kornia.geometry.pose module provides NamedPose, a rigid SE(2) or SE(3) transform that carries the names of its source and destination frames, so that composing two poses checks that their frames chain.

.. currentmodule:: kornia.geometry.pose

:class:`NamedPose` holds an :class:`~kornia.geometry.liegroup.Se2` or :class:`~kornia.geometry.liegroup.Se3` pose
together with the names of its source and destination frames. A pose ``b_from_a`` maps points in frame ``a`` to
frame ``b``, and ``c_from_b * b_from_a`` is ``c_from_a``: a composition whose frames do not chain raises instead of
returning a wrong pose.

.. autoclass:: NamedPose
   :members:
   :special-members: __init__, __mul__
