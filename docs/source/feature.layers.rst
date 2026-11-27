Layers and other models
=======================

.. currentmodule:: kornia.feature

Layers
------

:class:`FilterResponseNorm2d` and :class:`TLU` are the normalisation and activation layers of :class:`HyNet`,
from the Filter Response Normalization paper. They take a ``(B, num_features, H, W)`` feature map and use no
batch statistics.

.. autoclass:: FilterResponseNorm2d
.. autoclass:: TLU

Fast moving objects
-------------------

:class:`DeFMO` deblurs a fast-moving object, given the blurred image and an estimate of its background, into 24
RGBA sub-frames of the object alone.

.. autoclass:: DeFMO
   :members: forward
