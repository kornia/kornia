Normalization
=============

.. currentmodule:: kornia.enhance

:func:`normalize` and :func:`denormalize` treat dimension 0 as batch and dimension 1 as channel;
:func:`normalize_min_max` takes :math:`(*, C, H, W)`. ZCA uses its ``dim`` argument as the sample
axis and flattens all other axes into features.

Functions
---------

.. autofunction:: normalize
.. autofunction:: normalize_min_max
.. autofunction:: denormalize
.. autofunction:: zca_mean
.. autofunction:: zca_whiten
.. autofunction:: linear_transform

Modules
-------

.. autoclass:: Normalize
.. autoclass:: Denormalize
.. autoclass:: ZCAWhitening
    :members:
