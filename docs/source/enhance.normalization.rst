Normalization
=============

.. currentmodule:: kornia.enhance

Use :func:`normalize` with a known mean and standard deviation, :func:`normalize_min_max` to scale each channel plane
by its own range, and :func:`zca_whiten` or :class:`ZCAWhitening` to decorrelate features.

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
