Normalization
=============

.. currentmodule:: kornia.enhance

Normalization treats dimension 0 as batch and dimension 1 as channel. ZCA uses
its ``dim`` argument as the sample axis and flattens all other axes into features.

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
