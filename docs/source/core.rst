kornia.core
===========

.. meta::
   :description: The kornia.core module in Kornia provides foundational classes and utilities for tensor manipulation. Key classes like TensorWrapper allow for enhanced handling of image tensors with support for various operations and transformations in computer vision tasks.

.. currentmodule:: kornia.core

.. autoclass:: TensorWrapper
    :members:
    :undoc-members:

.. autofunction:: kornia.core.utils.batched_forward

Weights
-------

Helpers for fetching and reading pretrained weights. :func:`load_state_dict_from_url`
is the one models load a ``torch.load``-able checkpoint with;
:func:`download_hf_file` (or :func:`download_file_from_url`) and
:func:`load_safetensors` are the two halves of the same job for a
``.safetensors`` checkpoint.

.. autofunction:: hf_url

.. autofunction:: load_state_dict_from_url

.. autofunction:: download_file_from_url

.. autofunction:: download_hf_file

.. autofunction:: load_safetensors
