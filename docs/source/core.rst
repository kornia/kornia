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
``.safetensors`` checkpoint; :func:`check_safetensors` is the ``validate=``
callable that lets the download half reject a truncated cache entry.

.. autofunction:: hf_url

.. autofunction:: load_state_dict_from_url

.. autofunction:: download_file_from_url

.. autofunction:: download_hf_file

.. autofunction:: load_safetensors

.. autofunction:: check_safetensors

Exceptions
----------

.. currentmodule:: kornia.core.exceptions

The errors raised by kornia's ``KORNIA_CHECK*`` validation helpers in
``kornia.core.check`` (and by a few functions directly), such as a tensor with
the wrong shape. All of them derive from :exc:`BaseError`, which derives from
:exc:`Exception`, so ``except BaseError`` catches any of them. It does not catch
every input error: many functions validate their inputs themselves and raise the
built-in :exc:`ValueError` or :exc:`TypeError` instead.

.. autoexception:: BaseError
    :show-inheritance:

.. autoexception:: ShapeError
    :show-inheritance:

.. autoexception:: TypeCheckError
    :show-inheritance:

.. autoexception:: ValueCheckError
    :show-inheritance:

.. autoexception:: DeviceError
    :show-inheritance:

.. autoexception:: ImageError
    :show-inheritance:
