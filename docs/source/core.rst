kornia.core
===========

.. meta::
   :description: The kornia.core module provides the helpers that download and read pretrained weights, the ImageModule and ONNX base classes, TensorWrapper and small batched tensors, the device and autocast helpers, and, with them, the kornia.config settings and the kornia.constants enums.

.. currentmodule:: kornia.core

Building blocks shared across kornia: functions that download and read pretrained weights, base classes that give a
module NumPy and PIL inputs and outputs or an ONNX export, small tensor helpers, and the device and autocast helpers of
``kornia.core.utils``. The process-wide settings of ``kornia.config`` and the enums of ``kornia.constants`` that the
augmentations take are documented here too.

Weights
-------

Helpers for fetching and reading pretrained weights. :func:`load_state_dict_from_url` is the one models load a
``torch.load``-able checkpoint with; :func:`download_hf_file` (or :func:`download_file_from_url`) and
:func:`load_safetensors` are the two halves of the same job for a ``.safetensors`` checkpoint; :func:`check_safetensors`
is the ``validate=`` callable that lets the download half reject a truncated cache entry.

.. autofunction:: hf_url

.. autofunction:: load_state_dict_from_url

.. autofunction:: download_file_from_url

.. autofunction:: download_hf_file

.. autofunction:: load_safetensors

.. autofunction:: check_safetensors

Image modules
-------------

Subclass :class:`ImageModule`, or chain modules in :class:`ImageSequential`, for a module whose call also takes NumPy
arrays, PIL images and image paths and returns tensors, NumPy arrays or PIL images; :class:`ImageModuleMixIn` holds the
conversions. For augmentation pipelines, use :class:`~kornia.augmentation.container.ImageSequential` instead.

.. autoclass:: ImageModule
    :members:

.. autoclass:: ImageSequential
    :members:

.. autoclass:: ImageModuleMixIn
    :members:

ONNX export and runtime
-----------------------

The mixins behind ``to_onnx`` and :class:`~kornia.onnx.sequential.ONNXSequential`. They need the ``kornia[onnx]``
extra.

.. autoclass:: ONNXExportMixin
    :members:

.. autoclass:: ONNXRuntimeMixin
    :members:

.. autoclass:: ONNXMixin

Tensor helpers
--------------

.. autoclass:: TensorWrapper
    :members:
    :undoc-members:

.. autofunction:: eye_like

.. autofunction:: vec_like

.. autofunction:: kornia.core.utils.batched_forward

Device and autocast helpers
---------------------------

.. autofunction:: kornia.core.utils.get_cuda_device_if_available

.. autofunction:: kornia.core.utils.get_mps_device_if_available

.. autofunction:: kornia.core.utils.get_cuda_or_mps_device_if_available

.. autofunction:: kornia.core.utils.is_autocast_enabled

Validation
----------

.. currentmodule:: kornia.core.check

The ``KORNIA_CHECK*`` helpers validate inputs: each returns ``True`` when its check passes, and on failure raises one
of the errors below or, with ``raises=False``, returns ``False``. Pass a message that formats a size as a function; see
:func:`KORNIA_CHECK`. The shape check of local affine frames, :func:`~kornia.feature.KORNIA_CHECK_LAF`, is documented
with the other LAF functions.

.. autofunction:: KORNIA_CHECK

Shapes and sizes
^^^^^^^^^^^^^^^^

.. autofunction:: KORNIA_CHECK_SHAPE

.. autofunction:: KORNIA_CHECK_SIZES

.. autofunction:: KORNIA_CHECK_SAME_SIZE

.. autofunction:: KORNIA_CHECK_SAME_SHAPE

.. autofunction:: KORNIA_CHECK_DM_DESC

Image layouts
^^^^^^^^^^^^^

.. autofunction:: KORNIA_CHECK_IS_COLOR

.. autofunction:: KORNIA_CHECK_IS_GRAY

.. autofunction:: KORNIA_CHECK_IS_COLOR_OR_GRAY

.. autofunction:: KORNIA_CHECK_IS_IMAGE

Types and devices
^^^^^^^^^^^^^^^^^

.. autofunction:: KORNIA_CHECK_IS_TENSOR

.. autofunction:: KORNIA_CHECK_TYPE

.. autofunction:: KORNIA_CHECK_IS_LIST_OF_TENSOR

.. autofunction:: KORNIA_CHECK_SAME_DEVICE

.. autofunction:: KORNIA_CHECK_SAME_DEVICES

.. autofunction:: KORNIA_UNWRAP

Turning the checks off and on
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: disable_checks

.. autofunction:: enable_checks

.. autofunction:: are_checks_enabled

Exceptions
----------

.. currentmodule:: kornia.core.exceptions

The errors raised by the ``KORNIA_CHECK*`` helpers above and by functions that raise them directly.
``except BaseError`` catches every one of them. Each subclass also derives from the built-in exception that matches its
failure, so the matching ``except`` clause catches it together with the built-in errors that other functions raise:

- :exc:`ShapeError`, :exc:`ValueCheckError`, :exc:`DeviceError` and :exc:`ImageError` are each also a
  :exc:`ValueError`, and :exc:`TypeCheckError` is also a :exc:`TypeError`.
- :exc:`InputRankError`, raised by the augmentations and their containers for an input with an unsupported number of
  dimensions, is a :exc:`ShapeError` and also a :exc:`RuntimeError` and an :exc:`AssertionError`, the types those entry
  points raised for it before.
- :exc:`AsyncCheckError`, the host failure of a value check that an accelerator runs as ``torch._assert_async``, is a
  :exc:`ValueCheckError` and also a :exc:`RuntimeError`, the error torch raises for that check on the device.

:exc:`BaseError` itself derives only from :exc:`Exception`: a failed ``KORNIA_CHECK(condition)`` raises it, and neither
``except ValueError`` nor ``except TypeError`` catches it. In a call compiled with ``torch.jit.script``, a failed
check raises ``torch.jit.Error`` instead, which none of these clauses catches.

.. autoexception:: BaseError
    :show-inheritance:

.. autoexception:: ShapeError
    :show-inheritance:

.. autoexception:: InputRankError
    :show-inheritance:

.. autoexception:: TypeCheckError
    :show-inheritance:

.. autoexception:: ValueCheckError
    :show-inheritance:

.. autoexception:: AsyncCheckError
    :show-inheritance:

.. autoexception:: DeviceError
    :show-inheritance:

.. autoexception:: ImageError
    :show-inheritance:

Configuration
-------------

.. currentmodule:: kornia.config

``kornia.config.kornia_config`` holds the process-wide settings: ``lazyloader.installation_mode`` takes an
:class:`InstallationMode`, and ``hub_cache_dir`` and ``hub_onnx_dir`` set where sample images and ONNX models are
downloaded to.

.. autoclass:: InstallationMode
    :members:
    :undoc-members:

Constants
---------

.. currentmodule:: kornia.constants

The enums that augmentations take for ``resample``, ``padding_mode`` and ``border_type``, and the dtype code of the mix
augmentations. Each enum's ``get`` takes a member, its name in any case or its integer value and returns the member, so
those arguments accept all three. A name that names no member raises an error that is both a ``KeyError`` and a
``ValueError``. :class:`DataKey`, for the ``data_keys`` of the augmentation containers, is documented with
:doc:`the containers <augmentation.container>`.

.. autoclass:: Resample
    :members:
    :undoc-members:

.. autoclass:: SamplePadding
    :members:
    :undoc-members:

.. autoclass:: BorderType
    :members:
    :undoc-members:

.. autoclass:: DType
    :members:
    :undoc-members:
