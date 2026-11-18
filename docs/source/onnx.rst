kornia.onnx
===========

.. meta::
   :description: API reference for kornia.onnx: ONNXModule, ONNXSequential to chain ONNX models and Kornia operators into one graph, and ONNXLoader to fetch pre-exported operators and models from the Hugging Face Hub.

.. currentmodule:: kornia.onnx

Run ONNX models with ONNX Runtime: :class:`~kornia.onnx.module.ONNXModule` (or :func:`load`) runs one model, and
:class:`~kornia.onnx.sequential.ONNXSequential` chains several into one graph. Both take ``onnx.ModelProto`` objects,
local files, URLs, or ``hf://`` names of pre-exported Kornia operators and models in the ``kornia/ONNX_models``
Hugging Face repository, which :class:`~kornia.onnx.utils.ONNXLoader` downloads into the cache; later calls use the
cached file unless ``force_download=True``. Both are called on numpy arrays or CPU torch tensors and return a list of
numpy arrays. The :doc:`ONNX support guide </get-started/onnx>` walks through installation, chaining, export and
running on CUDA.

.. autoclass:: kornia.onnx.module.ONNXModule
    :members:

.. autofunction:: kornia.onnx.load

.. autoclass:: kornia.onnx.sequential.ONNXSequential
    :members:

.. autoclass:: kornia.onnx.utils.ONNXLoader
    :members:

.. autofunction:: kornia.onnx.utils.add_metadata

.. autofunction:: kornia.onnx.utils.io_name_conversion

.. autoclass:: kornia.onnx.download.CachedDownloader
    :members:
