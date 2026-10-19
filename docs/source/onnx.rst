kornia.onnx
===========

.. meta::
   :description: API reference for kornia.onnx: ONNXModule, ONNXSequential to chain ONNX models and Kornia operators into one graph, and ONNXLoader to fetch pre-exported operators and models from the Hugging Face Hub.

.. currentmodule:: kornia.onnx

Run and compose ONNX models with ONNX Runtime: :class:`~kornia.onnx.sequential.ONNXSequential` chains several
ONNX graphs (local files, ``onnx.ModelProto`` objects, or pre-exported Kornia operators and models from the
``kornia/ONNX_models`` Hugging Face repository) into one model with a PyTorch-like call interface. The
:doc:`ONNX support guide </get-started/onnx>` walks through installation, chaining, export and running on CUDA.

.. autoclass:: kornia.onnx.module.ONNXModule
    :members:

.. autoclass:: kornia.onnx.sequential.ONNXSequential
    :members:

.. autoclass:: kornia.onnx.utils.ONNXLoader
    :members:

    .. code-block:: python

        from kornia.onnx.utils import ONNXLoader

        # Load a Hugging Face operator
        ONNXLoader.load_model("hf://operators/kornia.color.gray.GrayscaleToRgb")
        # Load a local converted/downloaded operator
        ONNXLoader.load_model("operators/kornia.color.gray.GrayscaleToRgb")
