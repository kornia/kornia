ONNXSequential: Chain Multiple ONNX Models with Ease
====================================================

The `ONNXSequential` class is a powerful new feature that allows users to effortlessly combine and chain multiple ONNX models together. This is especially useful when you have several pre-trained models or custom ONNX operators that you want to execute sequentially as part of a larger pipeline.

Whether you're working with models for inference, experimentation, or optimization, `ONNXSequential` makes it easier to manage, combine, and run ONNX models in a streamlined manner. It also supports flexibility in execution environments with ONNXRuntime’s execution providers (CPU, CUDA, etc.).

Key Features
------------

- **Seamless Model Chaining**: Combine multiple ONNX models into a single computational graph.
- **Flexible Input/Output Mapping**: Control how the outputs of one model are passed as inputs to the next.
- **Optimized Execution**: Automatically create optimized `ONNXRuntime` sessions to speed up inference.
- **Export to ONNX**: Save the combined model into a single ONNX file for easy deployment and sharing.
- **Execution Providers Support**: Utilize ONNXRuntime's execution providers (e.g., `CUDAExecutionProvider`, `CPUExecutionProvider`) for accelerated inference on different hardware.
- **PyTorch-like Interface**: Use the `ONNXSequential` class like a PyTorch `nn.Sequential` model, including calling it directly for inference.

Quickstart Guide
----------------

Here's how you can quickly get started with `ONNXSequential`:

1. **Install ONNX and ONNXRuntime**

   If you haven't already installed `onnx` and `onnxruntime`, you can install them using `pip`:

   .. code-block:: bash

      pip install onnx onnxruntime

2. **Combining ONNX Models**

   You can initialize the `ONNXSequential` with a list of ONNX models or file paths. Models will be automatically chained together and optimized for inference.

   .. code-block:: python

      import numpy as np
      from kornia.onnx import ONNXSequential

      # Initialize ONNXSequential with two models, loading from our only repo
      onnx_seq = ONNXSequential(
         "hf://operators/kornia.color.gray.RgbToGrayscale",
         "hf://operators/kornia.geometry.transform.affwarp.Resize_512x512"
      )

      # Prepare some input data
      input_data = np.random.randn(1, 3, 256, 512).astype(np.float32)

      # Perform inference
      outputs = onnx_seq(input_data)

      # Print the model outputs
      print(outputs)

   .. note::
      By default, we assume each ONNX model contains only one input node named "input" and one output node named "output". For complex models, you may need to pass an `io_maps` argument.

3. **Input/Output Mapping Between Models**

   When combining models, you can specify how the outputs of one model are mapped to the inputs of the next. This allows you to chain models in custom ways.

   .. code-block:: python

      io_map = [("model1_output_0", "model2_input_0"), ("model1_output_1", "model2_input_1")]
      onnx_seq = ONNXSequential("model1.onnx", "model2.onnx", io_map=io_map)

4. **Exporting the Combined Model**

   You can easily export the combined model to an ONNX file:

   .. code-block:: python

      # Export the combined model to a file
      onnx_seq.export("combined_model.onnx")

5. **Optimizing with Execution Providers**

   Leverage ONNXRuntime's execution providers for optimized inference. For example, to run the model on a GPU:

   .. code-block:: python

      # Initialize with CUDA execution provider
      onnx_seq = ONNXSequential(
         "hf://operators/kornia.color.gray.RgbToGrayscale",
         "hf://operators/kornia.geometry.transform.affwarp.Resize_512x512",
         providers=['CUDAExecutionProvider']
      )

      # Run inference
      outputs = onnx_seq(input_data)


Frequently Asked Questions (FAQ)
--------------------------------

**1. Can I chain models from different sources?**

Yes! You can chain models from different ONNX files or directly from `onnx.ModelProto` objects. `ONNXSequential` handles the integration and merging of their graphs.

**2. What happens if the input/output sizes of models don't match?**

You can use the `io_map` parameter to control how outputs of one model are mapped to the inputs of the next. This allows for greater flexibility when chaining models with different architectures.

**3. Can I use custom ONNXRuntime session options?**

Absolutely! You can pass your own session options to the `create_session` method to fine-tune performance, memory usage, or logging.

Why Choose ONNXSequential?
--------------------------

With the increasing adoption of ONNX for model interoperability and deployment, `ONNXSequential` provides a simple yet powerful interface for combining models and operators. By leveraging ONNXRuntime’s optimization and execution provider capabilities, it gives you the flexibility to:
- Deploy on different hardware (CPU, GPU).
- Run complex pipelines in production environments.
- Combine and experiment with models effortlessly.

Whether you're building an advanced deep learning pipeline or simply trying to chain pre-trained models, `ONNXSequential` makes it easy to manage, optimize, and execute ONNX models at scale.

Get started today and streamline your ONNX workflows!


API Documentation
-----------------
.. autoclass:: kornia.onnx.sequential.ONNXSequential
    :members:
