ONNXSequential: Chain Multiple ONNX Models with Ease
====================================================

The `ONNXSequential` class is a powerful new feature that allows users to effortlessly combine and chain multiple ONNX models together. This is especially useful when you have several pre-trained models or custom ONNX operators that you want to execute sequentially as part of a larger pipeline.

Whether you're working with models for inference, experimentation, or optimization, `ONNXSequential` makes it easier to manage, combine, and run ONNX models in a streamlined manner. It also supports flexibility in execution environments with ONNXRuntime's execution providers (CPU, CUDA, etc.).

Key Features
------------

- **Seamless Model Chaining**: Combine multiple ONNX models into a single computational graph.
- **Flexible Input/Output Mapping**: Control how the outputs of one model are passed as inputs to the next.
- **Export to ONNX**: Save the combined model into a single ONNX file for easy deployment and sharing.
- **PyTorch-like Interface**: Use the `ONNXSequential` class like a PyTorch `nn.Sequential` model, including calling it directly for inference.

Optimized Execution
-------------------
- **ONNXRuntime**: Automatically create optimized `ONNXRuntime` sessions to speed up inference.
- **Execution Providers Support**: Utilize ONNXRuntime's execution providers (e.g., `CUDAExecutionProvider`, `CPUExecutionProvider`, `TensorrtExecutionProvider`, `OpenVINOExecutionProvider`) for accelerated inference on different hardware.
- **Concurrent Sessions**: You can manage multiple inference sessions concurrently, allowing for parallel processing of multiple inputs.
- **Asynchronous API**: We offer asyncio-based execution along with the runtime's asynchronous functions to perform non-blocking inference.

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
         "hf://operators/kornia.geometry.transform.flips.Hflip",
         # Or you may use a local model with either a filepath "YOUR_OWN_MODEL.onnx" or a loaded ONNX model.
         "hf://models/kornia.models.detection.rtdetr_r18vd_640x640",
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

**4. How to run with CUDA?

For using CUDA ONNXRuntime, you need to install `onnxruntime-gpu`.
For handling different CUDA version, you may refer to
https://github.com/microsoft/onnxruntime/issues/21769#issuecomment-2295342211.

For example, to install `onnxruntime-gpu==1.19.2` under CUDA 11.X, you may install with:

.. code-block:: console

   pip install onnxruntime-gpu==1.19.2 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/

You may then convert your sequence to CUDA, such as:

.. code-block:: python

   import kornia
   onnx_seq = ONNXSequential(
      "hf://operators/kornia.geometry.transform.flips.Hflip",
      "hf://models/kornia.models.detection.rtdetr_r18vd_640x640",  # Or you may use "YOUR_OWN_MODEL.onnx"
   )
   inp = kornia.utils.sample.get_sample_images()[0].numpy()[None]
   import time
   onnx_seq.as_cuda()
   onnx_seq(inp)  # GPU warm up
   start_time = time.time()
   onnx_seq(inp)
   print("--- GPU %s seconds ---" % (time.time() - start_time))
   onnx_seq.as_cpu()
   start_time = time.time()
   onnx_seq(inp)
   print("--- %s seconds ---" % (time.time() - start_time))

You may get a decent improvement:

.. code-block:: console

   --- GPU 0.014804363250732422 seconds ---
   --- CPU 0.17681646347045898 seconds ---

Why Choose ONNXSequential?
--------------------------

With the increasing adoption of ONNX for model interoperability and deployment, `ONNXSequential` provides a simple yet powerful interface for combining models and operators. By leveraging ONNXRuntime's optimization and execution provider capabilities, it gives you the flexibility to:
- Deploy on different hardware (CPU, GPU, TensorRT, OpenVINO, etc.).
- Run complex pipelines in production environments.
- Combine and experiment with models effortlessly.

Whether you're building an advanced deep learning pipeline or simply trying to chain pre-trained models, `ONNXSequential` makes it easy to manage, optimize, and execute ONNX models at scale.

Get started today and streamline your ONNX workflows!


API Documentation
-----------------
.. autoclass:: kornia.onnx.module.ONNXModule
    :members:

.. autoclass:: kornia.onnx.sequential.ONNXSequential
    :members:

.. autoclass:: kornia.onnx.utils.ONNXLoader

    .. code-block:: python

        # Load a HuggingFace operator
        ONNXLoader.load_model("hf://operators/kornia.color.gray.GrayscaleToRgb")  # doctest: +SKIP
        # Load a local converted/downloaded operator
        ONNXLoader.load_model("operators/kornia.color.gray.GrayscaleToRgb")  # doctest: +SKIP

    :members:
