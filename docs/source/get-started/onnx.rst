ONNX support
============

.. meta::
   :description: How to export Kornia operators and models to ONNX and chain them with ONNXSequential: installation, combining models, input/output mapping, export to a single file, and running on CUDA with ONNX Runtime execution providers.

:class:`~kornia.onnx.sequential.ONNXSequential` combines several ONNX models into a single computational graph and
runs it with ONNX Runtime. It is useful when you have pre-exported operators or models, from Kornia's
``kornia/ONNX_models`` Hugging Face repository or your own, that should run one after another as a pipeline.

- **Model chaining**: combine multiple ONNX models into one graph, with control over how the outputs of one model
  feed the inputs of the next.
- **Export**: save the combined model as a single ONNX file.
- **PyTorch-like interface**: call the sequence directly, like an ``nn.Sequential``.
- **Execution providers**: run on CPU, CUDA, TensorRT or OpenVINO through ONNX Runtime, with control over the
  session options.

Most of the library exports: |count-onnx-share| of Kornia's public operators (|count-onnx-exportable| of the
|count-onnx-operators| surveyed) go through ``torch.onnx.export`` and run under ONNX Runtime. The
:ref:`ONNX, torch.compile and torch.export support <export-support>` page lists the outcome for every operator,
model and augmentation, with the cause of each failure.

Quickstart
----------

1. **Install ONNX and ONNX Runtime**

   .. code-block:: bash

      pip install onnx onnxruntime

2. **Port your own kornia module**

   Any kornia operator or ``nn.Sequential`` of operators is a regular ``nn.Module``, so
   ``torch.onnx.export`` turns it into an ONNX file that :class:`~kornia.onnx.sequential.ONNXSequential`
   can load, chain and re-export. Use the legacy exporter at opset 17 (``dynamo=False``) so the IR
   version matches the pre-exported Hub operators:

   .. code-block:: python

      import torch
      import kornia

      torch.onnx.export(
          kornia.color.RgbToGrayscale(),
          torch.rand(1, 3, 256, 512),
          "gray.onnx",
          input_names=["input"],
          output_names=["output"],
          dynamic_axes={"input": {0: "B", 2: "H", 3: "W"}, "output": {0: "B", 2: "H", 3: "W"}},
          opset_version=17,
          dynamo=False,
      )

   The file plugs into a sequence like any other model: ``ONNXSequential("gray.onnx", ...)``.

3. **Combine ONNX models**

   Initialize :class:`~kornia.onnx.sequential.ONNXSequential` with ONNX models or file paths. The models are chained
   and an optimized inference session is created.

   .. code-block:: python

      import numpy as np
      from kornia.onnx import ONNXSequential

      # Two operators from the kornia/ONNX_models Hugging Face repo
      onnx_seq = ONNXSequential(
         "hf://operators/kornia.color.gray.RgbToGrayscale",
         "hf://operators/kornia.geometry.transform.affwarp.Resize_512x512"
      )

      input_data = np.random.randn(1, 3, 256, 512).astype(np.float32)
      outputs = onnx_seq(input_data)
      print(outputs)

   .. note::
      By default each ONNX model is assumed to have one input node named ``"input"`` and one output node named
      ``"output"``. For other models, pass an ``io_maps`` argument.

4. **Map inputs and outputs between models**

   .. code-block:: python

      io_maps = [("model1_output_0", "model2_input_0"), ("model1_output_1", "model2_input_1")]
      onnx_seq = ONNXSequential("model1.onnx", "model2.onnx", io_maps=io_maps)

5. **Export the combined model**

   .. code-block:: python

      onnx_seq.export("combined_model.onnx")

6. **Pick an execution provider**

   .. code-block:: python

      onnx_seq = ONNXSequential(
         "hf://operators/kornia.geometry.transform.flips.Hflip",
         # a local model works too: "YOUR_OWN_MODEL.onnx", or a loaded onnx.ModelProto
         "hf://models/kornia.models.detection.rtdetr_r18vd_640x640",
         providers=['CUDAExecutionProvider']
      )
      outputs = onnx_seq(input_data)

Running on CUDA
---------------

CUDA execution needs ``onnxruntime-gpu``; for other CUDA versions see
https://github.com/microsoft/onnxruntime/issues/21769#issuecomment-2295342211. For example, to install
``onnxruntime-gpu==1.19.2`` for CUDA 11.x:

.. code-block:: console

   pip install onnxruntime-gpu==1.19.2 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/

Then move a sequence between devices with ``as_cuda()`` / ``as_cpu()``:

.. code-block:: python

   import time

   import kornia
   from kornia.onnx import ONNXSequential

   onnx_seq = ONNXSequential(
      "hf://operators/kornia.geometry.transform.flips.Hflip",
      "hf://models/kornia.models.detection.rtdetr_r18vd_640x640",  # Or you may use "YOUR_OWN_MODEL.onnx"
   )
   inp = kornia.io.get_sample_images()[0].numpy()[None]

   onnx_seq.as_cuda()
   onnx_seq(inp)  # GPU warm-up
   start_time = time.time()
   onnx_seq(inp)
   print("--- GPU %s seconds ---" % (time.time() - start_time))

   onnx_seq.as_cpu()
   start_time = time.time()
   onnx_seq(inp)
   print("--- CPU %s seconds ---" % (time.time() - start_time))

A typical result:

.. code-block:: console

   --- GPU 0.014804363250732422 seconds ---
   --- CPU 0.17681646347045898 seconds ---

Frequently asked questions
--------------------------

**Can I chain models from different sources?**
Yes. Models can come from ONNX files, ``onnx.ModelProto`` objects or ``hf://`` references;
:class:`~kornia.onnx.sequential.ONNXSequential` merges their graphs.

**What if the input/output names of two models do not match?**
Use the ``io_maps`` argument to say which output feeds which input.

**Can I use custom ONNX Runtime session options?**
Yes, pass your own session options to the ``create_session`` method.

**Which operators export?**
The :ref:`ONNX, torch.compile and torch.export support <export-support>` page lists every surveyed operator with
its ONNX export, ``torch.export`` and ``torch.compile`` outcome, searchable by name and filterable by result.

**Where are the pre-exported Kornia operators?**
In the `kornia/ONNX_models <https://huggingface.co/kornia/ONNX_models>`_ Hugging Face repository; load them with
``"hf://operators/<name>"`` and ``"hf://models/<name>"`` or with :class:`~kornia.onnx.utils.ONNXLoader`.

See the :doc:`kornia.onnx </onnx>` API reference for the full class documentation.
