Edge devices and Rust
=====================

.. meta::
   :description: Deploying Kornia pipelines outside Python: export to ONNX for ONNX Runtime on CPU, GPU, TensorRT or OpenVINO, or use kornia-rs, the Rust-native computer vision library from the same organization.

A trained pipeline does not have to ship with Python and PyTorch. Two paths take Kornia to production and
resource-constrained targets:

ONNX
----

Kornia operators and models export to ONNX, and :class:`~kornia.onnx.sequential.ONNXSequential` chains
pre-exported graphs into a single model that runs under ONNX Runtime with the CPU, CUDA, TensorRT or OpenVINO
execution providers. Pre-exported operators and models are published in the
`kornia/ONNX_models <https://huggingface.co/kornia/ONNX_models>`_ Hugging Face repository.
See the :doc:`ONNX support guide <onnx>`.

kornia-rs
---------

`kornia-rs <https://github.com/kornia/kornia-rs>`_ is the Rust-native computer vision library developed by the
same organization: a from-scratch implementation designed for safety-critical and embedded applications, with
first-class support for camera I/O and efficient image decoding. It is what already powers :mod:`kornia.io`
inside this library, through the ``kornia_rs`` Python package.

- Source and documentation: `github.com/kornia/kornia-rs <https://github.com/kornia/kornia-rs>`_
- Rust crate: `crates.io/crates/kornia <https://crates.io/crates/kornia>`_ (API docs on
  `docs.rs <https://docs.rs/kornia>`_)

If your deployment target is a robot, a camera or another device where a Python runtime is unwanted, kornia-rs
is the intended path; reach out on `Discord <https://discord.gg/HfnywwpBnD>`_ if you are building on it.
