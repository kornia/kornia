kornia.augmentation
===================

.. meta::
   :name: description
   :content: "The Augmentation module in Kornia provides high-level data augmentation functionalities for computer vision tasks, including random rotations, affine transformations, color intensities, image noise distortion, and more. It supports batch processing, device compatibility, and backpropagation. Additionally, users can retrieve transformation details for more flexibility in complex pipelines."

This module implements in a high level logic. The main features of this module, and similar to the rest of the
library, is that can it perform data augmentation routines in a batch mode, using any supported device,
and can be used for backpropagation. Some of the available functionalities which are worth to mention are the
following: random rotations; affine and perspective transformations; several random color intensities transformations,
image noise distortion, motion blurring, and many of the different differentiable data augmentation policies.
In addition, we include a novel feature which is not found in other augmentations frameworks,
which allows the user to retrieve the applied transformation or chained transformations after each
call e.g. the generated random rotation matrix which can be used later to undo the image transformation
itself, or to be applied to additional metadata such as the label images for semantic segmentation,
in bounding boxes or landmark keypoints for object detection tasks. It gives the user the flexibility to
perform complex data augmentations pipelines.

.. note::
   **Input format.** Kornia augmentations expect **float tensors with values in** ``[0, 1]`` and shape
   ``(B, C, H, W)`` — the same format produced by ``torchvision.transforms.v2.ToDtype(torch.float32, scale=True)``
   (or the legacy ``ToTensor``). A ``uint8`` tensor raises a clear error, but a **float tensor already in the**
   ``[0, 255]`` **range does not raise** — it is silently treated as ``[0, 1]`` and the output clips at ``1.0``,
   which shows up later as degraded model accuracy rather than an exception. Scale first, e.g. ``img.float() / 255``.

Interactive Demo
~~~~~~~~~~~~~~~~
.. raw:: html

   <iframe
      id="augmentation-tester"
      src="https://kornia-kornia-augmentations-tester.hf.space"
      frameborder="0"
      width="850"
      height="450"
   ></iframe>

Benchmark
---------

Kornia is **GPU-batched and differentiable** — that is the regime it is built to lead, and it is
not the regime the other libraries target. A fair comparison holds the batch size and device fixed
across libraries; comparing kornia at a large batch against a single-image library is not meaningful.

Reproducible, honestly-framed benchmarks live under
`benchmarks/augmentation/ <https://github.com/kornia/kornia/tree/main/benchmarks/augmentation>`_ and
print their git commit, platform, and device so results are auditable:

- ``vs_torchvision.py`` — per-op kornia (eager / ``torch.compile``) vs torchvision v2, with a
  ``best/tv`` ratio and verdict per op.
- ``all_libraries.py`` — per-op across kornia, torchvision, albumentations, OpenCV, PIL, kornia-rs.
- ``pipeline.py`` — end-to-end multi-op pipeline throughput (the shape a training loop runs),
  including a compiled and an ``--half`` (fp16/AMP) path.

Reading them honestly: on a **GPU-batched, differentiable, compiled** pipeline kornia is the fastest
option (torchvision v2 is not differentiable; albumentations is CPU/``uint8``/single-image). On
**CPU single-image** throughput, SIMD/NumPy libraries such as albumentations are faster — that is
their regime, not kornia's. See ``benchmarks/augmentation/README.md`` for the per-regime breakdown.


.. currentmodule:: kornia.augmentation

.. toctree::

   augmentation.auto
   augmentation.base
   augmentation.container
   augmentation.module
