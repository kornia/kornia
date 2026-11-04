kornia.augmentation
===================

.. meta::
   :description: The Augmentation module in Kornia provides high-level data augmentation functionalities for computer vision tasks, including random rotations, affine transformations, color intensities, image noise distortion, and more. It supports batch processing, device compatibility, and backpropagation. Additionally, users can retrieve transformation details for more flexibility in complex pipelines.

Build a transform with ``aug = K.RandomAffine(degrees=30.0)``, then call ``out = aug(image)`` on a floating-point
image batch of shape ``(B, C, H, W)``. The module includes spatial transforms, color adjustments, blur, noise,
mixing and automatic policies. Each operator documents its accepted channels, value range and probability rule;
gradient support depends on the operation and the quantity being differentiated.

Use :class:`~kornia.augmentation.container.AugmentationSequential` when masks, boxes or keypoints must follow an
image's spatial transform. It applies the recorded parameters through each child's supported data handlers.
For images alone, use :class:`~kornia.augmentation.container.ImageSequential`; for frame-wise video transforms,
use :class:`~kornia.augmentation.container.VideoSequential`.

In eager mode, ``aug._params`` stores the sampled parameters for replay with ``params=``. Matrix-based geometric
transforms also expose ``transform_matrix``. A matrix describes the geometric part of a pipeline, not color changes
or non-rigid deformations. ``inverse()`` can undo supported 2D geometric steps, but leaves intensity transforms
applied and cannot restore content lost through cropping or interpolation. See the container's contract for
annotation and inverse limitations.

.. code-block:: python

   import torch
   import kornia.augmentation as K

   aug = K.AugmentationSequential(
       K.RandomAffine(degrees=30.0, p=1.0),
       K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
       data_keys=["input", "mask", "keypoints"],
   )
   image = torch.rand(2, 3, 64, 64)
   mask = (torch.rand(2, 1, 64, 64) > 0.5).float()
   keypoints = torch.tensor([[[16.0, 16.0], [48.0, 32.0]]]).repeat(2, 1, 1)

   img_out, mask_out, kpts_out = aug(image, mask, keypoints)  # same random parameters for all three
   img_back, mask_back, kpts_back = aug.inverse(img_out, mask_out, kpts_out)
   matrix = aug.transform_matrix  # (B, 3, 3) composed matrix for the geometric steps

.. note::
   **Input format.** Use floating-point tensors with shape ``(B, C, H, W)`` for image batches and
   ``(B, C, D, H, W)`` for volume batches. Color transforms generally expect values in ``[0, 1]``: convert
   8-bit images with ``img.float() / 255`` before applying them. Casting a ``[0, 255]`` image to float alone
   does not scale it. A ``uint8`` input raises ``TypeError`` in image augmentations. A float input in
   ``[0, 255]`` is accepted without an error and then clamped by color transforms such as
   :class:`~kornia.augmentation.ColorJiggle`, discarding intensity detail.
   Geometric transforms and :class:`~kornia.augmentation.Normalize` have different range rules; check the
   operator's contract before applying color transforms to normalized data.

Benchmark
---------

For CPU ``uint8`` single-image loaders, the measured comparisons favor native implementations such as
albumentations, OpenCV and kornia-rs. Batched floating-point GPU pipelines are Kornia's stronger regime, with
gradient support for differentiable operators; the results depend on the operator, device and compilation
settings. Benchmark the batch size and pipeline you will use, including parameter sampling.

Reproducible, honestly-framed benchmarks live under
`benchmarks/augmentation/ <https://github.com/kornia/kornia/tree/main/benchmarks/augmentation>`_ and
print their git commit, platform, and device so results are auditable:

- ``vs_torchvision.py`` — per-op kornia (eager / ``torch.compile``) vs torchvision v2, with a
  ``best/tv`` ratio and verdict per op.
- ``flagship.py`` — the flagship suite: augmentations through each library's random-transform
  class API (parameter sampling included) vs torchvision v2, albumentations, OpenCV, and PIL,
  with machine-readable JSON export.
- ``pipeline.py`` — end-to-end multi-op pipeline throughput (the shape a training loop runs),
  including a compiled and an ``--half`` (fp16/AMP) path.

See ``benchmarks/augmentation/README.md`` for the measured configurations and results. A result for a compiled
GPU pipeline does not predict eager CPU throughput, and a per-operation result does not include the costs of
an entire training pipeline.

Deployment: torch.export, torch.compile, ONNX
---------------------------------------------

Augmentations are PyTorch modules. Capture and export support depend on the operators and the input forms:

- **torch.export.** Deterministic transforms — ``Normalize``, ``Denormalize``, ``Resize``,
  ``CenterCrop``, ``PadTo`` — and an ``AugmentationSequential`` pipeline of them capture cleanly
  and match eager, so a model can be shipped together with its preprocessing as one program:

  .. code-block:: python

     import torch, kornia.augmentation as K

     tf = K.AugmentationSequential(
         K.Normalize(mean=torch.zeros(3), std=torch.ones(3)),
         K.Resize((224, 224)),
         data_keys=["input"],
     )
     exported = torch.export.export(tf, (torch.rand(1, 3, 256, 256),))

  The per-call state kept on the module for eager retrieval (``._params``, ``.transform_matrix``)
  is skipped during an export capture; the captured output is unchanged. Random augmentations and
  bounding-box/keypoint propagation through an exported graph are not covered.

- **torch.compile.** The :ref:`augmentation export survey <export-support-kornia-augmentation>` lists
  per-operator compile results for its measured configurations. Graph capture depends on the operators, flags
  and shapes; a successful capture does not by itself establish a speedup.

- **ONNX.** :class:`~kornia.onnx.sequential.ONNXSequential` chains ONNX graphs; it does not take an augmentation
  module. The :ref:`export survey <export-support>` records which individual augmentations exported. Without
  ``dynamic_shapes``, a pipeline exported with ``torch.onnx.export(..., dynamo=True)`` gets a graph of the input shape
  it was exported with, and some augmentations (``Resize``, for one) keep that shape fixed even with it, or fail to
  export with it on torch 2.5.1. Some augmentations fail to export. The exported file can be chained with
  ``ONNXSequential``. Pre-built ONNX models are published under the ``kornia/ONNX_models`` Hugging Face repo.

.. currentmodule:: kornia.augmentation

Where to find things
--------------------

.. list-table::
   :widths: 30 70

   * - :doc:`Image augmentations <augmentation.module>`
     - The operators, by category: :doc:`intensity <augmentation.intensity>` (color, blur, noise,
       illumination, normalization), :doc:`geometric <augmentation.geometric>` (crops, flips, affine,
       perspective, resize), :doc:`mix <augmentation.mix>` (CutMix, MixUp, Mosaic) and
       :doc:`3D <augmentation.transforms3d>`.
   * - :doc:`Containers <augmentation.container>`
     - Coordinate images and supported annotations with :class:`~kornia.augmentation.container.AugmentationSequential`; use
       ``ImageSequential`` for images, ``PatchSequential`` for image grids, ``VideoSequential`` for frames,
       or the dispatchers for multiple pipelines.
   * - :doc:`Automatic augmentation <augmentation.auto>`
     - Policy selection and operation sampling: :class:`~kornia.augmentation.auto.AutoAugment`,
       :class:`~kornia.augmentation.auto.RandAugment` and
       :class:`~kornia.augmentation.auto.TrivialAugment`.
   * - :doc:`Base classes <augmentation.base>`
     - Subclass ``IntensityAugmentationBase2D`` or ``GeometricAugmentationBase2D`` to write your own
       augmentation, with worked examples.

.. toctree::
   :hidden:

   intensity <augmentation.intensity>
   geometric <augmentation.geometric>
   mix <augmentation.mix>
   3d transforms <augmentation.transforms3d>
   containers <augmentation.container>
   automatic policies <augmentation.auto>
   base classes <augmentation.base>
