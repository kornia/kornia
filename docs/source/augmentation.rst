kornia.augmentation
===================

.. meta::
   :description: The Augmentation module in Kornia provides high-level data augmentation functionalities for computer vision tasks, including random rotations, affine transformations, color intensities, image noise distortion, and more. It supports batch processing, device compatibility, and backpropagation. Additionally, users can retrieve transformation details for more flexibility in complex pipelines.

This module implements data augmentation at a high level of abstraction. Like the rest of the library,
it performs augmentation routines in batch mode, on any supported device, and every operation is
differentiable, so it can take part in backpropagation. Among the available functionalities are random rotations;
affine and perspective transformations; several random color and intensity transformations; image noise;
motion blur; and many differentiable data augmentation policies such as AutoAugment and RandAugment.

In addition, the module includes a feature that is not found in other augmentation frameworks: after each
call, the user can retrieve the applied transformation, or the chain of transformations, e.g. the sampled
random rotation matrix. That matrix can later be used to undo the image transformation, or to apply the same
transformation to additional data such as segmentation masks, bounding boxes or landmark keypoints. This gives
the user the flexibility to build complex data augmentation pipelines.

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
   matrix = aug.transform_matrix  # (B, 3, 3) matrix of the last geometric transform

.. note::
   **Input format.** Kornia augmentations expect **float tensors with values in** ``[0, 1]`` and shape
   ``(B, C, H, W)`` — the same format produced by ``torchvision.transforms.v2.ToDtype(torch.float32, scale=True)``
   (or the legacy ``ToTensor``). A ``uint8`` tensor raises a clear error, but a **float tensor already in the**
   ``[0, 255]`` **range does not raise** — it is silently treated as ``[0, 1]`` and the output clips at ``1.0``,
   which shows up later as degraded model accuracy rather than an exception. Scale first, e.g. ``img.float() / 255``.

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
- ``flagship.py`` — the flagship suite: augmentations through each library's random-transform
  class API (parameter sampling included) vs torchvision v2, albumentations, OpenCV, and PIL,
  with machine-readable JSON export.
- ``pipeline.py`` — end-to-end multi-op pipeline throughput (the shape a training loop runs),
  including a compiled and an ``--half`` (fp16/AMP) path.

Reading them honestly: on a **GPU-batched, differentiable, compiled** pipeline kornia is the fastest
option (torchvision v2 is not differentiable; albumentations is CPU/``uint8``/single-image). On
**CPU single-image** throughput, SIMD/NumPy libraries such as albumentations are faster — that is
their regime, not kornia's. See ``benchmarks/augmentation/README.md`` for the per-regime breakdown.

Deployment: torch.export, torch.compile, ONNX
---------------------------------------------

Kornia augmentations are plain ``nn.Module`` s and are exportable for deployment:

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

- **torch.compile.** Most augmentations run fullgraph (0 graph breaks); a compiled
  ``AugmentationSequential`` fuses the pointwise chain end to end. See the per-op ``test_dynamo``
  tests for the compile-clean set.

- **ONNX.** Export a pipeline with :class:`kornia.onnx.ONNXSequential`; pre-built ONNX models are
  published under the ``kornia/ONNX_models`` Hugging Face repo.

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
     - :class:`AugmentationSequential` applies one sampled transform to images, masks, boxes and
       keypoints and can invert it; plus ``ImageSequential``, ``PatchSequential``, ``VideoSequential``
       and the dispatchers.
   * - :doc:`Automatic augmentation <augmentation.auto>`
     - The learned policies: :class:`~kornia.augmentation.auto.AutoAugment`,
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
