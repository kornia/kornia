kornia.augmentation
===================

.. meta::
   :description: The Augmentation module in Kornia provides high-level data augmentation functionalities for computer vision tasks, including random rotations, affine transformations, color intensities, image noise distortion, and more. It supports batch processing, device compatibility, and backpropagation. Additionally, users can retrieve transformation details for more flexibility in complex pipelines.

This module applies random augmentations to batches of float tensors on the CPU or the GPU: spatial transforms,
color and intensity changes, blur and noise, image mixing, 3D volume transforms, and the AutoAugment, RandAugment
and TrivialAugment policies. Most operators are differentiable with respect to their input.

Every call records what it sampled. In eager mode, ``aug._params`` holds the last draw, and
``aug(image, params=aug._params)`` replays it; matrix-based geometric transforms also expose the draw as
``aug.transform_matrix``. :class:`~kornia.augmentation.container.AugmentationSequential` uses this record to apply
one draw to an image and its masks, boxes and keypoints, and its ``inverse()`` undoes the matrix-based geometric
steps. Inversion cannot restore pixels lost to cropping or resampling, and it leaves color changes in place.
For images alone, chain transforms with :class:`~kornia.augmentation.container.ImageSequential`; for video, apply
them to every frame with :class:`~kornia.augmentation.container.VideoSequential`.

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
   **Input format.** Pass float tensors of shape ``(B, C, H, W)`` for images and ``(B, C, D, H, W)`` for
   volumes. Color transforms expect values in ``[0, 1]``, so convert 8-bit images with ``img.float() / 255``;
   a cast alone does not rescale. A ``uint8`` tensor raises ``TypeError``, but a float tensor in ``[0, 255]``
   raises nothing: most color transforms, such as :class:`~kornia.augmentation.ColorJiggle`, clamp it to
   ``[0, 1]`` and the image comes out nearly white. For the same reason, apply color transforms before
   :class:`~kornia.augmentation.Normalize`.

Benchmark
---------

Kornia is built for batches of float tensors on the GPU, with gradients through most operators. The libraries it is
compared with target other regimes: albumentations, OpenCV, PIL and kornia-rs transform one ``uint8`` image at a
time on the CPU, and torchvision v2 runs batched on the GPU but is not differentiable. Compare libraries at the
same batch size and on the same device.

The scripts under
`benchmarks/augmentation/ <https://github.com/kornia/kornia/tree/main/benchmarks/augmentation>`_ print their git
commit, platform and device with every result:

- ``flagship.py`` — each library's random-transform classes, parameter sampling included, against torchvision v2,
  albumentations, OpenCV and PIL, with JSON export.
- ``vs_torchvision.py`` — each kornia op, eager and compiled, against torchvision v2.
- ``pipeline.py`` — a multi-op training pipeline end to end, with compiled and ``--half`` (fp16/AMP) paths.

In the RTX 4090 snapshot in ``benchmarks/augmentation/README.md`` (256 × 256 float32 images, batch 32), kornia on
the GPU is 2–19× faster than the CPU ``uint8`` libraries on all eight operations, and faster than torchvision v2 on
three of them: ``ColorJiggle``, Gaussian blur and perspective. On the CPU at batch 1, albumentations or OpenCV is
faster on all eight.

Deployment: torch.export, torch.compile, ONNX
---------------------------------------------

Augmentations are ``nn.Module`` s, so the PyTorch capture tools apply to them, within these limits:

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

- **torch.compile.** In the :ref:`augmentation export survey <export-support-kornia-augmentation>`, most single
  operators compile into one graph; ``AugmentationSequential``, the automatic policies and the mix transforms
  compile with graph breaks. Compiling is not always faster: in the benchmark snapshot it slows CPU
  ``RandomAffine`` to 0.45× of eager.

- **ONNX.** Export with ``torch.onnx.export(..., dynamo=True)`` and chain the file with
  :class:`~kornia.onnx.sequential.ONNXSequential`, which takes ONNX graphs, not augmentation modules. Without
  ``dynamic_shapes`` the graph is fixed to the input shape used for export; some augmentations, such as ``Resize``,
  stay fixed even with it, or fail to export with it on torch 2.5.1. The :ref:`export survey <export-support>`
  records which augmentations export. Pre-built ONNX models are published under the ``kornia/ONNX_models``
  Hugging Face repo.

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
     - :class:`~kornia.augmentation.container.AugmentationSequential` applies one draw to images, masks, boxes
       and keypoints and can invert it; ``ImageSequential`` chains image-only transforms, ``VideoSequential``
       applies them per frame, ``PatchSequential`` per grid patch, and the dispatchers run several pipelines.
   * - :doc:`Automatic augmentation <augmentation.auto>`
     - The policies :class:`~kornia.augmentation.auto.AutoAugment`,
       :class:`~kornia.augmentation.auto.RandAugment` and :class:`~kornia.augmentation.auto.TrivialAugment`,
       and the operations they sample.
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
