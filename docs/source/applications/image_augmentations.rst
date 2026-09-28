Image Augmentation
==================

.. meta::
   :description: Build differentiable, GPU-accelerated image augmentation pipelines for PyTorch with Kornia.

Image Augmentation is a data augmentation method that generates more training data
from the existing training samples. Image Augmentation is especially useful in domains
where training data is limited or expensive to obtain like in biomedical applications.

.. image:: https://github.com/kornia/data/raw/main/girona_aug.png
   :align: center

Learn more: `https://paperswithcode.com/task/image-augmentation <https://paperswithcode.com/task/image-augmentation>`_

Kornia Augmentations
--------------------

Kornia leverages differentiable and GPU image data augmentation through the module `kornia.augmentation <https://kornia.readthedocs.io/en/latest/augmentation.html>`_
by implementing the functionality to be easily used with `torch.nn.Sequential <https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html?highlight=sequential#torch.nn.Sequential>`_
and other advanced containers such as
:py:class:`~kornia.augmentation.container.AugmentationSequential`,
:py:class:`~kornia.augmentation.container.ImageSequential`,
:py:class:`~kornia.augmentation.container.PatchSequential` and
:py:class:`~kornia.augmentation.container.VideoSequential`.

Our augmentation package is highly inspired by the torchvision augmentation API, although our intention is not to replace it.
Kornia aligns more closely with OpenCV functionality, enforcing floating-point operators to guarantee better precision
without any float -> uint8 conversions, plus on-device acceleration.

Kornia augmentations are ``nn.Module`` instances, so the simplest pipeline is a plain ``nn.Sequential``.
See also the `Colab: Kornia Playground <https://colab.research.google.com/drive/1T20UNAG4SdlE2n2wstuhiewve5Q81VpS#revisionId=0B4unZG1uMc-WR3NVeTBDcmRwN0NxcGNNVlUwUldPMVprb1dJPQ>`_
for a hands-on comparison with torchvision.

.. code-block:: python

   import torch
   import torch.nn as nn
   import kornia.augmentation as K

   transform = nn.Sequential(
      K.RandomAffine(360),
      K.ColorJiggle(0.2, 0.3, 0.2, 0.3),
   )
   images = torch.rand(8, 3, 64, 64)  # (B, C, H, W) in [0, 1]
   out = transform(images)


Best Practices 1: Image Augmentation
++++++++++++++++++++++++++++++++++++

Kornia augmentations provide a simple on-device augmentation framework with a number of conveniences
(e.g. returning the transformation matrix, or inverting a geometric transform). On top of that, the
:py:class:`~kornia.augmentation.container.AugmentationSequential` container eases the pain of building augmentation
pipelines: it applies the same sampled transform to images, masks, bounding boxes and keypoints, and can invert it.

.. code-block:: python

   import torch
   import kornia.augmentation as K

   aug = K.AugmentationSequential(
      K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
      K.RandomAffine(360, [0.1, 0.1], [0.7, 1.2], [30., 50.], p=1.0),
      K.RandomPerspective(0.5, p=1.0),
      data_keys=["input", "bbox", "keypoints", "mask"],  # declares the order of the inputs below
      same_on_batch=False,
   )
   img_tensor = torch.rand(1, 3, 64, 64)
   bbox = torch.tensor([[[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]]])  # (B, 4, 2) corners
   keypoints = torch.tensor([[[16.0, 16.0], [48.0, 32.0]]])                          # (B, N, 2) in (x, y)
   mask = (torch.rand(1, 1, 64, 64) > 0.5).float()

   # forward the operation
   out_tensors = aug(img_tensor, bbox, keypoints, mask)
   # invert the geometric part of the operation
   out_tensors_inv = aug.inverse(*out_tensors)

.. image:: https://discuss.pytorch.org/uploads/default/optimized/3X/2/4/24bb0f4520f547d3a321440293c1d44921ecadf8_2_690x119.jpeg
   :alt: Original, augmented and inverted images

From left to right: the original image, the transformed image, and the inverted image.


Best Practices 2: Video Augmentation
++++++++++++++++++++++++++++++++++++

Video data is a special case of 3D volumetric data that contains both spatial and temporal information, which is
sometimes referred to as 2.5D rather than 3D. In most applications, augmenting video data requires the same
augmentation, with the same parameters, to be applied to every frame of a clip.
:py:class:`~kornia.augmentation.container.VideoSequential` does exactly that, with the same interface as `nn.Sequential`,
and supports the :math:`(B, C, T, H, W)` and :math:`(B, T, C, H, W)` data formats.

.. code-block:: python

   import torch
   import kornia.augmentation as K

   transform = K.VideoSequential(
      K.RandomAffine(360),
      K.RandomGrayscale(p=0.5),
      K.RandomHorizontalFlip(p=0.5),
      data_format="BCTHW",
      same_on_frame=True,
   )
   clip = torch.rand(2, 3, 8, 64, 64)  # 2 clips of 8 RGB frames
   out = transform(clip)

.. image:: https://user-images.githubusercontent.com/17788259/101993516-4625ca80-3c89-11eb-843e-0b87dca6e2b8.png
   :alt: Video augmentation applied consistently across frames


Customization
+++++++++++++

Compared to torchvision, Kornia augmentations expose two extra controls: the ``same_on_batch`` flag, which
applies the same random parameters to every element of the batch, and the sampled parameters themselves, which
every augmentation stores in ``_params`` after a call and accepts back through the ``params`` argument.
Geometric augmentations additionally expose the ``(B, 3, 3)`` matrix of the last call as ``transform_matrix``.

.. code-block:: python

   import torch
   import torch.nn as nn
   import kornia.augmentation as K

   class MyAugmentationPipeline(nn.Module):
      def __init__(self) -> None:
         super().__init__()
         self.aff = K.RandomAffine(360, same_on_batch=True, p=1.0)
         self.jit = K.ColorJiggle(0.2, 0.3, 0.2, 0.3, same_on_batch=True, p=1.0)

      def forward(self, input):
         out = self.aff(input)
         transform = self.aff.transform_matrix  # (B, 3, 3), reusable to undo the warp
         out = self.jit(out)
         return out, transform

   out, transform = MyAugmentationPipeline()(torch.rand(4, 3, 64, 64))

Example for semantic segmentation using low-level randomness control, re-applying the parameters
sampled for the image to its mask:

.. code-block:: python

   import torch
   import torch.nn as nn
   import kornia.augmentation as K

   class MyAugmentationPipeline(nn.Module):
      def __init__(self) -> None:
         super().__init__()
         self.aff = K.RandomAffine(360, p=1.0)
         self.jit = K.ColorJiggle(0.2, 0.3, 0.2, 0.3, p=1.0)

      def forward(self, input, mask):
         assert input.shape[-2:] == mask.shape[-2:], (
            f"Input and mask should have the same spatial size, got {input.shape} and {mask.shape}"
         )
         aff_params = self.aff.forward_parameters(input.shape)
         input = self.aff(input, aff_params)
         mask = self.aff(mask, aff_params)  # same geometric transform as the image

         input = self.jit(input)  # color jitter is applied to the image only
         return input, mask

   image, mask = MyAugmentationPipeline()(torch.rand(4, 3, 64, 64), torch.rand(4, 1, 64, 64))

For pipelines with masks, boxes or keypoints, prefer
:py:class:`~kornia.augmentation.container.AugmentationSequential`, which does this bookkeeping for you.
