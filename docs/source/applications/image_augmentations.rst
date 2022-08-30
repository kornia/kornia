Image Augmentation
==================

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

Our augmentations package is highly inspired by torchvision augmentation APIs while our intention is to not replace it.
Kornia is a library that aligns better to OpenCV functionalities enforcing floating operators to guarantees a better precision
without any float -> uint8 conversions plus on device acceleration.

However, we provide the following guide to migrate kornia <-> torchvision. Please, checkout the `Colab: Kornia Playground <https://colab.research.google.com/drive/1T20UNAG4SdlE2n2wstuhiewve5Q81VpS#revisionId=0B4unZG1uMc-WR3NVeTBDcmRwN0NxcGNNVlUwUldPMVprb1dJPQ>`_.

.. code-block:: python

   import kornia.augmentation as K
   import torch.nn as nn

   transform = nn.Sequential(
      K.RandomAffine(360),
      K.ColorJiggle(0.2, 0.3, 0.2, 0.3)
   )


Best Practices 1: Image Augmentation
++++++++++++++++++++++++++++++++++++

Kornia augmentations provides simple on-device augmentation framework with the support of various syntax sugars
(e.g. return transformation matrix, inverse geometric transform). Therefore, we provide advanced augmentation
container :py:class:`~kornia.augmentation.container.AugmentationSequential` to ease the pain of building augmenation pipelines. This API would also provide predefined routines
for automating the processing of masks, bounding boxes, and keypoints.

.. code-block:: python

   import kornia.augmentation as K

   aug = K.AugmentationSequential(
      K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
      K.RandomAffine(360, [0.1, 0.1], [0.7, 1.2], [30., 50.], p=1.0),
      K.RandomPerspective(0.5, p=1.0),
      data_keys=["input", "bbox", "keypoints", "mask"],  # Just to define the future input here.
      return_transform=False,
      same_on_batch=False,
   )
   # forward the operation
   out_tensors = aug(img_tensor, bbox, keypoints, mask)
   # Inverse the operation
   out_tensor_inv = aug.inverse(*out_tensor)

.. image:: https://discuss.pytorch.org/uploads/default/optimized/3X/2/4/24bb0f4520f547d3a321440293c1d44921ecadf8_2_690x119.jpeg

From left to right: the original image, the transformed image, and the inversed image.


Best Practices 2: Video Augmentation
++++++++++++++++++++++++++++++++++++

Video data is a special case of 3D volumetric data that contains both spatial and temporal information, which can be referred as 2.5D than 3D.
In most applications, augmenting video data requires a static temporal dimension to have the same augmentations are performed for each frame.
Thus, :py:class:`~kornia.augmentation.container.VideoSequential` can be used to do such trick as same as `nn.Sequential`.
Currently, :py:class:`~kornia.augmentation.container.VideoSequential` supports data format like :math:`(B, C, T, H, W)` and :math:`(B, T, C, H, W)`.

.. code-block:: python

   import kornia.augmentation as K

   transform = K.VideoSequential(
      K.RandomAffine(360),
      K.RandomGrayscale(p=0.5),
      K.RandomAffine(p=0.5)
      data_format="BCTHW",
      same_on_frame=True
   )

.. image:: https://user-images.githubusercontent.com/17788259/101993516-4625ca80-3c89-11eb-843e-0b87dca6e2b8.png


Customization
+++++++++++++

Kornia augmentation implementations have two additional parameters compare to TorchVision,
``return_transform`` and ``same_on_batch``. The former provides the ability of undoing one geometry
transformation while the latter can be used to control the randomness for a batched transformation.
To enable those behaviour, you may simply set the flags to True.

.. code-block:: python

   import kornia.augmentation as K

   class MyAugmentationPipeline(nn.Module):
      def __init__(self) -> None:
         super(MyAugmentationPipeline, self).__init__()
	 self.aff = K.RandomAffine(
            360, return_transform=True, same_on_batch=True
         )
	 self.jit = K.ColorJiggle(0.2, 0.3, 0.2, 0.3, same_on_batch=True)

      def forward(self, input):
	 input, transform = self.aff(input)
	 input, transform = self.jit((input, transform))
	 return input, transform

Example for semantic segmentation using low-level randomness control:

.. code-block:: python

   import kornia.augmentation as K

   class MyAugmentationPipeline(nn.Module):
      def __init__(self) -> None:
	 super(MyAugmentationPipeline, self).__init__()
	 self.aff = K.RandomAffine(360)
	 self.jit = K.ColorJiggle(0.2, 0.3, 0.2, 0.3)

      def forward(self, input, mask):
         assert input.shape == mask.shape,
	    f"Input shape should be consistent with mask shape, "
            f"while got {input.shape}, {mask.shape}"

	 aff_params = self.aff.forward_parameters(input.shape)
	 input = self.aff(input, aff_params)
	 mask = self.aff(mask, aff_params)

	 jit_params = self.jit.forward_parameters(input.shape)
	 input = self.jit(input, jit_params)
	 mask = self.jit(mask, jit_params)
	 return input, mask
