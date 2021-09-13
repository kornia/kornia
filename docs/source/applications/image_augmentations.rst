Image Augmentation
==================

Image Augmentation is a data augmentation method that generates more training data
from the existing training samples. Image Augmentation is especially useful in domains
where training data is limited or expensive to obtain like in biomedical applications.

.. image:: https://github.com/kornia/data/raw/main/girona_aug.png
   :align: center

Learn more: `https://paperswithcode.com/task/image-registration <https://paperswithcode.com/task/image-augmentation>`_

Kornia Augmentations
--------------------

Kornia leverages differentiable and GPU image data augmentation through the module `kornia.augmentation <https://kornia.readthedocs.io/en/latest/augmentation.html>`_
by implementing the functionality to be easily used with `torch.nn.Sequential <https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html?highlight=sequential#torch.nn.Sequential>`_
and other advanced containers such as
:py:class:`~kornia.augmentation.container.AugmentationSequential`,
:py:class:`~kornia.augmentation.container.ImageSequential`,
:py:class:`~kornia.augmentation.container.PatchSequential` and
:py:class:`~kornia.augmentation.container.VideoSequential`.

Our augmentations package is highly inspired by Torchvision-like augmentation APIs while may not reproduce Torchvision,
because Kornia is a library aligns to OpenCV functionalities, not PIL. Besides, pure floating computation is used in Kornia
which guarantees a better precision without any float -> uint8 conversions.

For detailed comparison, please checkout the `Colab: Kornia Playground <https://colab.research.google.com/drive/1T20UNAG4SdlE2n2wstuhiewve5Q81VpS#revisionId=0B4unZG1uMc-WR3NVeTBDcmRwN0NxcGNNVlUwUldPMVprb1dJPQ>`_.

.. code-block:: python

   import kornia.augmentation as K
   import torch.nn as nn

   transform = nn.Sequential(
      K.RandomAffine(360),
      K.ColorJitter(0.2, 0.3, 0.2, 0.3)
   )

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
	 self.jit = K.ColorJitter(0.2, 0.3, 0.2, 0.3, same_on_batch=True)

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
	 self.jit = K.ColorJitter(0.2, 0.3, 0.2, 0.3)

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
