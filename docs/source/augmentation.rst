kornia.augmentation
===================

.. currentmodule:: kornia.augmentation

The classes in this section perform various data augmentation operations.

Kornia provides Torchvision-like augmentation APIs while may not reproduce Torchvision, because Kornia is a library aligns to OpenCV functionalities, not PIL. Besides, pure floating computation is used in Kornia which guarantees a better precision without any float -> uint8 conversions. To be specified, the different functions are:

- AdjustContrast
- AdjustBrightness

For detailed comparison, please checkout the `Colab: Kornia Playground <https://colab.research.google.com/drive/1T20UNAG4SdlE2n2wstuhiewve5Q81VpS#revisionId=0B4unZG1uMc-WR3NVeTBDcmRwN0NxcGNNVlUwUldPMVprb1dJPQ>`_.


Kornia augmentation implementations can be easily used in a TorchVision style using `nn.Sequential`.

.. code-block:: python

   import kornia.augmentation as K
   import torch.nn as nn

   transform = nn.Sequential(
      K.RandomAffine(360),
      K.ColorJitter(0.2, 0.3, 0.2, 0.3)
   )

Kornia augmentation implementations have two additional parameters compare to TorchVision, `return_transform` and `same_on_batch`. The former provides the ability of undoing one geometry transformation while the latter can be used to control the randomness for a batched transformation. To enable those behaviour, you may simply set the flags to True.

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

.. toctree::
   :maxdepth: 2

   augmentation.base
   augmentation.module
   augmentation.container
