kornia.augmentation
===================

.. currentmodule:: kornia.augmentation

The classes in this section perform various data augmentation operations.

Kornia provides Torchvision-like augmentation APIs while may not reproduce Torchvision, because Kornia is a library aligns to OpenCV functionalities, not PIL. Besides, pure floating computation is used in Kornia which gaurentees a better precision without any float -> uint8 conversions. To be specified, the different functions are:

- AdjustContrast
- AdjustBrightness

For detailed comparision, please checkout the `Colab: Kornia Playground <https://colab.research.google.com/drive/1T20UNAG4SdlE2n2wstuhiewve5Q81VpS#revisionId=0B4unZG1uMc-WR3NVeTBDcmRwN0NxcGNNVlUwUldPMVprb1dJPQ>`_.

Containers
----------

This is the base class for creating a new transform. The user only needs to overrive: `generate_parameters`, `apply_transform` and optionally, `compute_transformation`.


.. autoclass:: AugmentationBase2D

   .. automethod:: generate_parameters
   .. automethod:: compute_transformation
   .. automethod:: apply_transform

.. autoclass:: AugmentationBase3D

   .. automethod:: generate_parameters
   .. automethod:: compute_transformation
   .. automethod:: apply_transform

Create your own transformation:

.. code-block:: python

   import torch
   import kornia as K

   from kornia.augmentation import AugmentationBase2D

   class MyRandomTransform(AugmentationBase2D):
      def __init__(self, return_transform: bool = False) -> None:
         super(MyRandomTransform, self).__init__(return_transform)

      def generate_parameters(self, input_shape: torch.Size):
         # generate the random parameters for your use case.
         angles_rad torch.Tensor = torch.rand(input_shape[0]) * K.pi
	 angles_deg = kornia.rad2deg(angles_rad) 
	 return dict(angles=angles_deg)
      
      def compute_transformation(self, input, params):

    	 B, _, H, W = input.shape

	 # compute transformation
	 angles: torch.Tensor = params['angles'].type_as(input)
	 center = torch.tensor([[W / 2, H / 2]] * B).type_as(input)
	 transform = K.get_rotation_matrix2d(
            center, angles, torch.ones_like(angles))
	 return transform

      def apply_transform(self, input, params):

    	 _, _, H, W = input.shape
	 # compute transformation
	 transform = self.compute_transformation(input, params)

         # apply transformation and return
	 output = K.warp_affine(input, transform, (H, W))
         return (output, transform)

Module API
----------

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

	 aff_params = self.aff.generate_parameters(input.shape)
	 input = self.aff(input, aff_params)
	 mask = self.aff(mask, aff_params)
		
	 jit_params = self.jit.generate_parameters(input.shape)
	 input = self.jit(input, jit_params)
	 mask = self.jit(mask, jit_params)
	 return input, mask

Transforms2D
------------

Set of operators to perform data augmentation on 2D image tensors.

.. autoclass:: CenterCrop
.. autoclass:: ColorJitter
.. autoclass:: Denormalize
.. autoclass:: Normalize
.. autoclass:: RandomAffine
.. autoclass:: RandomCrop
.. autoclass:: RandomErasing
.. autoclass:: RandomGrayscale
.. autoclass:: RandomHorizontalFlip
.. autoclass:: RandomVerticalFlip
.. autoclass:: RandomMotionBlur
.. autoclass:: RandomPerspective
.. autoclass:: RandomResizedCrop
.. autoclass:: RandomRotation
.. autoclass:: RandomSolarize
.. autoclass:: RandomPosterize
.. autoclass:: RandomSharpness
.. autoclass:: RandomEqualize
.. autoclass:: RandomMixUp
.. autoclass:: RandomCutMix

Transforms3D
------------

Set of operators to perform data augmentation on 3D image tensors and volumetric data.

.. autoclass:: RandomDepthicalFlip3D
.. autoclass:: RandomHorizontalFlip3D
.. autoclass:: RandomVerticalFlip3D
.. autoclass:: RandomRotation3D
.. autoclass:: RandomAffine3D
.. autoclass:: RandomCrop3D
.. autoclass:: CenterCrop3D
.. autoclass:: RandomMotionBlur3D
.. autoclass:: RandomEqualize3D

Functional
----------

.. automodule:: kornia.augmentation.functional.functional
    :members:
.. automodule:: kornia.augmentation.functional.functional3d
    :members:
