kornia.augmentation
===================

.. currentmodule:: kornia.augmentation

The classes in this section perform various data augmentation operations.

Kornia provides Torchvision-like augmentation APIs while may not reproduce Torchvision, because Kornia is a library aligns to OpenCV functionalities, not PIL. Besides, pure floating computation is used in Kornia which gaurentees a better precision without any float -> uint8 conversions. To be specified, the different functions are:

- AdjustContrast
- AdjustBrightness
- RandomRectangleErasing

For detailed comparision, please checkout the For detailed comparision, please checkout the `Colab: Kornia vs. Torchvision <https://colab.research.google.com/drive/1T20UNAG4SdlE2n2wstuhiewve5Q81VpS#revisionId=0B4unZG1uMc-WdzZqaStjVzZ1U0hHOHphQkgvcGFCZ1RlUzJvPQ/>`_.

Containers
----------

This is the base class for creating a new transform. The user only needs to overrive: `generate_parameters`, `apply_transformation` and optionally, `compute_transformation`.


.. autoclass:: AugmentationBase
   :members:

Create your own transformation:

.. code-block:: python

	class MyRandomTransform(nn.Module):
		def __init__(self, return_transform: bool = False) -> None:
			super(MyRandomTransform, self).__init__(return_transform)

		def generate_parameters(self, input_shape: torch.Size):
			# generate the random parameters as you wish depending on your use case.
			angles_rad torch.Tensor = torch.rand(batch_shape) * K.pi
			angles_deg = kornia.rad2deg(angles_rad) * self.angle
			return dict(angles=angles_deg)

		def compute_transformation(self, input, params):
			# compute transformation
			angles: torch.Tensor = params['angles'].type_as(input)
			center = torch.tensor([[W / 2, H / 2]]).type_as(input)
			transform = K.get_rotation_matrix2d(center, angles, torch.ones_like(angles))
			return transform

		def apply_transformation(self, input, params):
			# compute transformation
		  	transform = self.compute_transform(input, params)

		  	# apply transformation and return
		  	output = K.warp_affine(input, transform, (H, W))

		  	return (output, transform)

Module
------

.. autoclass:: RandomHorizontalFlip
.. autoclass:: RandomVerticalFlip
.. autoclass:: RandomErasing
.. autoclass:: RandomGrayscale
.. autoclass:: RandomAffine
.. autoclass:: RandomPerspective
.. autoclass:: RandomRotation
.. autoclass:: ColorJitter
.. autoclass:: CenterCrop
.. autoclass:: RandomCrop
.. autoclass:: RandomResizedCrop
.. autoclass:: Normalize
.. autoclass:: Denormalize

Functional
----------

.. automodule:: kornia.augmentation.functional
    :members:
