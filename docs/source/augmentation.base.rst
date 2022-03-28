Base Classes
============

.. currentmodule:: kornia.augmentation

This is the base class for creating a new transform using `kornia.augmentation`.
The user only needs to override: `generate_parameters`, `apply_transform` and optionally, `compute_transformation`.

Create your own transformations with the following snippet:

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

.. autoclass:: AugmentationBase2D

   .. automethod:: generate_parameters
   .. automethod:: compute_transformation
   .. automethod:: apply_transform

.. autoclass:: AugmentationBase3D

   .. automethod:: generate_parameters
   .. automethod:: compute_transformation
   .. automethod:: apply_transform
