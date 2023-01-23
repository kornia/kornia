Base Classes
============

.. currentmodule:: kornia.augmentation

This is the base class for creating a new transform on top the predefined routine of `kornia.augmentation`.
Specifically, an any given augmentation can be recognized as either rigid (e.g. affine transformations that
manipulate images with standard transformation matrice), or non-rigid (e.g. cut out a random area). At
image-level, Kornia supports rigid transformation like `GeometricAugmentationBase2D` that modifies the geometric
location of image pixels and `IntensityAugmentationBase2D` that preserves the pixel locations, as well as
generic `AugmentationBase2D` that allows higher freedom for customized augmentation design.


The Predefined Augmentation Routine
-----------------------------------

Kornia augmentation follows the simplest `sample-apply` routine for all the augmentations.

- `sample`: Kornia aims at flexible tensor-level augmentations that augment all images in a tensor with
   different augmentations and probabilities. The sampling operation firstly samples a suite of random
   parameters. Then all the sampled augmentation state (parameters) is stored
   inside `_param` of the augmentation, the users can hereby reproduce the same augmentation results.
- `apply`: With generated or passed parameters, the augmentation will be performed accordingly.
   Apart from performing image tensor operations, Kornia also supports inverse operations that
   to revert the transform operations. Meanwhile, other data modalities (`datakeys` in Kornia) like
   masks, keypoints, and bounding boxes. Such features are better supported with `AugmentationSequential`.
   Notably, the augmentation pipeline for rigid operations are implemented already without further efforts.
   For non-rigid operations, the user may implement customized inverse and data modality operations, e.g.
   `apply_mask_transform` for applying transformations on mask tensors.


Custom Augmentation Classes
---------------------------

For rigid transformations, `IntensityAugmentationBase2D` and `GeometricAugmentationBase2D` are sharing the exact same logic
apart from the transformation matrix computations. Namely, the intensity augmentation always results in
identity transformation matrices, without changing the geometric location for each pixel.

If it is a rigid geometric operation, `compute_transformation` and `apply_transform` need to be implemented, as well as
`compute_inverse_transformation` and `inverse_transform` to compute its inverse.

.. autoclass:: GeometricAugmentationBase2D

   .. automethod:: compute_transformation
   .. automethod:: apply_transform
   .. automethod:: compute_inverse_transformation
   .. automethod:: inverse_transform


For `IntensityAugmentationBase2D`, the user only needs to override `apply_transform`.

.. autoclass:: IntensityAugmentationBase2D

   .. automethod:: apply_transform

A minimal example to create your own rigid geometric augmentations with the following snippet:


.. code-block:: python

   import torch
   import kornia as K

   from kornia.augmentation import GeometricAugmentationBase2D
   from kornia.augmentation import random_generator as rg


   class MyRandomTransform(GeometricAugmentationBase2D):

      def __init__(
         self,
         factor=(0., 1.),
         same_on_batch: bool = False,
         p: float = 1.0,
         keepdim: bool = False,
      ) -> None:
         super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
         self._param_generator = rg.PlainUniformGenerator((factor, "factor", None, None))

      def compute_transformation(self, input, params):
         # a simple identity transformation example
         factor = params["factor"].to(input) * 0. + 1
         return K.eyelike(input, 3) * factor

      def apply_transform(
         self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
      ) -> Tensor:
         factor = params["factor"].to(input)
         return input * factor


For non-rigid augmentations, the user may implement the `apply_transform*` and `apply_non_transform*` APIs
to meet the needs. Specifically, `apply_transform*` applies to the elements of a tensor that need to be transformed,
while `apply_non_transform*` applies to the elements of a tensor that are skipped from augmentation. For example,
a crop operation may change the tensor size partially, while we need to resize the rest to maintain the whole tensor
as an integrated one with the same size.


.. autoclass:: AugmentationBase2D

   .. automethod:: apply_transform
   .. automethod:: apply_non_transform
   .. automethod:: apply_transform_mask
   .. automethod:: apply_non_transform_mask
   .. automethod:: apply_transform_box
   .. automethod:: apply_non_transform_box
   .. automethod:: apply_transform_keypoint
   .. automethod:: apply_non_transform_keypoint
   .. automethod:: apply_transform_class
   .. automethod:: apply_non_transform_class


The similar logic applies to 3D augmentations as well.


Some Further Notes
------------------

Probabilities
^^^^^^^^^^^^^
Kornia supports two types of randomness for element-level randomness `p` and batch-level randomness `p_batch`,
as in `_BasicAugmentationBase`. Under the hood, operations like `crop`, `resize` are implemented with a fixed
element-level randomness of `p=1` that only maintains batch-level randomness.


Random Generators
^^^^^^^^^^^^^^^^^
For automatically generating the corresponding ``__repr__`` with full customized parameters, you may need to
implement ``_param_generator`` by inheriting ``RandomGeneratorBase`` for generating random parameters and
put all static parameters inside ``self.flags``. You may take the advantage of ``PlainUniformGenerator`` to
generate simple uniform parameters with less boilerplate code.


Random Reproducibility
^^^^^^^^^^^^^^^^^^^^^^
Plain augmentation base class without the functionality of transformation matrix calculations.
By default, the random computations will be happened on CPU with ``torch.get_default_dtype()``.
To change this behaviour, please use ``set_rng_device_and_dtype``.
