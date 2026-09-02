Base Classes
============

.. meta::
   :description: The Base Classes module in Kornia provides foundational classes for creating new image transformations. It supports rigid (e.g., affine) and non-rigid (e.g., cut-out) augmentations, with predefined routines for sampling, applying, and reversing transformations.

.. currentmodule:: kornia.augmentation

These are the base classes for creating a new transform on top of the predefined routine of `kornia.augmentation`.
Any given augmentation can be classified as either rigid (e.g. affine transformations that
manipulate images with a standard transformation matrix) or non-rigid (e.g. cutting out a random area). At the
image level, Kornia provides `GeometricAugmentationBase2D` for rigid transformations that modify the geometric
location of image pixels, `IntensityAugmentationBase2D` for transformations that preserve pixel locations, and the
generic `AugmentationBase2D`, which allows more freedom for customized augmentation design.

The Base-Class Hierarchy
------------------------

The bases form a chain, each layer adding one concern and shared by several subclasses — so pick the
*shallowest* base that already does what you need:

.. code-block:: text

   nn.Module
   └─ _BasicAugmentationBase          parameter sampling + the forward skeleton
      ├─ _AugmentationBase            dispatch to image / mask / box / keypoint / class data keys
      │  ├─ AugmentationBase2D        2D tensor validation  (subclass for a fully custom 2D op)
      │  │  └─ RigidAffineAugmentationBase2D     transform-matrix machinery
      │  │     ├─ IntensityAugmentationBase2D    pointwise ops — override apply_transform
      │  │     └─ GeometricAugmentationBase2D    warp ops — also override compute_transformation
      │  └─ AugmentationBase3D … (the 3D mirror of the 2D chain)
      └─ MixAugmentationBaseV2        mix ops (MixUp / CutMix) — bypass the per-key dispatch

Each level is a distinct, reused axis (sampling, data-key dispatch, 2D vs 3D, rigid-matrix vs free-form,
intensity vs geometric); the four ``*Base2D`` classes are public API that external code subclasses.
For a custom augmentation, subclass ``IntensityAugmentationBase2D`` or ``GeometricAugmentationBase2D``.

The Predefined Augmentation Routine
-----------------------------------

Kornia augmentation follows a simple `sample-apply` routine for all augmentations.

- `sample`: Kornia aims at flexible tensor-level augmentations that augment every image in a batch with
  different parameters and probabilities. The sampling step first draws a set of random
  parameters. The sampled augmentation state is then stored in the ``_params`` attribute of the augmentation,
  so users can reproduce the same augmentation results.
- `apply`: with the generated (or user-provided) parameters, the augmentation is performed accordingly.
  Apart from transforming image tensors, Kornia also supports inverse operations that revert the transform,
  and transforms of other data modalities (`data keys` in Kornia) such as masks, keypoints, and bounding boxes.
  Such features are best used through `AugmentationSequential`. Notably, the full pipeline for rigid
  operations is already implemented and needs no further effort. For non-rigid operations, the user may implement
  customized inverse and data-modality operations, e.g. `apply_transform_mask` for transforming mask tensors.

Custom Augmentation Classes
---------------------------

For rigid transformations, `IntensityAugmentationBase2D` and `GeometricAugmentationBase2D` share the exact same logic
apart from the transformation matrix computation. Namely, an intensity augmentation always results in an
identity transformation matrix, since it does not change the geometric location of any pixel.

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

The most common case is a pixel-wise augmentation with a random per-sample parameter. Subclass
`IntensityAugmentationBase2D`, declare a parameter generator in ``__init__``, and read the sampled
value in `apply_transform`:

.. code-block:: python

   from typing import Any, Dict, Optional

   import torch
   from torch import Tensor

   from kornia.augmentation import IntensityAugmentationBase2D
   from kornia.augmentation import random_generator as rg

   class RandomAddValue(IntensityAugmentationBase2D):
       """Add a per-sample value drawn uniformly from ``add_range``."""

       def __init__(self, add_range=(0.0, 0.2), same_on_batch=False, p=1.0, keepdim=False):
           super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
           # A PlainUniformGenerator sampler is a 4-tuple ``(range, name, center, bound)``:
           # sample a value inside ``range`` and expose it as ``params[name]``. ``center`` and
           # ``bound`` (``None`` here) are optional constraints for centred/bounded ranges.
           self._param_generator = rg.PlainUniformGenerator((add_range, "add", None, None))

       def apply_transform(
           self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any],
           transform: Optional[Tensor] = None,
       ) -> Tensor:
           # ``params["add"]`` has shape ``(B,)`` — reshape to broadcast over C, H, W.
           add = params["add"].to(input).view(-1, 1, 1, 1)
           return input + add

   aug = RandomAddValue((0.0, 0.2), p=1.0)
   out = aug(torch.rand(4, 3, 32, 32))                       # a different value per sample
   again = aug(torch.rand(4, 3, 32, 32), params=aug._params)  # reproduce with the stored params

Static (non-random) configuration goes in ``self.flags`` (a plain dict), read from the ``flags``
argument of `apply_transform`. A custom augmentation works standalone and inside
`AugmentationSequential` with no extra wiring.

For a rigid **geometric** augmentation, also implement `compute_transformation` to return the
``(B, 3, 3)`` transform matrix — kornia then applies it, inverts it, and propagates it to masks,
boxes and keypoints:

.. code-block:: python

   from typing import Any, Dict, Optional

   import torch
   from torch import Tensor

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

      def compute_transformation(self, input, params, flags):
         # return the (B, 3, 3) transform matrix for this augmentation
         # (identity shown for brevity; kornia applies, inverts and propagates it)
         return K.eye_like(3, input)

      def apply_transform(
         self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
      ) -> Tensor:
         factor = params["factor"].to(input).view(-1, 1, 1, 1)
         return input * factor

For non-rigid augmentations, the user may implement the `apply_transform*` and `apply_non_transform*` APIs
as needed. Specifically, `apply_transform*` applies to the elements of a batch that are selected for augmentation,
while `apply_non_transform*` applies to the elements that are skipped. For example, a crop operation changes the size
of the selected elements, so the skipped elements must be resized as well to keep the whole batch tensor at one size.

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

The same logic applies to 3D augmentations as well.

Some Further Notes
------------------

Probabilities
^^^^^^^^^^^^^
Kornia supports two types of randomness: element-level randomness `p` and batch-level randomness `p_batch`,
as defined in `_BasicAugmentationBase`. Under the hood, operations like `crop` and `resize` are implemented with a fixed
element-level probability of `p=1` and only keep the batch-level randomness.

Random Generators
^^^^^^^^^^^^^^^^^
To get an automatically generated ``__repr__`` that lists all custom parameters, implement
``_param_generator`` by inheriting from ``RandomGeneratorBase`` to generate the random parameters, and
put all static parameters inside ``self.flags``. You can take advantage of ``PlainUniformGenerator`` to
generate simple uniform parameters with less boilerplate code.

Random Reproducibility
^^^^^^^^^^^^^^^^^^^^^^
By default, the random parameters are sampled on the CPU with ``torch.get_default_dtype()``, independently of the
device of the input, so a seeded run gives the same parameters on CPU and GPU.
To change this behaviour, use ``set_rng_device_and_dtype``.
