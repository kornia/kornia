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
      │  │     ├─ IntensityAugmentationBase2D    intensity ops — override apply_transform
      │  │     └─ GeometricAugmentationBase2D    warp ops — also override compute_transformation
      │  └─ AugmentationBase3D … (the 3D mirror of the 2D chain)
      └─ MixAugmentationBaseV2        mix ops (MixUp / CutMix) — bypass the per-key dispatch

Each level is a distinct, reused axis (sampling, data-key dispatch, 2D vs 3D, rigid-matrix vs free-form,
intensity vs geometric); the four ``*Base2D`` classes are public API that external code subclasses.
For a custom augmentation, subclass ``IntensityAugmentationBase2D`` or ``GeometricAugmentationBase2D``.

The Predefined Augmentation Routine
-----------------------------------

Kornia augmentations generally follow a `sample-apply` routine.

- `sample`: Kornia aims at flexible tensor-level augmentations that augment every image in a batch with
  different parameters and probabilities. The sampling step first draws a set of random
  parameters. The sampled augmentation state is then stored in the ``_params`` attribute of the augmentation,
  for replay. Application-time draws can require additional RNG control; see the reproducibility notes below.
- `apply`: with the generated (or user-provided) parameters, the augmentation is performed accordingly.
  Apart from transforming image tensors, Kornia also supports inverse operations that revert the transform,
  and transforms of other data modalities (`data keys` in Kornia) such as masks, keypoints, and bounding boxes.
  These features depend on the concrete operation and its data-key handlers. `AugmentationSequential` passes the
  matrix of the call's parameters to the mask, box and keypoint handlers of geometric children and of custom
  `RigidAffineAugmentationBase2D` subclasses, recomputing it from replayed ``params`` that the child did not last
  apply to an image or that were edited since. An in-place edit made inside ``torch.inference_mode`` is not seen,
  because inference tensors have no version counter; replace the value instead. Non-rigid coordinate transforms
  are not supplied automatically (`#4420 <https://github.com/kornia/kornia/issues/4420>`_).

Custom Augmentation Classes
---------------------------

`IntensityAugmentationBase2D` supplies an identity matrix and default passthrough handlers for most annotations.
Subclasses can override these defaults: `RandomErasing` zero-fills the erased mask region. Direct and container
calls pass masks, boxes, and keypoints through unchanged unless a subclass overrides
`apply_transform_mask`, `apply_transform_box`, or `apply_transform_keypoint`.
`GeometricAugmentationBase2D` supplies the dispatch used for geometric coordinate transformations.

For a geometric operation, implement `compute_transformation` and `apply_transform`. Supporting image inversion
also requires `inverse_transform`; the base provides matrix inversion.

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
   again = aug(torch.rand(4, 3, 32, 32), params=aug._params)  # reuse the recorded draw

Static (non-random) configuration goes in ``self.flags`` (a plain dict), read from the ``flags``
argument of `apply_transform`. A custom augmentation works standalone and inside
`AugmentationSequential` with no extra wiring.

For a rigid **geometric** augmentation, implement `compute_transformation` to return the
``(B, 3, 3)`` matrix and `apply_transform` to implement the corresponding image operation.
The geometric base propagates the matrix to boxes and keypoints; mask processing uses its mask handler.
The following example only illustrates the method signatures, not an invertible geometric warp:

.. code-block:: python

   from typing import Any, Dict, Optional

   import torch
   from torch import Tensor

   from kornia.augmentation import GeometricAugmentationBase2D
   from kornia.augmentation import random_generator as rg
   from kornia.core.ops import eye_like

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
         # identity shown only to illustrate the required matrix shape
         return eye_like(3, input)

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

`RigidAffineAugmentationBase2D` sits between `AugmentationBase2D` and the two rigid bases. It adds the
transform-matrix machinery — `compute_transformation` and the `transform_matrix` attribute — but no `inverse`.

.. autoclass:: RigidAffineAugmentationBase2D

   .. automethod:: compute_transformation

The 3D bases provide analogous shape, matrix and data-key machinery; see :class:`AugmentationBase3D` for their
contract, including the missing inverse.

.. autoclass:: AugmentationBase3D

.. autoclass:: RigidAffineAugmentationBase3D

.. autoclass:: GeometricAugmentationBase3D

.. autoclass:: IntensityAugmentationBase3D

Mix augmentations derive from `MixAugmentationBaseV2` and have their own forward and data-key contracts.
Some combine different samples; `RandomJigsaw` rearranges patches within each image. Label and box handling
are class-specific: `RandomMixUpV2` accepts class labels, while `RandomMosaic` accepts boxes but does not
implement class-label transforms. `RandomTransplantation` and `RandomTransplantation3D` require a segmentation
mask; a mask-only call needs ``data_keys=["mask"]``. The 3D transplantation class also inherits
`AugmentationBase3D`.

.. autoclass:: MixAugmentationBaseV2

Some Further Notes
------------------

Probabilities
^^^^^^^^^^^^^
:class:`AugmentationBase2D` states the ``p`` / ``p_batch`` model. A concrete constructor can map its public ``p``
to either gate: `RandomMixUpV2` gates the batch and `RandomJigsaw` individual samples
(`#4425 <https://github.com/kornia/kornia/issues/4425>`_). When ``0 < p_batch < 1``, the base draws the batch
Bernoulli before the per-sample gate; endpoints skip that draw, and ``p=1.0, p_batch=0.0`` selects no sample.

With static ``p=1.0, p_batch=1.0`` every sample is transformed: a ``batch_prob`` put into ``params`` is not consulted
by the image, the transformation matrix, the labels or ``inverse``, in eager and compiled runs alike
(`#5585 <https://github.com/kornia/kornia/issues/5585>`_). Pass ``p < 1`` to skip samples by hand.

Random Generators
^^^^^^^^^^^^^^^^^
To get an automatically generated ``__repr__`` that lists all custom parameters, implement
``_param_generator`` by inheriting from ``RandomGeneratorBase`` to generate the random parameters, and
put all static parameters inside ``self.flags``. You can take advantage of ``PlainUniformGenerator`` to
generate simple uniform parameters with less boilerplate code.

Random Reproducibility
^^^^^^^^^^^^^^^^^^^^^^
See :class:`AugmentationBase2D` for where parameters are sampled and what ``params=`` replays, and
:doc:`/get-started/conventions` for seeding, worker seeds and consumption order.

Serialization
^^^^^^^^^^^^^
- Several constructors that accept ``nn.Parameter`` ranges propagate gradients to them.
- A ``kornia.augmentation.auto`` policy pickles when every operation wrapper in it does, and a wrapper does when
  its magnitude mapping is picklable; the built-in mappings are.
- :class:`AugmentationBase2D` covers the range buffers in ``state_dict()``, and
  :class:`RigidAffineAugmentationBase2D` what a lazily built matrix keeps.

Explicit random streams
-----------------------

Built-in 2D/3D augmentations and image, video, patch and augmentation containers support a persistent
stream via ``set_generator`` and a temporary stream via ``generator=`` on forward or parameter sampling.
A container passes its stream to its built-in augmentation children, including random order selection.
The call-local stream restores the previous child streams even if the call raises.

.. code-block:: python

   import torch
   import kornia.augmentation as K

   image = torch.linspace(0, 1, 128).reshape(2, 1, 8, 8)
   generator = torch.Generator(device="cpu").manual_seed(123)
   aug = K.RandomAffine(45.0).set_generator(generator)
   augmented = aug(image)
   # A different stream for this call only; the persistent stream is not consumed.
   augmented = aug(image, generator=torch.Generator().manual_seed(456))
   # Compile a deterministic application separately from eager parameter sampling.
   params = aug.forward_parameters(image.shape, generator=generator)

Generators must match the sampling device, normally CPU regardless of the image's device.
``.to()`` does not move a generator. Its state is not included in ``state_dict``: persist it with
``generator.get_state()`` and restore it with ``generator.set_state()``. PyTorch does not guarantee
identical draws across devices or releases.

With a stream bound, every built-in sampler draws on that stream's device. A few samplers build their
bounds from host-side constants, so the ``None`` path keeps drawing them on CPU even under
``set_rng_device_and_dtype("cuda")``; binding a CUDA stream moves those draws to CUDA, which keeps an
explicit run identical to the equivalent globally seeded run. Generator-free results are therefore
unchanged only for the default CPU sampling device.

A DataLoader worker must receive its own appropriately seeded generator. The workers inherit a copy of the
persistent stream, so without a fix each worker replays the same draws; copy the state to every worker and
they all stay duplicated. Seed a fresh generator per worker from the worker's own seed:

.. code-block:: python

   def worker_init_fn(worker_id):
       info = torch.utils.data.get_worker_info()
       info.dataset.aug.set_generator(torch.Generator().manual_seed(info.seed))

   loader = torch.utils.data.DataLoader(dataset, num_workers=2, worker_init_fn=worker_init_fn)

``info.seed`` differs per worker, so each worker samples its own stream while the global RNG is left alone.

This first implementation covers built-in parameter/noise samplers and the standard containers.
Auto-augmentation policies, model presets and ``RandomDissolving`` reject explicit streams before sampling; their learned
probability/model sampling requires separate work. Custom augmentations and arbitrary ``nn.Module`` children
must explicitly use the supplied stream themselves; ``set_generator`` does not intercept arbitrary torch
calls. Use eager parameter sampling with an explicit generator and replay recorded parameters when compiling
or exporting a transform. As with their recorded call state, simultaneous calls to the same augmentation
instance are not supported; independent instances with independent generators can run concurrently.
