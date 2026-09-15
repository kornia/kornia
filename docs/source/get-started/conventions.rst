Conventions & Pitfalls
======================

.. meta::
   :description: The conventions every Kornia function follows: (B, C, H, W) float images in [0, 1], (x, y) points versus (h, w) sizes, degrees versus radians, homography direction, align_corners defaults, box formats and a pitfall checklist.

Every Kornia function follows the conventions below unless its documentation
explicitly says otherwise. If you are generating code (human or LLM), read
this page first — nearly every subtle Kornia bug is a convention mismatch,
not a math error.

Image tensors
-------------

- Images are 4D float tensors ``(B, C, H, W)``, channel order **RGB**, values
  in ``[0, 1]``. The batched layout is accepted everywhere; some op families
  (e.g. color conversions) also accept ``(*, C, H, W)`` with arbitrary
  leading dims.
- Ops run on the device/dtype of their inputs. There is no implicit
  ``.cuda()``, ``.float()``, or value rescaling.
- Convert NumPy HWC images with :func:`kornia.image.image_to_tensor`
  (``kornia.utils.image_to_tensor`` is deprecated since 0.8.3):

.. code-block:: python

    import numpy as np
    import torch
    import kornia

    np_img = (np.random.rand(48, 64, 3) * 255).astype(np.uint8)  # (H, W, C) uint8
    t = kornia.image.image_to_tensor(np_img)[None].float() / 255.0  # (1, 3, 48, 64) in [0, 1]

Coordinates and sizes
---------------------

- Point coordinates are ``(x, y)``: x indexes **columns**, y indexes
  **rows**, origin at the **top-left** pixel. Keypoint tensors are
  ``(B, N, 2)``.
- Sizes and ``dsize`` arguments are ``(h, w)`` — the *opposite* order from
  points. ``warp_perspective(img, M, dsize=(2, 8))`` produces a 2-row,
  8-column image.
- Normalized coordinates, where used, are ``[-1, 1]`` in both axes,
  identical to :func:`torch.nn.functional.grid_sample` **called with**
  ``align_corners=True`` — not to its default, ``align_corners=False``,
  which places the same values up to half a pixel off — exactly half a
  pixel at the image borders, and identically at the image center.
  :func:`kornia.geometry.create_meshgrid` returns a normalized grid by
  default (``normalized_coordinates=True``).
- 3D grids and 3D pixel coordinates are ``(d, x, y)`` — depth first, not
  ``(x, y, z)``; :func:`kornia.geometry.grid.create_meshgrid3d` produces this
  order and the ``*_pixel_coordinates3d`` conversions consume it.
- Pixel ``(0, 0)`` is centred at ``(0, 0)`` — the OpenCV "integer" convention,
  not COLMAP's half-pixel one. :doc:`camera-conventions` catalogues the
  pixel-centre and camera-frame conventions of the surrounding ecosystem and
  the converters between them.

Angles and rotations
--------------------

- Angles are **degrees** in the 2D image APIs (``rotate``,
  ``get_rotation_matrix2d``, ``RandomRotation``,
  ``angle_to_rotation_matrix``), and **radians** in the 3D
  rotation-representation APIs (``axis_angle_to_rotation_matrix``, ``So3``)
  and in the polar conversions ``cart2pol``/``pol2cart``
  (``rad2deg``/``deg2rad`` exist to convert).
- 2D image rotations: positive angle rotates **counter-clockwise as
  displayed** (top-left origin, matching OpenCV):

.. code-block:: python

    import torch
    from kornia.geometry.transform import rotate

    img = torch.zeros(1, 1, 5, 5)
    img[0, 0, 1, 3] = 1.0  # marker up-right of center
    out = rotate(img, torch.tensor([90.0]))
    assert out[0, 0].round().nonzero().tolist() == [[1, 1]]  # moved up-LEFT: CCW on screen

- 3D rotation conversions follow the **right-hand rule in math convention**.
  Because image y points *down*, a positive rotation about +z from
  :func:`kornia.geometry.conversions.axis_angle_to_rotation_matrix` moves
  image points **clockwise on screen** — the opposite screen direction from
  ``rotate(img, +angle)``. Do not mix the two without negating the angle:

.. code-block:: python

    import torch
    from kornia.geometry.conversions import axis_angle_to_rotation_matrix

    R = axis_angle_to_rotation_matrix(torch.tensor([[0.0, 0.0, torch.pi / 2]]))
    # math convention: (1, 0) -> (0, 1). With y down on screen, that points DOWNWARD.
    assert torch.allclose(R[0, :2, :2], torch.tensor([[0.0, -1.0], [1.0, 0.0]]), atol=1e-6)

- Quaternions use **WXYZ** coefficient order (scalar first):

.. code-block:: python

    from kornia.geometry.quaternion import Quaternion

    q = Quaternion.identity()
    assert q.data.tolist() == [1.0, 0.0, 0.0, 0.0]  # w, x, y, z

Transformation matrices and homographies
----------------------------------------

- Transformation matrices are **batched**: homographies ``(B, 3, 3)``,
  affine ``(B, 2, 3)``. Add the batch dim to a single matrix with
  ``M[None]``.
- :func:`kornia.geometry.transform.warp_perspective` takes the
  **source→destination** homography in **pixel** coordinates.
- :func:`kornia.geometry.transform.homography_warp` is different on every
  axis: it takes the **destination→source** homography, **normalized** to
  ``[-1, 1]`` by default (``normalized_homography=True``), and defaults to
  ``align_corners=False``. Convert a pixel source→destination homography by
  normalizing it FIRST with
  :func:`kornia.geometry.conversions.normalize_homography` (which expects
  the forward src→dst homography) and inverting AFTER — inverting first
  with unswapped sizes is silently wrong whenever the source and
  destination sizes differ:

.. code-block:: python

    import torch
    from kornia.geometry.conversions import normalize_homography
    from kornia.geometry.transform import homography_warp, warp_perspective

    img = torch.rand(1, 1, 8, 8)
    M = torch.tensor([[[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]])  # src->dst, pixels

    a = warp_perspective(img, M, (16, 8))  # note: dst size differs from src
    M_norm_inv = torch.inverse(normalize_homography(M, (8, 8), (16, 8)))  # normalize, THEN invert
    b = homography_warp(img, M_norm_inv, (16, 8), align_corners=True)
    assert torch.allclose(a, b, atol=1e-5)

``normalize_homography`` takes its own ``align_corners`` (default ``True``),
and it must match the one you pass to the warp — the normalized ``[-1, 1]``
coordinates mean different things under the two conventions. Above, both are
``True``; note that ``homography_warp`` alone would default to ``False``.

``align_corners`` defaults
--------------------------

Defaults are **not uniform** across the library. When mixing Kornia warps
with ``torch.nn.functional.interpolate``/``grid_sample``, pass
``align_corners`` explicitly everywhere.

.. list-table::
   :header-rows: 1

   * - Function
     - ``align_corners`` default
   * - ``warp_perspective``, ``warp_affine``, ``rotate``
     - ``True``
   * - ``resize``
     - ``None`` (PyTorch's per-mode default)
   * - ``homography_warp``, ``elastic_transform2d``
     - ``False``
   * - ``undistort_image``
     - ``True``
   * - ``warp_frame_depth``
     - ``True``
   * - ``DepthWarper`` / ``depth_warp``
     - ``True`` (default)
   * - ``remap``
     - ``None`` (resolved to ``False`` internally)

The flag selects only how Kornia normalizes coordinates for ``grid_sample``
(``True``: pixel *centers* 0 and ``size-1`` sit at ``±1``; ``False``: the outer
pixel *edges* do). Transforms you pass in are pixel-space either way, so a warp
that should be an identity is one under both settings, and ``warp_affine`` and
``warp_perspective`` agree with each other:

.. code-block:: python

    import torch
    from kornia.geometry.transform import get_perspective_transform, warp_affine, warp_perspective

    img = torch.arange(16.0).view(1, 1, 4, 4)
    pts = torch.tensor([[[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0]]])
    M = get_perspective_transform(pts, pts)  # identity

    for align_corners in (True, False):
        a = warp_affine(img, M[:, :2, :], (4, 4), align_corners=align_corners)
        p = warp_perspective(img, M, (4, 4), align_corners=align_corners)
        assert torch.allclose(a, img, atol=1e-4)
        assert torch.allclose(p, img, atol=1e-4)

Where the two settings genuinely differ is out-of-bounds sampling, since ``±1``
covers a slightly different extent of the source image.

.. warning::

   :func:`kornia.geometry.transform.remap` is the 2D exception left: it normalizes its
   pixel maps with the ``align_corners=True`` convention regardless of the flag it passes
   to ``grid_sample``, so with its default (``None``, i.e. ``False``) even an identity
   pixel map resamples the image, by ``11.25`` on a 4x4 ``arange`` image. Pass
   ``align_corners=True`` until this is fixed. Tracked in
   `#4504 <https://github.com/kornia/kornia/issues/4504>`_.

   The 3-D warps are the other exception. ``normal_transform_pixel3d`` and
   ``normalize_homography3d`` take no ``align_corners`` at all, so
   :func:`kornia.geometry.transform.warp_affine3d` and
   :func:`kornia.geometry.transform.warp_perspective3d` normalize with the corner-aligned
   convention whatever flag they pass to ``grid_sample``, and have the same mismatch at
   ``align_corners=False``: an identity ``warp_perspective3d`` changes a 4x4x4 ``arange``
   volume by up to ``55.1`` there, against exactly ``0`` at ``align_corners=True``. Pass
   ``align_corners=True`` to the 3-D warps until this is fixed. Tracked in
   `#4503 <https://github.com/kornia/kornia/issues/4503>`_.

Bounding boxes
--------------

- The ``kornia.geometry.bbox`` module uses ``(B, 4, 2)`` corner format:
  clockwise from top-left, ``(x, y)`` per corner.
- Width/height are **inclusive**:
  :func:`kornia.geometry.bbox.infer_bbox_shape` computes
  ``width = x_right - x_left + 1``. A box with corners (1,1) and (2,2) has
  width 2, not 1:

.. code-block:: python

    import torch
    from kornia.geometry.bbox import infer_bbox_shape

    boxes = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]])
    h, w = infer_bbox_shape(boxes)
    assert (h.item(), w.item()) == (2.0, 2.0)

- :class:`kornia.augmentation.container.AugmentationSequential` accepts three box
  formats via ``data_keys``: ``"bbox"`` (4-corner), ``"bbox_xyxy"``, and
  ``"bbox_xywh"``. Keypoints are ``"keypoints"``, ``(B, N, 2)`` in
  ``(x, y)``.

Color
-----

- ``rgb_to_hsv`` returns hue in **radians** ``[0, 2π)`` — not degrees, not
  ``[0, 1]``:

.. code-block:: python

    import torch
    import kornia

    green = torch.zeros(1, 3, 1, 1)
    green[0, 1] = 1.0
    hue = kornia.color.rgb_to_hsv(green)[0, 0].item()
    assert abs(hue - 2.0943951) < 1e-4  # 120 degrees = 2*pi/3 radians

Augmentations
-------------

- Use one :class:`kornia.augmentation.container.AugmentationSequential` call
  for an image and its annotations. Geometric children that implement the
  corresponding data-key handlers share their recorded parameters across
  those inputs. Non-rigid operations, mask lists and dtype conversion have
  the limitations below; a transform matrix alone does not establish support
  for every data key.
- ``.inverse()`` recovers keypoint coordinates up to numerical precision when
  every geometric step supports inversion. Slice-mode crops raise; use their
  ``cropping_mode="resample"`` where inversion is needed. Geometric 3D
  children raise too. Intensity and non-rigid children are skipped, leaving
  their effects applied. Resampling cannot recover information lost through
  cropping, padding or interpolation. Tensor box outputs use axis-aligned
  enclosures, so a rotation can lose the corners needed to restore a box.
- Non-rigid children do not carry coordinate annotations along with the
  image. :class:`kornia.augmentation.RandomElasticTransform` warps masks
  through its displacement field but leaves keypoints and boxes unchanged.
  :class:`kornia.augmentation.RandomThinPlateSpline` and
  :class:`kornia.augmentation.RandomFisheye` leave coordinates unchanged and
  raise on masks (`#4420 <https://github.com/kornia/kornia/issues/4420>`_).
  Intensity transformations usually leave masks unchanged, but
  :class:`kornia.augmentation.RandomErasing` fills the erased mask region with
  zero.
- The geometric mask path normally uses nearest interpolation. That avoids
  interpolating labels, but padding can still introduce its fill value.
- Put the image before masks, including in dictionary insertion order, so
  conversion uses its working dtype. Earlier masks use the previous image
  dtype, or ``float32`` on a fresh container. Integer labels outside the
  working dtype's exact range can change even through a flip; for example,
  ``2049`` becomes ``2048`` in ``float16``. Every output mask is cast to the
  last mask argument's dtype (its first element's dtype for a list).
  A common mask dtype avoids that cross-mask conversion, but does not prevent
  precision loss during processing. Empty batches with masks can also raise
  (`#4478 <https://github.com/kornia/kornia/issues/4478>`_). Direct geometric
  ``transform_masks`` calls require floating tensors; the container converts
  integer and boolean masks around those calls.
- Mask lists are not a reliable replacement for separate full-batch tensors.
  Direct augmentation children use list index ``i`` to select a sample's
  gate. Full-batch entries can therefore become desynchronized under mixed
  gates; per-sample entries can fail in warps or reuse the wrong crop window.
  Prefer separate ``mask`` keys with a common dtype, subject to the precision
  and filtering limitations above
  (`#4477 <https://github.com/kornia/kornia/issues/4477>`_).
- Boxes use the inclusive ``xyxy_plus`` convention of
  :class:`kornia.geometry.boxes.Boxes` (see *Bounding boxes* above). Flips use
  integer pixel centres: ``x' = W - 1 - x``.
- Nested containers can expose missing or stale transformation matrices,
  even with only rigid children. Do not rely on the outer
  ``.transform_matrix`` to describe a nested pipeline
  (`#4476 <https://github.com/kornia/kornia/issues/4476>`_).
- Dictionary keys use raw prefix matching with exceptions for coordinate box
  names. Unrecognized metadata is removed from the caller's dictionary before
  being returned in the output. See the container's dictionary guidance and
  `#4483 <https://github.com/kornia/kornia/issues/4483>`_.
- A positive ``degrees`` turns the displayed image counter-clockwise with
  :class:`kornia.augmentation.RandomRotation`, matching
  :func:`kornia.geometry.transform.rotate`, and clockwise with
  :class:`kornia.augmentation.RandomAffine`
  (`#4408 <https://github.com/kornia/kornia/issues/4408>`_).
- The 2D intensity augmentations assume the ``[0, 1]`` float range, and no
  base-class check validates it on the way in. They disagree about what
  happens outside that range: some keep
  the output inside ``[0, 1]``, :class:`kornia.augmentation.RandomPlanckianJitter`
  bounds only the upper end, some carry the input's range through, and
  :class:`kornia.augmentation.RandomEqualize` raises. Several return an
  all-zero image for an input whose values are all negative
  (`#4430 <https://github.com/kornia/kornia/issues/4430>`_). See
  :class:`kornia.augmentation.IntensityAugmentationBase2D`, and each class's
  own documentation for which of the four it is.

.. code-block:: python

    import torch
    import kornia.augmentation as K

    aug = K.AugmentationSequential(
        K.RandomAffine(degrees=30.0, p=1.0),
        data_keys=["input", "mask", "keypoints"],
    )
    image = torch.rand(1, 3, 64, 64)
    mask = (torch.rand(1, 1, 64, 64) > 0.5).float()
    kpts = torch.tensor([[[16.0, 16.0], [48.0, 32.0]]])
    img_out, mask_out, kpts_out = aug(image, mask, kpts)
    img_back, mask_back, kpts_back = aug.inverse(img_out, mask_out, kpts_out)
    assert (kpts_back - kpts).abs().max() < 1e-3

Randomness in augmentations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Parameter sampling generally starts on the CPU, independently of the image
  device. Application-time randomness can instead use the input or model
  device. Sampling precision depends on the generator, constructor ranges
  and torch defaults; changing the default dtype does not uniformly update
  cached samplers.
- ``set_rng_device_and_dtype`` requests a sampling device and dtype. It
  updates the gate configuration and asks the parameter generator to rebuild
  its samplers, but some generators retain internal CPU tensors or ignore
  the requested precision
  (`#4426 <https://github.com/kornia/kornia/issues/4426>`_).
  Returned parameter placement is separate: an affine without shear can
  sample angles on MPS and return them on CPU. Numeric ranges or tensor-valued
  constructor ranges can determine the returned device/dtype, so inspecting
  ``_params`` alone does not establish where or at what precision draws ran.
- Module migration through ``.to(...)`` also updates the augmentation gate
  and registered parameter generators' sampling configuration, including
  moves through a container. A dtype-only move preserves the sampling device;
  a device-only move preserves its dtype. Invalid integer-dtype requests are
  rejected before changing the samplers. This does not make every generator
  support every device/dtype, or force returned parameters onto the sampling
  device. This limitation affects multiple generators, including numeric-range
  ``RandomAffine``, ``RandomPerspective``, ``RandomRotation``, ``RandomCrop``,
  and ``RandomShear`` with scalar, pair, or four-value ranges: returned transform
  parameters can remain CPU float32 while ``batch_prob`` is on the accelerator
  (`#4426 <https://github.com/kornia/kornia/issues/4426>`_).
- Reproducibility uses the global generators on the sampling devices.
  ``torch.manual_seed`` reproduces draws for the same configuration, inputs,
  backend and dtype; matching across devices or PyTorch versions is not
  promised. There is no general per-instance ``generator=`` interface.
  The standard :class:`kornia.augmentation.AugmentationBase2D` forward accepts
  and ignores unknown keywords, while mix forwards can reject them
  (`#4427 <https://github.com/kornia/kornia/issues/4427>`_).
- On the standard 2D base forward path, ``params=`` stores the caller's
  dictionary by reference and inserts an all-true gate if ``batch_prob`` is
  missing. Mix augmentations use a separate forward contract and can require
  that key. Prefer complete recorded parameters for replay. The recorded
  parameters capture every draw, the ``RandomPlasma*`` noise included, except
  for ``RandomDissolving``, which samples VAE latents during application.
  Replaying it also needs control of that application-time random state.
- ``same_on_batch=True`` shares the standard per-sample gate and sampled
  transform factors. It does not equate batch-pairing indices, and the
  ``RandomPlasma*`` noise stored in the parameters is drawn independently per
  sample whatever the flag says.
  Color adjustment order is one permutation shared across the batch,
  independently of this flag. On ``AugmentationSequential``, ``None`` keeps
  each child's setting, while ``True`` and ``False`` overwrite it.
- A concrete constructor's ``p`` can configure either the base's per-sample
  gate or its whole-batch gate. Mix augmentations in particular do not share
  one probability contract. Check the concrete class rather than inferring
  its gate from the base signature. Exposing ``p_batch`` is also
  constructor-dependent (`#4425 <https://github.com/kornia/kornia/issues/4425>`_).
- Under :class:`torch.utils.data.DataLoader`, each worker's global CPU
  generator is seeded ``base_seed + worker_id``. Reproducibility also depends
  on worker configuration and consumption order. A ``worker_init_fn`` that
  reseeds every worker to the same fixed value duplicates their random
  streams. See `Randomness in multi-process data loading
  <https://pytorch.org/docs/stable/notes/randomness.html#dataloader>`_.
- Changing random draw consumption can shift later draws in a seeded
  pipeline; reordering draws can change which parameter gets each value.
  The convention tests compare final RNG states to reference operations,
  so they check consumption, not assignment of values to parameter keys.

Serializing an augmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Numeric ranges do not create trainable range parameters. Several
  constructors, including ``RandomRotation`` and ``RandomAffine``, accept
  ``nn.Parameter`` ranges that appear in ``named_parameters()`` and
  ``state_dict()``. Gradients depend on the operation, backend and dtype.
- Some numeric ranges are copied into buffers, but loading those buffers
  does not update the cached sampling distributions; ``repr`` may or may not
  reflect the loaded range. Reconstruct the augmentation to change what it
  samples (`#4428 <https://github.com/kornia/kornia/issues/4428>`_).
- Pickle and deepcopy can retain recorded parameters and transform state,
  but support is configuration-dependent. The ``kornia.augmentation.auto``
  policies cannot currently be pickled
  (`#4469 <https://github.com/kornia/kornia/issues/4469>`_).
  A normal forward draws fresh parameters; replay requires passing the
  saved parameters and controlling any application-time randomness.
- Built-in lazy matrices keep only the input's shape, dtype and device
  alongside transformation parameters, so an unread matrix does not keep
  the image batch in the saved state. A lazy subclass overriding ``transform_tensor``,
  ``generate_transformation_matrix``, ``compute_transformation`` or
  ``identity_matrix`` retains the input until the matrix is read or another
  forward replaces the pending state. Unchanged inherited implementations
  keep the compact metadata state. See :doc:`/augmentation.base` for details.

Pitfall checklist
-----------------

Quick self-review for generated code, most common first:

1. ``(H, W, C)`` NumPy array passed where a tensor is expected — convert
   with ``kornia.image.image_to_tensor(np_img)[None]``.
2. ``(w, h)`` passed to a ``dsize``/``size`` argument — they are ``(h, w)``.
3. ``(y, x)`` (row, col) point order — points are ``(x, y)``.
4. Assuming uniform ``align_corners`` defaults — see the table above.
5. Unbatched ``(3, 3)`` homography — add the batch dim: ``M[None]``.
6. Radians passed to the degree APIs (``rotate``, ``get_rotation_matrix2d``)
   or degrees passed to the radian APIs (``axis_angle_*``).
7. Uint8 ``[0, 255]`` values where float ``[0, 1]`` is expected.
8. Image and mask augmented through two separate augmentation calls.
9. ``homography_warp`` fed a source→destination pixel homography — it wants
   destination→source, normalized (or pass ``normalized_homography=False``).
10. Mixing ``axis_angle_to_rotation_matrix`` (math convention; screen-
    clockwise for +z) with ``rotate`` (screen-counter-clockwise) without
    negating the angle.
11. Treating ``infer_bbox_shape`` output as exclusive width/height — it is
    inclusive (``+ 1``).
12. Quaternions constructed in XYZW order — Kornia uses WXYZ.
13. Expecting hue in ``[0, 360]`` or ``[0, 1]`` — ``rgb_to_hsv`` returns
    radians ``[0, 2π)``.
14. Wrong ``data_keys`` box format — ``"bbox"`` means 4-corner ``(B, N, 4, 2)``;
    use ``"bbox_xyxy"``/``"bbox_xywh"`` for coordinate formats.
15. Assuming one rotation direction across the augmentations — a positive
    ``degrees`` on ``RandomAffine`` turns the image clockwise, on
    ``RandomRotation`` counter-clockwise.
16. Expecting a non-rigid augmentation (``RandomElasticTransform``,
    ``RandomThinPlateSpline``, ``RandomFisheye``) to carry boxes and keypoints
    along with the image — it does not. Only ``RandomElasticTransform`` carries
    a mask along, and the other two raise on a ``mask`` key.
17. Inferring the augmentation sampling backend from ``_params`` placement
    — samplers can draw on an accelerator and cast the returned tensors back to CPU.
18. Feeding mean/std-normalized or otherwise out-of-``[0, 1]`` tensors
    through an intensity augmentation and expecting the values to pass
    through — some rescale, some clamp, and several return zeros for an
    all-negative image.

.. tip::

   Machine-readable copies of these conventions live at
   `llms.txt <https://kornia.readthedocs.io/en/latest/llms.txt>`_ and
   `llms-full.txt <https://kornia.readthedocs.io/en/latest/llms-full.txt>`_
   at the docs root.
