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

.. _coordinate-conventions:

Coordinates and sizes
---------------------

- Point coordinates are ``(x, y)``: x indexes **columns**, y indexes
  **rows**, origin at the **top-left** pixel. Keypoint tensors are
  ``(B, N, 2)``.
- Sizes and ``dsize`` arguments are ``(h, w)`` — the *opposite* order from
  points. ``warp_perspective(img, M, dsize=(2, 8))`` produces a 2-row,
  8-column image.
- Normalized coordinates, where used, are ``[-1, 1]`` in both axes.
  :func:`kornia.geometry.grid.create_meshgrid` defaults to a corner-aligned
  normalized grid (``normalized_coordinates=True``, ``align_corners=True``),
  matching :func:`torch.nn.functional.grid_sample` with ``align_corners=True``:
  the first and last pixel centres map to the endpoints. Passing
  ``align_corners=False`` to ``create_meshgrid`` uses the half-pixel mapping
  that matches ``grid_sample(..., align_corners=False)`` instead, where the
  endpoints are the outer pixel edges. Use the same flag in both calls.
- 3D grids and 3D pixel coordinates are ``(d, x, y)`` — depth first, not
  ``(x, y, z)``; :func:`kornia.geometry.grid.create_meshgrid3d` produces this
  order and the ``*_pixel_coordinates3d`` conversions consume it.
  :func:`torch.nn.functional.grid_sample` reads a 3D grid as ``(x, y, z)``:
  pass ``grid[..., [1, 2, 0]]`` and ``align_corners=True``, because the
  normalized 3D grid is corner-aligned and has no ``align_corners`` argument
  (`#4503 <https://github.com/kornia/kornia/issues/4503>`_).
- Sub-pixel outputs follow the same orders: the soft-argmax functions of
  :doc:`kornia.geometry.subpix </geometry.subpix>` return ``(x, y)`` in 2D
  and ``(d, x, y)`` in 3D, and the quadratic refiners
  (``conv_quad_interp3d``, ``iterative_quad_interp3d``) return ``(d, x, y)``
  voxel indices of their input. Normalized outputs are corner-aligned, as
  above.
- Non-maximum suppression is **strict** in
  :func:`kornia.geometry.subpix.nms2d`, ``nms3d``, ``nms3d_minmax`` and the
  detectors built on them: with a window of at least 3 on every axis, every
  pixel of a plateau is suppressed. ``skimage.feature.peak_local_max`` at its
  default ``min_distance=1``, ``scipy.ndimage.maximum_filter(x, size=k) == x``
  and ``cv2.dilate(x, kernel) == x`` keep every pixel of the plateau instead.
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

.. _rotation-conventions:

Rotations and rigid motions
---------------------------

:class:`~kornia.geometry.quaternion.Quaternion` multiplies by the Hamilton product, so ``(q1 * q2).matrix()`` is
``q1.matrix() @ q2.matrix()``: the right operand acts first. The Lie groups :class:`~kornia.geometry.liegroup.So3`,
:class:`~kornia.geometry.liegroup.Se3`, :class:`~kornia.geometry.liegroup.So2` and
:class:`~kornia.geometry.liegroup.Se2` compose the same way and act on a point as ``R p + t``, with ``R`` the
``matrix()`` of the rotation part. ``So2`` and ``Se2`` do so for any complex number, and a non-unit one also scales by
its modulus; ``So3`` and ``Se3`` need a unit quaternion
(`#4942 <https://github.com/kornia/kornia/issues/4942>`_). The tangent vectors of ``Se3`` and ``Se2`` put the
rotation part, in radians, last: ``[υ, ω]`` and ``[vx, vy, θ]``. ``log`` is principal: its rotation angle is at most
:math:`\pi` in magnitude. The Jacobians of ``So3`` satisfy
:math:`\exp(\omega + \delta) \approx \exp(\omega) \exp(J_r \delta) = \exp(J_l \delta) \exp(\omega)`. A transform
``trans_01`` maps frame-1 coordinates into frame 0, and
:func:`~kornia.geometry.linalg.relative_transformation` of ``trans_01`` and ``trans_02`` is ``trans_12``;
:class:`~kornia.geometry.pose.NamedPose` names the same transform ``dst_from_src``.

.. list-table::
   :header-rows: 1

   * - Topic
     - kornia
     - scipy
     - Sophus
     - Eigen
   * - quaternion storage
     - ``(w, x, y, z)``
     - ``Rotation.from_quat`` reads ``(x, y, z, w)`` unless ``scalar_first=True``
     - ``SO3::data()`` exposes Eigen's ``coeffs()`` order: ``(x, y, z, w)``
     - the ``Quaternion(w, x, y, z)`` constructor is scalar first, ``coeffs()`` is ``(x, y, z, w)``
   * - composition
     - ``a * b``, ``b`` acts first
     - ``r1 * r2``, the same
     - ``a * b``, the same
     - ``q1 * q2``, the same
   * - SE(3) tangent
     - ``[υ, ω]``, translation first
     - ``RigidTransform.as_exp_coords`` returns ``[ω, υ]``, rotation first
     - ``SE3::log`` returns ``[υ, ω]``, the same (GTSAM's ``Pose3`` is ``[ω, υ]``)
     - ``Isometry3d`` has no tangent or ``log``
   * - rotation ``log``
     - principal
     - ``as_rotvec``, principal
     - ``SO3::log``, principal
     - ``AngleAxis(q)``, angle in :math:`[0, \pi]`
   * - ``slerp``
     - the short arc
     - ``Slerp``, the short arc
     - ``interpolate``, the short arc
     - ``Quaternion::slerp``, the short arc
   * - frame naming
     - ``trans_01`` and ``dst_from_src`` map frame 1 (``src``) into frame 0 (``dst``)
     - ``tf_A_B`` maps ``B`` into ``A``
     - ``foo_T_bar`` maps ``bar`` into ``foo``
     - no frames

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
   volume by up to ``55.1`` there, against roundoff at ``align_corners=True``: exactly
   ``0`` in ``float32`` on torch 2.14 and about ``2e-6`` on torch 2.5.1. Pass
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

.. _color-conventions:

Color
-----

Color-space conversions use channel axis ``-3`` in ``(*, C, H, W)``. The color space determines
channel units; converted data is not generally a unit-range RGB image.

.. list-table:: Color channel units
   :header-rows: 1
   :widths: 22 42 36

   * - Space
     - Channels
     - Input encoding
   * - HSV and HLS
     - Hue in radians; saturation and value or lightness in unit range
     - Nonlinear RGB in unit range
   * - XYZ
     - X, Y, Z
     - Linear RGB; :func:`kornia.color.rgb_to_xyz` does not remove the sRGB transfer function
   * - Lab and Luv
     - L*, a*, b* or L*, u*, v*, with L* on the 0–100 scale
     - Nonlinear sRGB, linearized internally; D65 / 2° reference white
   * - YCbCr
     - Y, Cb, Cr; chroma is offset by 0.5
     - RGB in unit range
   * - YUV
     - Y, U, V; chroma is signed
     - RGB in unit range

Use :func:`kornia.color.rgb_to_linear_rgb` and :func:`kornia.color.linear_rgb_to_rgb` to change
transfer encoding. :func:`kornia.color.lab_to_rgb` clips its final RGB output unless ``clip=False``;
:func:`kornia.color.luv_to_rgb` does not clip its output. :func:`kornia.color.ycbcr_to_rgb`
clips its final RGB output to the unit range.

See the individual :doc:`color conversion pages </color.conversions>` for RAW mosaic layouts,
chroma subsampling, and known defects.

.. _enhancement-conventions:

Enhancement
-----------

- :func:`kornia.enhance.adjust_brightness` adds its factor, while
  :func:`kornia.enhance.adjust_brightness_accumulative` multiplies by it.
  :func:`kornia.enhance.adjust_contrast` multiplies pixel values; the mean-subtraction variant
  adjusts contrast around an image mean. torchvision and PIL brightness, contrast and saturation
  correspond to the ``_accumulative``, ``_with_mean_subtraction`` and ``_with_gray_subtraction``
  variants.
- :func:`kornia.enhance.adjust_hue` and :func:`kornia.enhance.adjust_hue_raw` take radians
  (torchvision's ``hue_factor`` is ``factor / (2 * pi)``); the raw hue and saturation helpers
  operate on HSV data.
- :func:`kornia.enhance.normalize` and :func:`kornia.enhance.denormalize` use channel axis 1
  in ``(B, C, ...)``. :func:`kornia.enhance.normalize_min_max` takes ``(*, C, H, W)`` and rescales
  each ``H x W`` plane independently. Outside rank 4, ``denormalize`` checks ``(C,)`` statistics
  against the wrong axis; pass ``(1, C)``
  (`#5318 <https://github.com/kornia/kornia/issues/5318>`_).
- :func:`kornia.enhance.integral_image` sums inclusively over the last two axes. The returned
  image has the input shape, without an extra zero border.
- :class:`kornia.enhance.ZCAWhitening` uses ``dim`` as the sample axis and flattens all other
  axes into features. ``unbiased=True`` selects the ``N - 1`` covariance denominator.

See the :doc:`enhancement API </enhance>` for clipping, histogram ranges, and known defects.

Augmentations
-------------

- Use one :class:`kornia.augmentation.container.AugmentationSequential` call
  for an image and its annotations. Geometric children that implement a data
  key's handler share their recorded parameters across the inputs; a transform
  matrix alone does not mean every data key is supported.
- ``.inverse()`` restores keypoints to numerical precision when every
  geometric step is invertible. Slice-mode crops and the 3D geometric
  augmentations raise (crops accept ``cropping_mode="resample"`` for
  inversion); intensity and non-rigid children are skipped, so their effect
  stays. Tensor box outputs are axis-aligned enclosures, so a rotation round
  trip does not restore a box.
- Non-rigid children do not move coordinate annotations:
  :class:`kornia.augmentation.RandomElasticTransform` warps masks but leaves
  keypoints and boxes unchanged, and
  :class:`kornia.augmentation.RandomThinPlateSpline` and
  :class:`kornia.augmentation.RandomFisheye` leave coordinates unchanged and
  raise on masks (`#4420 <https://github.com/kornia/kornia/issues/4420>`_).
- Intensity augmentations leave masks unchanged, except
  :class:`kornia.augmentation.RandomErasing`, which erases the same rectangle
  in the mask and fills it with ``0`` (background) whatever ``value`` fills
  the image.
- Geometric children normally resample masks with nearest interpolation, so
  labels are not blended, but padding can introduce its fill value. The container
  processes a mask in the dtype of the image it is working on (the most recent
  image argument before the mask, or the call's first image when the mask
  comes first) and returns it in the mask's own dtype, so
  integer labels outside that dtype's exact range change even through a flip:
  ``2049`` becomes ``2048`` in ``float16``
  (`#4478 <https://github.com/kornia/kornia/issues/4478>`_). A direct
  geometric ``transform_masks`` call requires a floating tensor.
- A list of masks is not a substitute for separate full-batch tensors: list
  index ``i`` selects sample ``i``'s gate, so full-batch entries desynchronize
  under mixed gates and per-sample entries can fail in warps or reuse the wrong
  crop window. Pass separate ``mask`` keys instead
  (`#4477 <https://github.com/kornia/kornia/issues/4477>`_).
- Boxes use the inclusive ``xyxy_plus`` convention of
  :class:`kornia.geometry.boxes.Boxes` (see *Bounding boxes* above). Flips use
  integer pixel centres: ``x' = W - 1 - x``.
- The outer ``.transform_matrix`` of a nested container can be missing or
  stale, even with only rigid children
  (`#4476 <https://github.com/kornia/kornia/issues/4476>`_).
- Dictionary keys match a data-key name exactly or before an ``_``/``-``
  suffix, the longest match winning. Unrecognized keys are returned unchanged
  as metadata, and the caller's dictionary is not modified.
- A positive ``degrees`` turns the displayed image counter-clockwise with
  :class:`kornia.augmentation.RandomRotation`, matching
  :func:`kornia.geometry.transform.rotate`, and clockwise with
  :class:`kornia.augmentation.RandomAffine`
  (`#4408 <https://github.com/kornia/kornia/issues/4408>`_).
- The 2D intensity augmentations assume float input in ``[0, 1]``, and no
  base-class check enforces it. Outside that range each class follows its own
  policy -- some clamp, rescale or round-trip through ``uint8``, some do not
  clamp, :class:`kornia.augmentation.RandomPlanckianJitter` clamps only the
  upper end, and :class:`kornia.augmentation.RandomGamma` does not clamp, so a
  negative input gives NaN for a non-integer ``gamma``. Several classes return
  an all-zero image for an all-negative input, and
  :class:`kornia.augmentation.RandomSolarize` does so when every value is at
  least ``1.5``. :class:`kornia.augmentation.RandomEqualize` and
  :class:`kornia.augmentation.RandomClahe` raise an error naming the range; the
  check is asynchronous, so on an accelerator the error can surface at a later
  synchronizing call (`#4430 <https://github.com/kornia/kornia/issues/4430>`_).
  See :class:`kornia.augmentation.IntensityAugmentationBase2D` and each class's
  documentation.

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

- Parameters are drawn from torch's global generator, on the CPU by default
  whatever the image device, so ``torch.manual_seed`` reproduces a run for the
  same configuration, inputs, device and dtype. There is no per-instance
  ``generator=``: the standard 2D forward silently accepts and ignores unknown
  keywords, ``generator=`` included, while mix forwards reject them
  (`#4427 <https://github.com/kornia/kornia/issues/4427>`_). Under a
  :class:`torch.utils.data.DataLoader`, a ``worker_init_fn`` that reseeds
  every worker to the same value duplicates the augmentations across workers
  (see `Randomness in multi-process data loading
  <https://pytorch.org/docs/stable/notes/randomness.html#dataloader>`_).
- ``.to(...)`` and ``set_rng_device_and_dtype`` move the gate and ask the
  parameter generators to rebuild their samplers, but some generators keep
  CPU tensors or ignore the requested precision, and returned parameters need
  not follow: numeric-range ``RandomAffine`` returns CPU ``float32``
  parameters while ``batch_prob`` is on the accelerator. Do not infer where
  draws ran from ``_params``
  (`#4426 <https://github.com/kornia/kornia/issues/4426>`_).
- To replay a transform, pass its recorded ``_params`` as ``params=``. The
  standard 2D forward uses that dictionary as given, without a copy, and fills
  a missing ``batch_prob`` in place with an all-true gate; mix augmentations
  require the key. The recorded parameters capture every draw except the VAE
  latent that :class:`kornia.augmentation.RandomDissolving` samples while it is
  applied.
- A kornia release can change how many draws an augmentation consumes, so a
  seeded pipeline is not bit-stable across kornia versions.
- ``same_on_batch=True`` shares the per-sample gate and the sampled factors
  across the batch; it does not make mix pairing indices equal. The colour
  order of ``ColorJiggle`` and ``ColorJitter`` is shared across the batch
  regardless. On ``AugmentationSequential``, ``same_on_batch=None`` keeps each
  child's setting, and ``True`` or ``False`` overwrites it.
- Whether a constructor's ``p`` sets the per-sample or the whole-batch gate,
  and whether it exposes ``p_batch``, depends on the concrete class
  (`#4425 <https://github.com/kornia/kornia/issues/4425>`_). The gate selects
  after the transform has run on the whole batch, so a skipped sample can still
  raise or carry a NaN gradient
  (`#4576 <https://github.com/kornia/kornia/issues/4576>`_).

Serializing an augmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``state_dict()`` is not a complete record of an augmentation's
  configuration: rebuild the augmentation from its constructor arguments.
  Numeric ranges create no trainable parameters, and with numeric ranges the
  3D and mix augmentations have an empty ``state_dict()``. ``nn.Parameter``
  ranges (``RandomRotation`` and ``RandomAffine`` among others) are registered
  and receive gradients. Some numeric ranges are copied into
  ``_param_generator.*`` buffers that loading does not feed back into the
  samplers (`#4428 <https://github.com/kornia/kornia/issues/4428>`_).
- ``pickle`` and ``copy.deepcopy`` keep the configuration and the recorded
  ``_params``, which replay on the same input. The default
  ``kornia.augmentation.auto`` policies cannot be pickled (a policy can be
  pickled only when every operation wrapper in it can), though they deep-copy
  (`#4469 <https://github.com/kornia/kornia/issues/4469>`_). What lazily
  computed matrices retain is described in :doc:`/augmentation.base`.

Morphology
----------

:func:`kornia.morphology.dilation` reflects the kernel and :func:`kornia.morphology.erosion` does not; ``origin``
is a ``[row, col]`` index and ``border_type`` takes torch's pad names. Below, ``K`` is a kernel of shape
:math:`(k_h, k_w)`, and each call row assumes the matching border from the border rows:

.. list-table::
   :header-rows: 1

   * - Behaviour
     - kornia
     - ``scipy.ndimage``
     - scikit-image
     - OpenCV
   * - ``dilation`` reflects the kernel
     - yes
     - yes
     - no
     - no
   * - kornia call equal to their dilation by ``K``
     - (reference)
     - ``dilation(x, K)``
     - ``dilation(x, K.flip((0, 1)))``
     - ``dilation(x, K.flip((0, 1)), origin=[k_h - 1 - a_y, k_w - 1 - a_x])`` for ``anchor=(a_x, a_y)``; at the
       default anchor the flip alone is enough only for an odd-sized ``K``
   * - ``dilation`` by ``ones(2, 2)`` of a hot pixel at ``(2, 3)``
     - rows 1-2, cols 2-3
     - rows 1-2, cols 2-3
     - rows 1-2, cols 2-3
     - rows 2-3, cols 3-4
   * - kornia call equal to their erosion by ``K``
     - (reference)
     - ``erosion(x, K)``
     - ``erosion(x, K, origin=[(k_h - 1) // 2, (k_w - 1) // 2])``, which differs from the default only for an
       even size
     - ``erosion(x, K, origin=[a_y, a_x])``; the default anchor is the default origin
   * - kornia call equal to their opening / closing by ``K``
     - (reference)
     - ``opening(x, K)`` / ``closing(x, K)``, except under ``geodesic``: one ``cval`` serves both passes, so
       ``grey_opening`` has no ignore mode
     - ``opening(x, K, origin=[(k_h - 1) // 2, (k_w - 1) // 2])`` / ``closing(x, K.flip((0, 1)))``, and likewise
       ``white_tophat`` / ``black_tophat`` for ``top_hat`` / ``bottom_hat``
     - ``MORPH_OPEN`` / ``MORPH_CLOSE`` agree only for a ``K`` symmetric about its anchor
   * - origin/anchor semantics
     - ``[row, col]`` index, default ``[k_h // 2, k_w // 2]`` for even sizes too
     - offset from ``k // 2``
     - not exposed
     - ``(x, y)`` index
   * - default border
     - ``geodesic`` (ignore outside)
     - ``reflect``, which repeats the edge sample (**not** torch's ``reflect``)
     - ``reflect``, the same rule as scipy's
     - ignore outside
   * - kornia's ``geodesic``
     - ``geodesic``
     - ``mode="constant"`` with ``cval=-np.inf`` (dilation) or ``np.inf`` (erosion)
     - ``mode="ignore"``
     - default border
   * - torch's ``reflect``
     - ``reflect``
     - ``mirror``
     - ``mirror``
     - ``BORDER_REFLECT_101``
   * - torch's ``replicate``
     - ``replicate``
     - ``nearest``
     - ``nearest``
     - ``BORDER_REPLICATE``
   * - torch's ``circular``
     - ``circular``
     - ``wrap``
     - ``wrap``
     - no equivalent: ``BORDER_WRAP`` raises, except on uint8, where the result is not a wrap

The equivalences cover empty ``geodesic`` windows too: scipy, scikit-image and kornia all return ``-inf`` from
such a window in a dilation and ``+inf`` in an erosion, whatever the data range.

.. _filtering-conventions:

Filtering
---------

:doc:`kornia.filters </filters>` follows torch's vocabulary: its kernels are correlated by default, ``border_type``
takes the :func:`torch.nn.functional.pad` mode names, and an even kernel is anchored where
``F.conv2d(padding='same')`` anchors it. The correlation matches ``cv2.filter2D`` and ``scipy.ndimage.correlate``; the
border names and the even-kernel anchor do not.

- :func:`~kornia.filters.filter2d`, :func:`~kornia.filters.filter3d` and :func:`~kornia.filters.fft_conv`
  **correlate** by default, as ``cv2.filter2D`` and ``scipy.ndimage.correlate`` do; ``behaviour='conv'`` flips the
  kernel first, as ``scipy.ndimage.convolve`` does. :func:`~kornia.filters.correlate2d` and
  :func:`~kornia.filters.convolve2d` are :func:`~kornia.filters.filter2d` with ``behaviour`` fixed to ``'corr'`` and
  ``'conv'``; :func:`~kornia.filters.correlate3d` and :func:`~kornia.filters.convolve3d` are the same for
  :func:`~kornia.filters.filter3d`.
- :func:`~kornia.filters.filter2d_separable` takes ``kernel_x`` (along ``W``) before ``kernel_y``, the order of
  ``cv2.sepFilter2D(src, ddepth, kernelX, kernelY)``.

The border modes, shown padding ``a b c d`` by two samples on each side. :doc:`kornia.morphology </morphology>` uses
the same names for its non-``geodesic`` borders:

.. list-table::
   :header-rows: 1

   * - kornia ``border_type`` (the ``F.pad`` mode)
     - pads ``a b c d`` to
     - ``scipy.ndimage`` ``mode``
     - OpenCV ``borderType``
   * - ``reflect``, the default of ``filter2d``, ``filter2d_separable`` and ``fft_conv``
     - ``c b | a b c d | c b``
     - ``mirror``
     - ``BORDER_REFLECT_101``, the ``cv2.filter2D`` default
   * - ``replicate``, the default of ``filter3d``
     - ``a a | a b c d | d d``
     - ``nearest``
     - ``BORDER_REPLICATE``
   * - ``circular``
     - ``c d | a b c d | a b``
     - ``wrap``
     - ``BORDER_WRAP``, which ``cv2.filter2D`` rejects for small kernels
   * - ``constant``
     - ``0 0 | a b c d | 0 0``
     - ``constant`` with ``cval=0``
     - ``BORDER_CONSTANT`` with value 0
   * - no kornia mode
     - ``b a | a b c d | d c``
     - ``reflect``, the scipy default
     - ``BORDER_REFLECT``

An even kernel has no centre tap. The filters anchor a kernel ``k`` taps long at ``(k - 1) // 2``, the anchor of
``F.conv2d(padding='same')``. OpenCV and scipy anchor it at ``k // 2`` by default, and so does the default ``origin`` of
:func:`~kornia.morphology.erosion` (see `Morphology`_), so along every even axis a correlated output sits one pixel
before theirs; ``behaviour='conv'`` needs no shift (below). An odd kernel is anchored at its centre by all of them.
For a kernel ``K`` of shape ``(kh, kw)``, with the border mapped by the table above:

- ``filter2d(x, K[None])`` equals ``scipy.ndimage.correlate(x, K, origin=o)`` with ``o = -1`` on each even axis and
  ``0`` on each odd one, and ``cv2.filter2D(x, -1, K, anchor=((kw - 1) // 2, (kh - 1) // 2))``.
- ``filter2d(x, K[None], behaviour='conv')`` equals ``scipy.ndimage.convolve(x, K)`` at scipy's default origin, even
  for an even ``K``.

The kernel builders against their references:

- :func:`~kornia.filters.get_gaussian_kernel1d` samples the Gaussian: for an odd ``k`` and ``sigma > 0`` it equals
  ``cv2.getGaussianKernel(k, sigma)`` and the weights ``scipy.ndimage.gaussian_filter1d`` applies with
  ``radius=k // 2``. :func:`~kornia.filters.get_gaussian_erf_kernel1d` is the pixel-integrated Gaussian and
  :func:`~kornia.filters.get_gaussian_discrete_kernel1d` is Lindeberg's discrete Gaussian,
  ``scipy.special.ive(abs(n), sigma**2)`` normalized.
- :func:`~kornia.filters.get_hanning_kernel1d` is the symmetric window of ``numpy.hanning(k)``,
  ``scipy.signal.windows.hann(k)`` and ``torch.hann_window(k, periodic=False)``; torch's default periodic window
  differs.
- :func:`~kornia.filters.get_spatial_gradient_kernel2d` with ``('sobel', 1)`` stacks the outer products of OpenCV's
  ``getDerivKernels(1, 0, 3)`` and ``getDerivKernels(0, 1, 3)``, and answers 8 to a unit slope, as ``cv2.Sobel`` and
  ``scipy.ndimage.sobel`` do; scikit-image's ``sobel_h`` and ``sobel_v`` answer 2.
- :func:`~kornia.filters.get_laplacian_kernel2d` of size 3 is the 8-neighbour stencil and estimates :math:`3 \nabla^2`,
  where ``scipy.ndimage.laplace`` and ``cv2.Laplacian(ksize=1)`` return :math:`\nabla^2`, ``cv2.Laplacian(ksize=3)``
  :math:`4 \nabla^2` and ``skimage.filters.laplace`` :math:`-\nabla^2`.

The blurs give sizes and standard deviations rows first, as torch orders ``(H, W)``: ``kernel_size`` is
``(kh, kw)`` and ``sigma`` is :math:`(\sigma_y, \sigma_x)` in :func:`~kornia.filters.gaussian_blur2d`,
:func:`~kornia.filters.box_blur`, :func:`~kornia.filters.median_blur` and the filters built on them. OpenCV passes
both pairs x first, so swap them when porting; scipy uses kornia's order. For odd ``kh`` and ``kw``, with the border
mapped by the table above (OpenCV accepts ``BORDER_WRAP`` in these calls only for some dtypes and sizes):

- ``gaussian_blur2d(x, (kh, kw), (sy, sx))`` equals ``cv2.GaussianBlur(x, (kw, kh), sigmaX=sx, sigmaY=sy)`` and
  ``scipy.ndimage.gaussian_filter(x, sigma=(sy, sx), radius=(kh // 2, kw // 2))``.
- ``box_blur(x, (kh, kw))`` equals ``cv2.blur(x, (kw, kh))`` and ``scipy.ndimage.uniform_filter(x, size=(kh, kw))``.

The edge-preserving filters, the sharpening and the blur pools against their references:

- :func:`~kornia.filters.bilateral_blur` with ``'l1'`` uses the colour distance of ``cv2.bilateralFilter``, but
  weighs the whole ``kernel_size`` rectangle where OpenCV weighs only the disc of radius ``d // 2``, so the two differ
  even for ``kernel_size=(d, d)``.
- ``guided_blur(guide, src, 2 * r + 1, eps)`` equals ``cv2.ximgproc.guidedFilter(guide, src, r, eps)`` away from the
  border: ``eps`` is the same quantity, but OpenCV pads with ``BORDER_REFLECT``, which has no kornia mode.
  :func:`~kornia.filters.joint_bilateral_blur` takes its guide second, where
  ``cv2.ximgproc.jointBilateralFilter(joint, src, ...)`` takes it first.
- ``unsharp_mask(x, (k, k), (r, r))`` with ``k = 2 * int(4 * r + 0.5) + 1``, the window scikit-image truncates its
  Gaussian to, equals ``skimage.filters.unsharp_mask(x, radius=r, amount=1, preserve_range=True)`` away from the
  border. scikit-image pads with scipy's ``reflect``, which has no kornia mode, and with its default
  ``preserve_range=False`` it clips the result, which :func:`~kornia.filters.unsharp_mask` never does.
- ``blur_pool2d(x, k, s)`` equals the antialiased-cnns ``BlurPool(channels, pad_type='zero', filt_size=k, stride=s)``,
  which defines ``filt_size`` up to 7. The reference's default ``pad_type='reflect'`` keeps a constant map constant
  where :func:`~kornia.filters.blur_pool2d` zero-pads and darkens its border.
- ``blur_pool2d(x, 5, 2)`` blurs with the 5 x 5 binomial kernel of :func:`~kornia.geometry.transform.pyrdown` but
  samples differently: it zero-pads and keeps every second pixel from index 0, :math:`\lceil H / 2 \rceil` rows,
  where ``pyrdown`` reflects the border and interpolates between pixels, :math:`\lfloor H / 2 \rfloor` rows.

.. _two-view-conventions:

Two-view geometry
-----------------

The two-view estimators take the first image's points first and follow OpenCV's order:
:func:`~kornia.geometry.epipolar.find_fundamental` returns an ``F`` with :math:`x_2^\top F x_1 = 0` in image
coordinates and :func:`~kornia.geometry.epipolar.find_essential` an ``E`` with :math:`x_2^\top E x_1 = 0` in
normalised camera coordinates, and :func:`~kornia.geometry.homography.find_homography_dlt` and
:class:`~kornia.geometry.ransac.RANSAC` with ``model_type="homography"`` return an ``H`` that maps ``points1`` to
``points2``. Extrinsics are world-to-camera, as in :doc:`/get-started/camera-conventions`. Side by side:

.. list-table::
   :header-rows: 1

   * - Topic
     - kornia
     - OpenCV
   * - fundamental matrix
     - ``find_fundamental(points1, points2)``, scaled to ``F[2, 2] = 1`` unless that entry is numerically zero;
       ``method="7POINT"`` returns three candidates ``(B, 3, 3, 3)`` in no particular order, padded with zero
       matrices when the cubic has one real root
     - ``findFundamentalMat(points1, points2)``, the same ``F``; ``FM_7POINT`` stacks only the real solutions as
       ``(3k, 3)``
   * - essential matrix
     - ``find_essential`` takes normalised camera coordinates :math:`K^{-1} [u, v, 1]^\top` and returns ten
       slots, ``NaN`` for complex roots (all ten for a sample with no real solution)
     - ``findEssentialMat`` takes pixels and ``cameraMatrix`` (or one matrix per camera); with exactly 5 points it
       stacks the real solutions as ``(3k, 3)``, with more it returns the single ``E`` its RANSAC or LMedS selects
   * - homography
     - ``find_homography_dlt(points1, points2)`` maps ``points1`` to ``points2``
     - ``findHomography(src, dst)``, the same direction
   * - pose from ``E``
     - ``decompose_essential_matrix`` returns ``R1``, ``R2`` and a unit ``t``; which candidate is the true pose is
       not fixed.
       ``motion_from_essential_choose_solution`` selects it by cheirality from pixel coordinates, and returns
       candidate 0 when no point passes (`#4879 <https://github.com/kornia/kornia/issues/4879>`_)
     - ``decomposeEssentialMat`` returns the same candidate set, whose labels are not fixed either and differ from
       kornia's, so a candidate index does not port; ``recoverPose`` selects the same pose and also returns the
       inlier count
   * - projection matrix
     - ``KRt_from_projection`` returns the translation ``t`` of ``P = K [R | t]``; ``P`` and ``-P`` give the same
       positive-diagonal ``K``, rotation ``R`` and ``t``
     - ``decomposeProjectionMatrix`` returns the homogeneous camera centre :math:`C = -R^\top t`; for
       ``det P[:, :3] < 0`` it keeps ``det R = 1`` and returns ``K[2, 2] < 0``
   * - triangulation
     - ``triangulate_points`` returns Euclidean points ``(*, N, 3)``
     - ``triangulatePoints`` returns homogeneous points ``(4, N)``
   * - epipolar lines
     - ``compute_correspond_epilines(x1, F)`` for first-image points; pass ``F.transpose(-2, -1)`` for
       second-image points
     - ``computeCorrespondEpilines(x1, 1, F)``; ``whichImage=2`` for second-image points
   * - Sampson distance
     - ``sampson_epipolar_distance(pts1, pts2, F)``, squared by default
     - ``sampsonDistance(pt1, pt2, F)``, the same argument order
   * - RANSAC threshold
     - ``inl_th`` is a point distance in the keypoints' units, calibrated units for ``model_type="essential"``;
       for line segments, the mean distance of the transferred endpoints from the target segment's line, which
       local optimization re-weights by as well
     - ``ransacReprojThreshold`` of ``findHomography``, the same unit for points
   * - polynomial roots
     - ``solve_quadratic``, ``solve_cubic`` and ``solve_quartic`` take coefficients highest degree first and
       return only the real roots, a missing root padded with ``0.0``; ``solve_quartic``'s order is unspecified;
       a zero leading coefficient lowers the degree
     - ``numpy.roots`` takes the same coefficient order, returns the complex roots too and drops leading zeros

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
16. Expecting ``RandomElasticTransform``, ``RandomThinPlateSpline`` or
    ``RandomFisheye`` to move boxes and keypoints with the image — they do not.
17. Feeding mean/std-normalized or otherwise out-of-``[0, 1]`` tensors
    through an intensity augmentation — some clamp, some rescale,
    ``RandomEqualize`` and ``RandomClahe`` raise, and several return zeros
    (`#4430 <https://github.com/kornia/kornia/issues/4430>`_).

.. tip::

   Machine-readable copies of these conventions live at
   `llms.txt <https://kornia.readthedocs.io/en/latest/llms.txt>`_ and
   `llms-full.txt <https://kornia.readthedocs.io/en/latest/llms-full.txt>`_
   at the docs root.
