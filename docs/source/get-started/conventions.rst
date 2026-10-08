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
its modulus; ``So3`` and ``Se3`` rotate by the direction :math:`q / |q|` of a non-unit quaternion, without scaling,
like ``Quaternion.matrix()``. The tangent vectors of ``Se3`` and ``Se2`` put the
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
  each ``H x W`` plane independently.
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
- Nested ``AugmentationSequential`` children contribute matrices from the
  current call. A plain ``ImageSequential`` child is still omitted from an
  outer ``AugmentationSequential`` matrix
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

The derivative filters apply those kernels at different scales, and :func:`~kornia.filters.canny` thresholds the raw
one:

- :func:`~kornia.filters.spatial_gradient` and :func:`~kornia.filters.sobel` use normalized first-order gradients
  by default: an interior axis-aligned unit slope gives 1 in the corresponding ``spatial_gradient`` channel, and
  ``sobel`` returns :math:`\sqrt{1 + \epsilon}`. ``normalized=False`` returns the raw Sobel response above.
  :func:`~kornia.filters.spatial_gradient` always replicates the border, so
  ``cv2.Sobel(x, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REPLICATE)`` and
  ``scipy.ndimage.sobel(x, axis=-1, mode='nearest')`` equal channel 0 of ``spatial_gradient(x, normalized=False)``;
  OpenCV's default ``BORDER_REFLECT_101`` differs on the outermost rows and columns.
  ``skimage.filters.sobel(x, mode='nearest')`` equals ``sqrt(2) * sobel(x, eps=0)``.
- For floating inputs, :func:`~kornia.filters.laplacian` is normalized by default too, but by the stencil's absolute
  sum, 16 for size 3: ``laplacian(x, 3)`` estimates :math:`3 \nabla^2 / 16`, and ``normalized=False`` the
  :math:`3 \nabla^2` above. See its Convention block for the integer-input defect.
- :func:`~kornia.filters.canny` compares its thresholds with the **unnormalized** Sobel magnitude of the blurred
  image, about eight times what :func:`~kornia.filters.sobel` returns by default, up to the ``eps`` inside the square
  root. For a **single-channel grayscale** uint8 image ``img``, ``cv2.Canny(img, t1, t2, L2gradient=True)`` corresponds
  to ``canny(x, t1 / 255, t2 / 255, kernel_size=1, eps=0)`` for edge-map comparison, with ``x`` the float32 or float64
  tensor ``img / 255``: the thresholds scale with the image and ``kernel_size=1`` skips the Gaussian blur that
  OpenCV does not apply. OpenCV's default L1 magnitude :math:`|g_x| + |g_y|` has no kornia counterpart.
  :func:`~kornia.filters.canny` resolves ties along the gradient as OpenCV does, but floating-point rounding of
  ``img / 255``, the gradient and threshold comparisons can still change a near-tie or near-threshold decision.
  With the default ``eps=1e-6``, the square root raises the magnitude above its ``eps=0`` value, when retained by
  the dtype's precision: it can promote a pixel even when the threshold is slightly above the raw magnitude,
  without an exact tie. Color images need separate preprocessing: Kornia converts RGB to grayscale, while
  OpenCV's color Canny selects the channel with the strongest gradient, so this correspondence does not apply
  directly to a 3-channel image.
- ``skimage.feature.canny`` thresholds the same unnormalized magnitude of a floating-point image and has the same
  defaults, 0.1 and 0.2, so thresholds carry over; its Gaussian blur and its interpolating suppression differ from
  :func:`~kornia.filters.canny`'s, so the edge maps do not match.

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

.. _losses-metrics-conventions:

Losses and metrics
------------------

:doc:`kornia.losses </losses>` and :doc:`kornia.metrics </metrics>` take the prediction first and the target second,
as torch's losses do. The order does not change the value of the symmetric functions, among them SSIM, MS-SSIM, PSNR,
the robust losses and the Jensen-Shannon divergence, but it does for others: :func:`~kornia.losses.kl_div_loss_2d` takes
``pred`` first and returns :math:`\mathrm{KL}(\text{target} \,\|\, \text{pred})`, and
:func:`~kornia.losses.inverse_depth_smoothness_loss` takes the inverse depth it penalises first and the image that
weights it second.

What the families take:

- :func:`~kornia.metrics.ssim`, :func:`~kornia.metrics.ssim3d`, :func:`~kornia.metrics.psnr`, the SSIM losses and
  :class:`~kornia.losses.MS_SSIMLoss` compare images with values in ``[0, L]``, where ``L`` is the data range:
  ``max_val`` everywhere except in :class:`~kornia.losses.MS_SSIMLoss`, which calls it ``data_range`` as scikit-image,
  pytorch-msssim and torchmetrics do. ``L`` sets the SSIM constants :math:`C_1 = (0.01 L)^2` and
  :math:`C_2 = (0.03 L)^2` and the PSNR peak :math:`\text{MAX}_I`. Pixel values are never rescaled, so images in
  ``[0, 255]`` need ``max_val=255.0`` or ``data_range=255.0``. The default is ``1.0`` (``psnr``, ``psnr_loss`` and
  ``PSNRLoss`` have none), where pytorch-msssim defaults to 255.
- The robust losses take two tensors of the same, arbitrary shape; the residual is in the units of the data.
- :func:`~kornia.losses.total_variation` takes one image ``(*, H, W)``.
- :func:`~kornia.losses.kl_div_loss_2d` and :func:`~kornia.losses.js_div_loss_2d` take probabilities: every
  ``(b, n)`` slice of a ``(B, N, H, W)`` input is a distribution over ``H x W``.
- The losses on the segmentation page take raw logits (see `Dense-prediction losses`_) and
  :class:`~kornia.losses.HausdorffERLoss` takes probabilities; the mutual-information losses take intensities of any
  range, which they normalise per signal: per channel of each sample for an image batch.

Where a function takes ``reduction``, the names are torch's: ``'none'``, ``'mean'`` and ``'sum'``. What is averaged or
added and the default differ between families, so each Convention block states both; the table lists the defaults. A
string outside a function's vocabulary raises, ``BaseError`` from the robust losses and
:func:`~kornia.losses.total_variation` and ``NotImplementedError`` from the others.

.. list-table::
   :header-rows: 1

   * - functions
     - default ``reduction``
     - default output
   * - :func:`~kornia.losses.charbonnier_loss`, :func:`~kornia.losses.cauchy_loss`,
       :func:`~kornia.losses.geman_mcclure_loss`, :func:`~kornia.losses.welsch_loss`,
       :func:`~kornia.losses.focal_loss`, :func:`~kornia.losses.binary_focal_loss_with_logits`
     - ``'none'``
     - one value per element, in the shape of the prediction
   * - :func:`~kornia.losses.ssim_loss`, :func:`~kornia.losses.ssim3d_loss`, :class:`~kornia.losses.MS_SSIMLoss`,
       :func:`~kornia.losses.kl_div_loss_2d`, :func:`~kornia.losses.js_div_loss_2d`,
       :class:`~kornia.losses.HausdorffERLoss`, :class:`~kornia.losses.HausdorffERLoss3D`,
       :func:`~kornia.metrics.aepe`, :func:`~kornia.metrics.mean_absolute_disparity_error` and the other disparity
       metrics
     - ``'mean'``
     - a scalar
   * - :func:`~kornia.losses.total_variation`
     - ``'sum'``, with ``'mean'`` the only alternative
     - one value per leading index, ``(*,)``: ``(B, C)`` for an image batch
   * - :func:`~kornia.losses.dice_loss`, :func:`~kornia.losses.tversky_loss`, :func:`~kornia.losses.lovasz_hinge_loss`,
       :func:`~kornia.losses.lovasz_softmax_loss`, :func:`~kornia.metrics.psnr`, :func:`~kornia.losses.psnr_loss`,
       :func:`~kornia.losses.inverse_depth_smoothness_loss`
     - no ``reduction``
     - a scalar
   * - ``mutual_information_loss`` and the other mutual-information losses
     - no ``reduction``
     - one value per signal: ``(B, C)`` for an image batch in the 2-D and 3-D variants
   * - :func:`~kornia.metrics.ssim`, :func:`~kornia.metrics.ssim3d` and the other metrics
     - no ``reduction``
     - stated by each function

The modules take the ``reduction`` of their functions, except :class:`~kornia.losses.TotalVariation`, which has none
and always sums.

Porting from other libraries:

- ``ssim(x, y, 11, max_val=L, eps=0.0, padding='valid').mean()`` for one image matches scikit-image's
  ``structural_similarity(x, y, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=L,
  channel_axis=-1)`` on the ``(H, W, C)`` arrays up to floating-point differences, and over a batch it matches
  pytorch-msssim's ``ssim(x, y, data_range=L)``. Set ``eps=0.0`` to remove Kornia's added denominator term when
  matching these implementations. The default ``eps=1e-12`` can matter: identical black images score about
  ``0.99998889`` at ``L=1.0``, ``0.9`` at ``L=0.1`` and ``0.00089919`` at ``L=0.01``, where the references give 1.
  scikit-image's defaults, a 7 x 7 uniform window with the sample covariance, give another value. torchmetrics'
  ``structural_similarity_index_measure(x, y, data_range=L)`` averages over a reflected ``'same'`` map instead;
  it has no added denominator epsilon and clamps negative variance estimates from roundoff to zero.
- :func:`~kornia.losses.ssim_loss` is the structural dissimilarity ``(1 - SSIM) / 2``, clamped to ``[0, 1]``. The
  ``1 - SSIM`` loss is twice that: ``1 - ssim(x, y, w).mean()`` equals ``2 * ssim_loss(x, y, w)`` wherever the clamp
  does not act. :func:`~kornia.losses.ssim3d_loss` uses the same clamped DSSIM formula for volumes.
- :class:`~kornia.losses.MS_SSIMLoss` filters with one Gaussian per entry of ``sigmas`` at full resolution, the
  approximation of Zhao et al.; pytorch-msssim's ``ms_ssim`` and torchmetrics'
  ``multiscale_structural_similarity_index_measure`` downsample through a dyadic pyramid, so ``1 - ms_ssim(x, y)`` is
  not ``MS_SSIMLoss(alpha=1.0, compensation=1.0)(x, y)``.
- :func:`~kornia.metrics.psnr` of a batch pools one MSE over all its images: it equals scikit-image's
  ``peak_signal_noise_ratio(x, y, data_range=L)`` on the whole batch array and torchmetrics'
  ``peak_signal_noise_ratio(x, y, data_range=L)`` at its default ``dim=None``. The mean of per-image PSNRs is
  scikit-image's value averaged over the images, or torchmetrics' with ``dim=(1, 2, 3)``.
- ``general.lossfun(x - y, alpha, scale)`` of Barron's ``robust_loss_pytorch`` equals the kornia robust loss of
  ``x / scale`` and ``y / scale`` with the same ``alpha``: 1 for :func:`~kornia.losses.charbonnier_loss`, 0 for
  :func:`~kornia.losses.cauchy_loss`, -2 for :func:`~kornia.losses.geman_mcclure_loss` and ``-inf`` for
  :func:`~kornia.losses.welsch_loss`.
- torchmetrics' ``total_variation(img, reduction='none')`` equals ``total_variation(img).sum(-1)`` for a
  ``(B, C, H, W)`` batch. Its ``'mean'`` averages those per-image sums over the batch, where the ``'mean'`` of
  :func:`~kornia.losses.total_variation` averages each difference term over its own count.
- For strictly positive probabilities, ``kl_div_loss_2d(pred, target)`` equals
  ``F.kl_div(pred.log(), target, reduction='batchmean')`` on the ``(B * N, H * W)`` reshape; torch's
  ``reduction='mean'`` divides by every element instead. Kornia defines zero-target cells with finite nonnegative
  predictions as contributing zero, including shared zero cells, where the raw PyTorch recipe returns NaN. Replace
  both arguments with 1 in these cells before taking the logarithm to reproduce Kornia's handling. torchmetrics'
  ``kl_divergence(target, pred)`` on the reshape matches Kornia's value, including shared zeros. For scipy, reshape
  ``pred`` and ``target`` to NumPy arrays ``p`` and ``q`` of shape ``(B, N, H * W)``. Then
  ``scipy.stats.entropy(q, p, axis=-1).mean()`` matches the default KL loss, and
  ``(scipy.spatial.distance.jensenshannon(p, q, axis=-1) ** 2).mean()`` matches
  :func:`~kornia.losses.js_div_loss_2d`, using scipy's default natural logarithm. Without the final ``mean()``, each
  returns ``(B, N)``, matching ``reduction='none'``. SciPy normalises its inputs, so these mappings require each
  spatial slice to sum to one.
- :func:`~kornia.losses.inverse_depth_smoothness_loss` does not normalise the inverse depth, where Monodepth2 divides
  the disparity by its mean plus ``1e-7`` before its smoothness term: pass
  ``idepth / (idepth.mean((2, 3), keepdim=True) + 1e-7)`` to port it, keeping zero inverse depth finite.

Dense-prediction losses
^^^^^^^^^^^^^^^^^^^^^^^

The losses on the :doc:`segmentation page </losses.segmentation>` take raw logits, not probabilities;
:func:`~kornia.losses.focal_loss` documents the multi-class input, with the classes on axis 1 and an integer label map
as the target.

**Class 0 is the background** wherever a loss or a metric singles one class out:
:func:`~kornia.losses.focal_loss` weights class 0 by ``1 - alpha`` and classes ``1`` to ``C - 1`` by ``alpha``, and
:func:`~kornia.metrics.mean_average_precision` never scores class 0, which its ``n_classes`` still counts. Keep the
background at label 0 when porting a label map: moving it changes both results. ``focal_loss`` with ``alpha=None``,
:func:`~kornia.losses.dice_loss`, :func:`~kornia.losses.tversky_loss`, :func:`~kornia.losses.lovasz_softmax_loss` and
:class:`~kornia.losses.HausdorffERLoss` treat class 0 like every other class.

With ``alpha=None`` and ``gamma=0``, the target slice of :func:`~kornia.losses.focal_loss` is the cross entropy, but
its ``'mean'`` is not that of :func:`~torch.nn.functional.cross_entropy`. It divides the sum by every element of the
``(B, C, *)`` output, ``C`` times the pixel count, ignored pixels included, and it does not normalise ``weight``, where
``cross_entropy`` divides by the summed weights of the target classes of the non-ignored pixels. To port
``F.cross_entropy(pred, target, weight=w, ignore_index=i)``, divide
``focal_loss(pred, target, alpha=None, gamma=0.0, reduction='sum', weight=w, ignore_index=i)`` by
``w[target[target != i]].sum()``, or by the number of non-ignored pixels when there is no ``w``.

The other dense-prediction losses against their references:

- :func:`~kornia.losses.binary_focal_loss_with_logits` with its defaults, ``alpha=0.25`` on the positive term,
  ``gamma=2`` and ``reduction='none'``, equals ``torchvision.ops.sigmoid_focal_loss`` on targets of 0 and 1. On a
  fractional target ``t`` they differ: kornia weights its positive and negative focal terms by ``t`` and ``1 - t``,
  torchvision applies ``alpha_t * (1 - p_t) ** gamma`` to the whole binary cross entropy. Its ``pos_weight`` of shape
  ``(C,)`` runs along dim 1: ``binary_focal_loss_with_logits(pred, target, alpha=None, gamma=0.0, pos_weight=pw)``
  equals ``F.binary_cross_entropy_with_logits(pred, target, pos_weight=pw.view(C, 1, 1), reduction='none')`` on
  ``(B, C, H, W)``, where torch broadcasts a ``(C,)`` ``pos_weight`` along the last axis.
- The Lovász losses score each image on its own, as the reference implementation of Berman et al.
  (``bermanmaxim/LovaszSoftmax``) does with ``per_image=True``: :func:`~kornia.losses.lovasz_hinge_loss` is its
  ``lovasz_hinge(pred[:, 0], target, per_image=True)``, the reference default, and
  :func:`~kornia.losses.lovasz_softmax_loss` is ``lovasz_softmax(pred.softmax(1), target, classes='all',
  per_image=True)``. The ``lovasz_softmax`` defaults, ``classes='present'`` and ``per_image=False``, score only the
  classes present and flatten the batch into one image, which gives another value.

A class absent from the target has no overlap to score, and the per-class functions treat it differently:

- :func:`~kornia.metrics.mean_iou` returns IoU 1 for a class absent from both the target and the prediction, through
  its ``eps``, and leaves the averaging over classes to the caller: drop the classes that occur in neither map before
  averaging its ``(B, K)`` output. scikit-learn's ``jaccard_score`` leaves such a class out unless ``labels`` names it,
  and then scores it 0, or 1 with ``zero_division=1.0``.
- ``dice_loss(average='macro')`` and :func:`~kornia.losses.tversky_loss` leave it out of the sample's mean, whether
  it is predicted or not.
- :func:`~kornia.metrics.mean_average_precision` pools the images and leaves out a class without a ground-truth object
  in any of them, detected or not; its entry in the per-class dictionary is ``-1``.

Task metrics
^^^^^^^^^^^^

:func:`~kornia.metrics.confusion_matrix` puts the target class on the rows and the predicted class on the columns, and
:func:`~kornia.metrics.mean_iou` reads its per-class IoU from that matrix.

The task metrics differ in what they return, in what they pool and in their scale: percent, fraction, pixels or
degrees. Each Convention block states its own, and the table collects them. For the background class 0 and for a
class absent from a sample, see `Dense-prediction losses`_.

.. list-table::
   :header-rows: 1

   * - functions
     - output
     - pooled over
     - scale
   * - :func:`~kornia.metrics.accuracy`
     - a list of 0-d tensors, one per entry of ``topk``
     - the batch
     - percent, ``[0, 100]``
   * - :func:`~kornia.metrics.confusion_matrix`
     - ``(B, K, K)``, rows the target and columns the prediction
     - nothing: one matrix per sample
     - counts
   * - :func:`~kornia.metrics.mean_iou`
     - ``(B, K)``
     - nothing: one IoU per sample and class
     - fraction, ``[0, 1]``
   * - :func:`~kornia.metrics.mean_iou_bbox`
     - ``(B1, B2)``
     - nothing: one IoU per pair of boxes
     - fraction, ``[0, 1]``
   * - :func:`~kornia.metrics.mean_average_precision`
     - a 0-d tensor and a ``{class id: AP}`` dict
     - the detections of all images, per class, then a mean over the classes with objects
     - fraction, ``[0, 1]``; ``-1`` for a class without objects, and for the mAP when no image has a foreground
       object
   * - :func:`~kornia.metrics.aepe`, :func:`~kornia.metrics.average_endpoint_error`, :class:`~kornia.metrics.AEPE`
     - 0-d, or ``(*)`` for ``reduction='none'``
     - every position of every sample
     - the units of the flow
   * - :func:`~kornia.metrics.mean_absolute_disparity_error`,
       :func:`~kornia.metrics.root_mean_squared_disparity_error`
     - 0-d, or ``(*)`` for ``reduction='none'``
     - every valid pixel of every image
     - pixels
   * - :func:`~kornia.metrics.mean_bad_pixel_error`, :func:`~kornia.metrics.kitti_d1_error`
     - 0-d, or ``(*)`` for ``reduction='none'``
     - every valid pixel of every image
     - fraction, ``[0, 1]``
   * - :func:`~kornia.metrics.angle_error_mat`, :func:`~kornia.metrics.angle_error_vec`
     - ``(*)``, 0-d for one pair
     - nothing: one angle per pair
     - degrees
   * - :func:`~kornia.metrics.pose_errors`
     - a dict of ``(B,)`` tensors, ``(1,)`` for one pose
     - nothing: one error per pose
     - degrees
   * - :func:`~kornia.metrics.translation_ate`
     - ``(*)``, ``(1,)`` for one translation
     - nothing: one distance per sample
     - the units of the translations
   * - :func:`~kornia.metrics.auc_from_errors`
     - a ``{threshold: AUC}`` dict of Python floats
     - all errors
     - percent, ``[0, 100]``
   * - :class:`~kornia.metrics.AverageMeter`
     - ``avg``, a Python float
     - every update, weighted by its ``n``
     - the scale of the values passed

Porting the task metrics from other libraries:

- scikit-learn takes the target first and kornia the prediction; both put the target on the rows:
  ``confusion_matrix(pred, target, K)[b]`` equals
  ``sklearn.metrics.confusion_matrix(target[b].ravel(), pred[b].ravel(), labels=range(K))``. ``normalized=True`` is
  ``normalize='true'`` up to the ``1e-6`` added to every row sum, not the per-prediction ``normalize='pred'``. Without
  tied scores, ``accuracy(pred, target, topk=(k,))[0]`` is 100 times
  ``top_k_accuracy_score(target, pred, k=k, labels=range(C))`` for more than two classes, and 100 times
  ``accuracy_score(target, pred.argmax(1))`` for ``k=1``.
- The IoU of a whole batch or dataset, which
  ``jaccard_score(target.ravel(), pred.ravel(), labels=range(K), average=None)`` and torchmetrics'
  ``MulticlassJaccardIndex(num_classes=K, average='none')`` report, is the IoU of the summed matrix
  ``confusion_matrix(pred, target, K).sum(0)``, not the mean of :func:`~kornia.metrics.mean_iou` over the batch.
- :func:`~kornia.metrics.mean_average_precision` matches a detection to an object only when their IoU is strictly
  greater than ``threshold``, as ``voc_eval`` of py-faster-rcnn does, and computes that IoU on exclusive boxes, as
  :func:`~kornia.metrics.mean_iou_bbox` does, where py-faster-rcnn and the PASCAL VOC devkit add 1 to every box width
  and height. The devkit (``VOCevaldet.m``) matches at ``IoU >= 0.5``, so a detection whose IoU equals the threshold
  is a false positive in kornia and a true positive there.
- COCO's evaluation, run as torchmetrics' ``MeanAveragePrecision(iou_thresholds=[0.5],
  rec_thresholds=[i / 10 for i in range(11)], backend='pycocotools')``, pools the detections of all images,
  interpolates at kornia's 11 recall levels and leaves a class without objects out of the mean, with AP ``-1`` where it
  lists one, as kornia does. It reproduces kornia's AP except in five cases: it matches at ``IoU >= 0.5``; it gives a
  detection whose best object is already matched to the best object still free; it scores class 0 like every other
  class; it scores only the 100 highest-scoring detections of a class in each image; and it ranks detections of equal
  score by image, then in input order, where kornia leaves their order to ``torch.sort``.
- torchvision's RAFT models return a list of ``(B, 2, H, W)`` flows, channel first, where
  :func:`~kornia.metrics.aepe` takes ``(B, H, W, 2)``. The endpoint error of RAFT's training code,
  ``((pred - gt) ** 2).sum(1).sqrt()`` on channel-first flows (``sum(0)`` on one flow in its evaluation code), equals
  ``aepe(pred.permute(0, 2, 3, 1), gt.permute(0, 2, 3, 1), reduction='none')``. Its mean is the default ``'mean'``
  where every pixel is valid, as in RAFT's Chairs and Sintel evaluation; RAFT's training metrics and its KITTI
  evaluation average only the valid pixels, KITTI image by image, so take those means of the ``'none'`` map.
- :func:`~kornia.metrics.kitti_d1_error` applies the outlier rule of the KITTI 2015 devkit (``evaluate_scene_flow.cpp``)
  and pools like it: the devkit adds up outliers and valid pixels over all images before dividing, so its value is not
  the mean of per-image D1. The devkit also reads a ground-truth disparity of 0 as invalid and scores its own
  interpolated version of the estimate: pass a dense prediction and ``valid_mask=target > 0`` to reproduce its
  number.
- SuperGlue's ``pose_auc`` and the ``cal_error_auc`` of glue-factory, the evaluation code of LightGlue, return the pose
  AUC as a fraction, glue-factory's rounded to four decimals. Both equal :func:`~kornia.metrics.auc_from_errors`
  divided by 100 for errors without NaN: the references count a NaN as a failure, kornia propagates it. SuperGlue's
  ``match_pairs.py`` prints 100 times that fraction, a percentage like kornia's; glue-factory reports the fraction
  itself. glue-factory's pose error, the larger of the rotation angle and the folded translation angle, is the
  ``"max_err"`` of :func:`~kornia.metrics.pose_errors` with the default ``fold_translation=True``. For a zero
  ground-truth translation glue-factory reads a translation error of 90 degrees, where
  :func:`~kornia.metrics.pose_errors` returns NaN.

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
       ``motion_from_essential_choose_solution`` selects it by cheirality from pixel coordinates and also returns
       the number of points that passed; ``0`` means none did
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
