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
   * - ``homography_warp``
     - ``False``
   * - ``undistort_image``
     - ``True``
   * - ``warp_frame_depth``
     - ``True``
   * - ``DepthWarper`` / ``depth_warp``
     - ``True`` (default)

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

- One :class:`kornia.augmentation.container.AugmentationSequential` call draws once and
  applies that draw to every registered data type — with the non-rigid
  exceptions in the next bullet; ``.inverse()`` undoes the geometric part for
  all of them. Never augment image and mask through two separate calls — the
  random draws will differ.
- That holds for the **rigid** (matrix) augmentations. A non-rigid op has no
  transform matrix, so it warps the image only:
  :class:`kornia.augmentation.RandomElasticTransform` returns keypoints, boxes
  **and** masks unchanged, while
  :class:`kornia.augmentation.RandomThinPlateSpline` and
  :class:`kornia.augmentation.RandomFisheye` return keypoints and boxes
  unchanged and raise ``NotImplementedError`` when a ``mask`` key is
  registered (`#4420 <https://github.com/kornia/kornia/issues/4420>`_).
- Masks are resampled with nearest interpolation and keep their value set and
  their dtype. Boxes are read and written in the inclusive ``xyxy_plus``
  convention of :class:`kornia.geometry.boxes.Boxes` — see *Bounding boxes*
  above — and flips are inclusive about the integer pixel centre,
  ``x' = W - 1 - x``.
- A positive ``degrees`` turns the image **counter-clockwise as displayed** on
  :class:`kornia.augmentation.RandomRotation`, matching
  :func:`kornia.geometry.transform.rotate`, and **clockwise** on
  :class:`kornia.augmentation.RandomAffine`
  (`#4408 <https://github.com/kornia/kornia/issues/4408>`_).

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

- Parameters are sampled on the **CPU**, whatever the device of the input, and
  come back in ``torch.get_default_dtype()``, read at call time, rather than in
  the input's dtype. Set ``torch.set_default_dtype`` before the call to sample
  in another dtype. The ``p`` / ``p_batch`` gate is the exception: its dtype is
  the one recorded at construction.
- ``set_rng_device_and_dtype`` is not the way to move sampling to an
  accelerator: on most classes it moves the ``p`` / ``p_batch`` gate and nothing
  else (`#4426 <https://github.com/kornia/kornia/issues/4426>`_). Pointed at an
  accelerator device, the classes that do move some of their drawn keys with it
  are ``CenterCrop``, ``Resize``, ``LongestMaxSize``, ``SmallestMaxSize``,
  ``CenterCrop3D``, ``RandomElasticTransform``, ``RandomThinPlateSpline``,
  ``ColorJitter``, ``RandomChannelDropout``, ``RandomChannelShuffle``,
  ``RandomGaussianBlur``, ``RandomGaussianIllumination``,
  ``RandomGaussianNoise`` and ``RandomPlanckianJitter`` — ``RandomCrop`` and
  ``RandomResizedCrop``, despite the family resemblance, do not. On
  ``RandomShear``, ``RandomLinearIllumination`` and
  ``RandomLinearCornerIllumination`` the call leaves the module in a state where
  the next ``forward`` raises ``RuntimeError``.
- Reproducibility is **global-seed only**: ``torch.manual_seed`` before the
  call reproduces the draw bitwise. There is no per-instance ``generator=`` —
  it raises at construction, and ``forward`` accepts and silently drops it, as
  it does any other unknown keyword
  (`#4427 <https://github.com/kornia/kornia/issues/4427>`_).
- ``same_on_batch=True`` asks every sample of the batch to share one draw. The
  ``p`` gate and the transform parameters follow it; keys that index or pair up
  the batch — ``batch_idx``, the mix-pairing permutation, the jitter ``order``
  — stay per sample by construction, and
  :class:`kornia.augmentation.RandomRain` does not honour it for its drop count
  (`#4448 <https://github.com/kornia/kornia/issues/4448>`_). A handful of
  classes do not take the argument at all. On
  ``AugmentationSequential`` the flag is three-state: ``None`` keeps each
  child's own setting, ``True`` and ``False`` overwrite it.
- Under a :class:`torch.utils.data.DataLoader` the rule is torch's, not
  Kornia's: each worker's global CPU generator is seeded ``base_seed +
  worker_id``, so the workers draw different augmentations and the run is
  reproducible from the base seed. A ``worker_init_fn`` that reseeds every
  worker to one fixed value makes them draw the *same* augmentation — the
  classic duplicated-augmentation bug. See `Randomness in multi-process data
  loading <https://pytorch.org/docs/stable/notes/randomness.html#dataloader>`_.
- How many values a class takes out of the generator, and in which order, is
  observable behaviour: changing it shifts every later draw in a seeded
  pipeline, so it is a breaking change even when each individual draw is still
  correctly distributed.

Serializing an augmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- An augmentation carries no learnable parameters. Some classes expose their
  sampling range as a buffer in ``state_dict()``, but ``load_state_dict`` from
  an instance with a different range changes the buffer and changes neither the
  draw nor the ``repr``
  (`#4428 <https://github.com/kornia/kornia/issues/4428>`_). Re-construct the
  augmentation to change what it samples.
- ``pickle`` and ``copy.deepcopy`` do carry the last ``_params`` and the
  ``transform_matrix``, so a saved augmentation replays its last draw.

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
14. Wrong ``data_keys`` box format — ``"bbox"`` means 4-corner ``(B, 4, 2)``;
    use ``"bbox_xyxy"``/``"bbox_xywh"`` for coordinate formats.
15. Assuming one rotation direction across the augmentations — a positive
    ``degrees`` on ``RandomAffine`` turns the image clockwise, on
    ``RandomRotation`` counter-clockwise.
16. Expecting a non-rigid augmentation (``RandomElasticTransform``,
    ``RandomThinPlateSpline``, ``RandomFisheye``) to carry masks, boxes and
    keypoints along with the image — it does not.
17. Calling ``set_rng_device_and_dtype`` to move parameter sampling to the GPU
    — on most classes it moves the ``p`` gate and nothing else.

.. tip::

   Machine-readable copies of these conventions live at
   `llms.txt <https://kornia.readthedocs.io/en/latest/llms.txt>`_ and
   `llms-full.txt <https://kornia.readthedocs.io/en/latest/llms-full.txt>`_
   at the docs root.
