Half-Precision Support
======================

This page documents which kornia modules support half-precision floating-point dtypes
(``torch.float16`` and ``torch.bfloat16``) and what limitations to expect.

.. list-table:: Half-Precision Support by Module
   :header-rows: 1
   :widths: 28 14 14 44

   * - Module
     - float16
     - bfloat16
     - Notes
   * - ``kornia.color``
     - ⚠️ Partial
     - ❌ No
     - Most color space conversions work. ``rgb_to_grayscale`` and
       ``bgr_to_grayscale`` explicitly reject bfloat16.
   * - ``kornia.filters``
     - ⚠️ Partial
     - ❌ No
     - Basic convolution-based filters (Gaussian, Sobel, Median, Box) generally
       work with float16. FFT-based operations (``fft_conv``) may fail on CUDA.
       All filters reject bfloat16.
   * - ``kornia.enhance``
     - ⚠️ Partial
     - ❌ No
     - Histogram equalization, CLAHE, and gamma correction work with float16.
       ZCA whitening uses ``torch.linalg.eigh`` / ``linalg.inv`` and does not
       support either half-precision dtype.
   * - ``kornia.morphology``
     - ✅ Yes
     - ✅ Yes
     - Uses only convolution and pooling; no dtype restrictions anywhere in
       the module. Both float16 and bfloat16 work correctly.
   * - ``kornia.augmentation``
     - ✅ Yes
     - ❌ No
     - All augmentation ops are validated to accept float16.
       ``AugmentationBase2D.validate_tensor`` explicitly rejects bfloat16.
   * - ``kornia.geometry.transform``
     - ⚠️ Partial
     - ❌ No
     - Affine, homography, resize, and warp operations use ``_torch_inverse_cast``
       which promotes to float32/float64 before computing and casts back, so
       float16 works. Thin-plate spline uses the same pattern via
       ``_torch_solve_cast``.
   * - ``kornia.geometry.camera``
     - ⚠️ Partial
     - ❌ No
     - Pinhole camera model and projection ops work with float16.
       ``StereoCamera.reproject_disparity_to_3D`` explicitly rejects bfloat16.
   * - ``kornia.geometry.calibration``
     - ❌ No
     - ❌ No
     - ``solve_pnp_dlt()`` explicitly checks that inputs are ``float32`` or
       ``float64`` and raises otherwise. float16 is rejected at validation time,
       before any linalg call is made.
   * - ``kornia.geometry.epipolar``
     - ⚠️ Partial
     - ❌ No
     - SVD and solve operations use ``_torch_svd_cast`` / ``_torch_solve_cast``
       which promote to float32/float64, so float16 generally works.
       A few internal helpers (e.g. ``matrix_cofactor_tensor``) call
       ``torch.linalg.inv`` directly without the cast guard and may fail for
       float16 inputs.
   * - ``kornia.geometry.homography``
     - ⚠️ Partial
     - ❌ No
     - Homography estimation uses ``_torch_svd_cast``; float16 inputs are
       promoted to float32 before SVD and the result is cast back.
   * - ``kornia.geometry.liegroup``
     - ⚠️ Partial
     - ❌ No
     - Most rotation/translation operations (SO2, SO3, SE2, SE3) work with
       float16. Operations that go through ``kornia.core.utils`` cast helpers
       are protected. A few code paths may still call linalg routines without
       casting and fail for float16.
   * - ``kornia.geometry.solvers``
     - ⚠️ Partial
     - ❌ No
     - RANSAC-based solvers use ``_torch_solve_cast``; float16 inputs are
       promoted before the solve and the result is cast back.
   * - ``kornia.geometry.subpix``
     - ⚠️ Partial
     - ❌ No
     - Soft-argmax and weighted softmax work with float16.
       Spatial-softmax operations requiring high numerical precision may produce
       inaccurate results.
   * - ``kornia.losses``
     - ⚠️ Partial
     - ❌ No
     - Photometric losses (SSIM, PSNR, MS-SSIM) work with float16.
       Losses based on linalg operations (Hausdorff, etc.) do not.
   * - ``kornia.feature``
     - ⚠️ Partial
     - ❌ No
     - Local feature detectors and descriptors (SIFT, HardNet, DISK, DeDoDe)
       may work with float16 for inference. Feature *matching* uses
       ``torch.cdist``, which is not implemented for float16 on CUDA. ALIKED
       uses ``torch.linalg.svd``.
   * - ``kornia.metrics``
     - ⚠️ Partial
     - ❌ No
     - Simple pixel-level metrics work with float16. Metrics involving linalg
       operations do not.
   * - ``kornia.models``
     - ⚠️ Partial
     - ❌ No
     - Model-dependent. Models that are entirely convolution-based may work with
       float16. Attention-based models (e.g. VLMs, ViTs) may have internal
       dtype mismatches when inputs are cast to float16.

Legend
------

- ✅ **Yes** — Works correctly; results are accurate at the given precision.
- ⚠️ **Partial** — Some operations work; others fail at runtime or produce inaccurate results due to limited numerical range/precision.
- ❌ **No** — Not supported; raises a ``RuntimeError`` or ``TypeError`` at runtime.

Test Suite Behaviour
--------------------

Because half-precision support is incomplete, all tests parameterised with
``float16`` or ``bfloat16`` are automatically marked ``xfail(strict=False)``
by a global autouse fixture in ``conftest.py``.

- **XFAIL** (``x``) — test fails as expected; no action needed.
- **XPASS** (``X``) — test unexpectedly passes; the operation actually works in half-precision.

Run the full half-precision sweep with:

.. code-block:: bash

   pixi run test tests/ --dtype=float16
   pixi run test tests/ --dtype=bfloat16

See :doc:`/get-started/installation` for environment setup and
``TESTING.md`` in the repository root for a full description of the xfail
mechanism and guidance on fixing half-precision failures.
