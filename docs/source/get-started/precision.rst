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

Test Results
------------

Measured on commit ``dee4388`` (2026-03-20), full test suite with ``--runslow``.
Pass% = passed ÷ (passed + failed + errors); skipped tests are excluded.

.. list-table::
   :header-rows: 1
   :widths: 32 10 10 10 10 10

   * - Run
     - Passed
     - Failed
     - Errors
     - Skipped
     - Pass%
   * - CPU float32 *(baseline)*
     - 7647
     - 3
     - 4
     - 3269
     - **99.9%**
   * - CUDA float32 *(baseline)*
     - 7634
     - 3
     - 4
     - 3280
     - **99.9%**
   * - CPU float16
     - 6866
     - 747
     - 4
     - 3306
     - **90.1%**
   * - CPU bfloat16
     - 6838
     - 812
     - 4
     - 3269
     - **89.3%**
   * - CUDA float16 *(--isolate-half-precision)*
     - 485 †
     - 22
     - 26
     - 10412 †
     - *(see note)*
   * - CUDA bfloat16 *(--isolate-half-precision)*
     - 6840
     - 797
     - 4
     - 3280
     - **89.5%**

† **CUDA float16 caveat:** ``fork()`` (used by ``pytest-forked``) copies the
parent's CUDA context handle into the child.  When a float16 kernel triggers a
device-side assert in the child process, the underlying GPU state is corrupted
and the parent's handle sees the same error.  The ``cuda_device_assert_guard``
fixture detects this and skips subsequent tests, producing 10 412 skips instead
of the expected 3 280.  Only 533 of ~7 641 tests actually ran; the 91 % pass
rate applies only to that subset.  True isolation for CUDA float16 requires
spawning a fresh Python process (``subprocess.run``), not ``fork()``.

Test Suite Behaviour
--------------------

Half-precision tests live in the same directories and files as their
float32/float64 counterparts.  They are run as **separate, isolated pytest
invocations** rather than being mixed into a combined ``--dtype=all`` run.
This prevents a CUDA device-side assert in a half-precision test from
corrupting the CUDA context and causing unrelated float32 tests to fail.

.. code-block:: bash

   # Standard precision — default CI
   pixi run test tests/ --dtype=float32,float64

   # Half-precision — run in isolation, per directory
   pytest tests/color/     --dtype=float16,bfloat16
   pytest tests/geometry/  --dtype=float16,bfloat16 --device=cuda

Two autouse fixtures in the root ``conftest.py`` enforce safe behaviour:

- **``skip_half_precision_on_cuda``** — skips float16/bfloat16 tests on CUDA
  in combined runs so no half-precision kernel is ever launched (and therefore
  no device-side assert can fire).
- **``cuda_device_assert_guard``** — synchronises CUDA before and after each
  CUDA test to catch async device-side assert errors in the test that caused
  them, not in the next one.  If the context is already corrupted, the test
  is skipped rather than allowed to fail spuriously.

See ``TESTING.md`` in the repository root for a full description of the
contamination mechanism and fixture implementation.
