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
     - ⚠️ Partial
     - Most color space conversions work for both half-precision dtypes.
       FFT-based operations may fail on CUDA.
   * - ``kornia.filters``
     - ⚠️ Partial
     - ⚠️ Partial
     - Basic convolution-based filters (Gaussian, Sobel, Median, Box) work
       for both dtypes. FFT-based operations (``fft_conv``) may fail on CUDA.
   * - ``kornia.enhance``
     - ⚠️ Partial
     - ⚠️ Partial
     - Histogram equalization, CLAHE, gamma correction, and ZCA whitening work
       for both dtypes. ZCA linalg ops go through ``_torch_svd_cast`` /
       ``_torch_inverse_cast`` which promote to float32 before computing.
   * - ``kornia.morphology``
     - ✅ Yes
     - ✅ Yes
     - Convolution and pooling; ``top_hat``, ``bottom_hat`` and ``gradient``
       also subtract two dilation/erosion results, so bfloat16 loses roughly
       0.4% relative accuracy on them. That is within kornia's own bfloat16
       tolerance, but six tests override it with a tighter constant and fail;
       see `issue #4081 <https://github.com/kornia/kornia/issues/4081>`_.
   * - ``kornia.augmentation``
     - ⚠️ Partial
     - ⚠️ Partial
     - Both dtypes are accepted by ``validate_tensor``. Most ops work;
       precision-sensitive transforms (e.g. affine with large rotations) may
       produce inaccurate results at half precision.
   * - ``kornia.geometry.transform``
     - ⚠️ Partial
     - ⚠️ Partial
     - Affine, homography, resize, and warp operations use ``_torch_inverse_cast``
       / ``_torch_solve_cast`` which promote to float32 and cast back;
       both dtypes work.
   * - ``kornia.geometry.camera``
     - ⚠️ Partial
     - ⚠️ Partial
     - Pinhole camera model and most projection ops work for both dtypes.
       ``StereoCamera`` accepts both float16 and bfloat16.
   * - ``kornia.geometry.calibration``
     - ❌ No
     - ❌ No
     - ``solve_pnp_dlt()`` explicitly checks that inputs are ``float32`` or
       ``float64`` and raises otherwise.
   * - ``kornia.geometry.epipolar``
     - ⚠️ Partial
     - ⚠️ Partial
     - SVD and solve operations use ``_torch_svd_cast`` / ``_torch_solve_cast``
       / ``_torch_inverse_cast``; both dtypes work via casting to float32.
   * - ``kornia.geometry.homography``
     - ⚠️ Partial
     - ⚠️ Partial
     - Uses ``_torch_svd_cast``; both dtypes are promoted to float32 before SVD
       and the result is cast back.
   * - ``kornia.geometry.liegroup``
     - ⚠️ Partial
     - ⚠️ Partial
     - Most rotation/translation operations (SO2, SO3, SE2, SE3) work for both
       dtypes via cast helpers. A few code paths may still fail.
   * - ``kornia.geometry.solvers``
     - ⚠️ Partial
     - ⚠️ Partial
     - RANSAC-based solvers use ``_torch_solve_cast``; both dtypes are promoted
       before the solve and the result is cast back.
   * - ``kornia.geometry.subpix``
     - ⚠️ Partial
     - ⚠️ Partial
     - Soft-argmax and weighted softmax work for both dtypes.
       Precision-sensitive ops may produce inaccurate results.
   * - ``kornia.losses``
     - ⚠️ Partial
     - ⚠️ Partial
     - Photometric losses (SSIM, PSNR, MS-SSIM) work for both dtypes.
       Losses based on linalg operations (Hausdorff, etc.) may not.
   * - ``kornia.feature``
     - ⚠️ Partial
     - ⚠️ Partial
     - Local feature detectors and descriptors (SIFT, HardNet, DISK, DeDoDe)
       work for inference. Feature *matching* uses a manual ``cdist`` fallback
       for both half-precision dtypes on CUDA.
   * - ``kornia.metrics``
     - ⚠️ Partial
     - ⚠️ Partial
     - Simple pixel-level metrics work for both dtypes. Metrics involving linalg
       operations may not.
   * - ``kornia.models``
     - ⚠️ Partial
     - ⚠️ Partial
     - Conv-based models work for both dtypes. Attention-based models (e.g.
       VLMs, ViTs) may have internal dtype mismatches.

Legend
------

- ✅ **Yes** — Works correctly; results are accurate at the given precision.
- ⚠️ **Partial** — Some operations work; others fail at runtime or produce inaccurate results due to limited numerical range/precision.
- ❌ **No** — Not supported; raises a ``RuntimeError`` or ``TypeError`` at runtime (explicit dtype check in the implementation).

Test Results
------------

Full test suite (no ``--runslow``). Pass% = passed ÷ (passed + failed);
skipped and xfailed tests are excluded. CPU rows were measured on commit
``4ab79c78`` (2026-08-29); CUDA rows are still from ``6131e98`` (2026-03-21).

The CPU half-precision suites run as separate blocking ``float16`` and ``bfloat16`` CI jobs with strict manifests
for known failures, addressing `issue #4070 <https://github.com/kornia/kornia/issues/4070>`_. The historical table
counts are still refreshed by hand: the two CPU half rows with ``pixi run test-half``, and the CPU float32 baseline
with ``pixi run test-f32``, since ``test-half`` pins ``KORNIA_TEST_DTYPE`` to ``float16,bfloat16``.

.. list-table::
   :header-rows: 1
   :widths: 32 10 10 10 10

   * - Run
     - Passed
     - Failed
     - Skipped
     - Pass%
   * - CPU float32 *(baseline)*
     - 8499
     - 0
     - 3535
     - **100.0%**
   * - CPU float16
     - 7751
     - 689
     - 3595
     - **91.8%**
   * - CPU bfloat16
     - 7794
     - 695
     - 3545
     - **91.8%**
   * - CUDA float32 *(baseline)*
     - 7634
     - 3
     - 3280
     - **99.9%**
   * - CUDA float16 *(KORNIA_TEST_IN_SUBPROCESS=1)*
     - 6727
     - 643
     - 3556
     - **91.3%**
   * - CUDA bfloat16 *(KORNIA_TEST_IN_SUBPROCESS=1)*
     - 6695
     - 713
     - 3518
     - **90.4%**

.. note::

   CUDA half-precision tests are measured using ``KORNIA_TEST_IN_SUBPROCESS=1``
   which bypasses the ``skip_half_precision_on_cuda`` fixture.  Each test then
   runs in the same process but with the ``cuda_device_assert_guard`` fixture
   synchronising CUDA before and after each test.  For full isolation the current
   implementation uses ``subprocess.run`` for true process isolation; a fresh
   ``--isolate-half-precision`` flag spawns each test in a fresh ``subprocess.run``
   process with no shared CUDA state.

Test Suite Behaviour
--------------------

Half-precision tests live in the same directories and files as their
float32/float64 counterparts.  They are run as **separate, isolated pytest
invocations** rather than being mixed into a combined ``--dtype=all`` run.
This prevents a CUDA device-side assert in a half-precision test from
corrupting the CUDA context and causing unrelated float32 tests to fail.

.. code-block:: bash

   # Standard precision — default CI
   pixi run test --dtype=float32,float64

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

With ``--isolate-half-precision``, each float16/bfloat16 CUDA test is
intercepted by a custom ``pytest_runtest_protocol`` hook and executed in a
completely fresh Python process via ``subprocess.run``.  There is no shared
CUDA context between tests, so a device-side assert in one test cannot affect
any other.

See ``TESTING.md`` in the repository root for a full description of the
contamination mechanism and fixture implementation.
