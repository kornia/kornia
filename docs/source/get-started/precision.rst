CUDA, float16 and bfloat16 support
==================================

.. meta::
   :description: Kornia CUDA and half-precision support, CPU known failures, CI xfail and xpass outcomes, and CUDA testing limitations.

The module table below counts entries in the Linux CPU known-failure manifests
(`cpu_float16.txt <https://github.com/kornia/kornia/blob/main/testing/half_precision_xfails/cpu_float16.txt>`_,
`cpu_bfloat16.txt <https://github.com/kornia/kornia/blob/main/testing/half_precision_xfails/cpu_bfloat16.txt>`_), pinned to Python 3.11 and PyTorch 2.14.0.
There are **250 float16 and 274 bfloat16 recorded failures**. These are parametrized test cases, not counts of
unsupported functions. The remaining work is tracked in `#4153 <https://github.com/kornia/kornia/issues/4153>`_.
No CI job covers CUDA or MPS half precision; CPU results do not establish support on those backends.

.. list-table:: Known failures by module (Linux CPU)
   :header-rows: 1
   :widths: 25 10 10 12 43

   * - Module
     - float16
     - bfloat16
     - Known failures
     - Notes
   * - ``kornia.color``
     - ⚠️
     - ⚠️
     - 1 / 9
     - float16: RGB255 round-trip accuracy; bfloat16: Lab, Luv and RGB255 accuracy
   * - ``kornia.filters``
     - ⚠️
     - ⚠️
     - 6 / 8
     - Canny and Gaussian kernel/blur accuracy; CPU FFT convolution computes in float32
   * - ``kornia.enhance``
     - ✅
     - ⚠️
     - 0 / 1
     - bfloat16: ZCA accuracy; JPEG computes half inputs in float32 and returns the input dtype
   * - ``kornia.morphology``
     - ✅
     - ✅
     - 0 / 0
     -
   * - ``kornia.augmentation``
     - ⚠️
     - ⚠️
     - 59 / 48
     - float16: container autocast/runtime errors and accuracy; both dtypes: 3D backward and random-generator accuracy
   * - ``kornia.geometry.transform``
     - ⚠️
     - ⚠️
     - 31 / 31
     - Rotation matrices, affine/perspective warps and homography-warper accuracy
   * - ``kornia.geometry.camera``
     - ⚠️
     - ⚠️
     - 13 / 23
     - Projection, distortion round trips and stereo reprojection; includes test-side dtype assertions
   * - ``kornia.geometry.calibration``
     - ⚠️
     - ⚠️
     - 13 / 12
     - solve_pnp_dlt rejects half inputs; undistort_points accuracy
   * - ``kornia.geometry.epipolar``
     - ⚠️
     - ⚠️
     - 36 / 36
     - Essential-matrix decomposition/motion and KRt_from_projection lack CPU half kernels; fundamental-matrix accuracy
   * - ``kornia.geometry.homography``
     - ⚠️
     - ⚠️
     - 10 / 16
     - DLT solver accuracy, including iterated and line-based variants
   * - ``kornia.geometry.liegroup``
     - ✅
     - ✅
     - 0 / 0
     - So2/Se2 reject bfloat16 rotation construction: PyTorch has no complex bfloat16 dtype; these expected errors/skips
       are outside the manifests
   * - ``kornia.geometry.solvers``
     - ✅
     - ✅
     - 0 / 0
     -
   * - ``kornia.geometry.subpix``
     - ⚠️
     - ⚠️
     - 6 / 3
     - SpatialSoftArgmax2d and ConvQuadInterp3d accuracy; bfloat16: render_gaussian2d accuracy
   * - ``kornia.geometry.conversions``
     - ⚠️
     - ⚠️
     - 65 / 65
     - Angle-axis, quaternion and rotation-matrix accuracy; includes tests/integration/test_conversions.py
   * - ``kornia.geometry.ransac``
     - ⚠️
     - ⚠️
     - 2 / 2
     - Essential-model fitting raises in epipolar solvers
   * - ``kornia.geometry (other)``
     - ⚠️
     - ⚠️
     - 6 / 13
     - Box-transform and depth accuracy
   * - ``kornia.image``
     - ⚠️
     - ⚠️
     - 2 / 2
     - Circle/ellipse polygon-fill comparisons use vertices and reference masks computed in half precision
   * - ``kornia.losses``
     - ✅
     - ⚠️
     - 0 / 2
     - bfloat16: total-variation accuracy
   * - ``kornia.feature``
     - ✅
     - ✅
     - 0 / 0
     - Matching uses a manual cdist fallback for half dtypes; LightGlue's float16 tests are skipped
   * - ``kornia.metrics``
     - ✅
     - ✅
     - 0 / 0
     -
   * - ``kornia.models``
     - ✅
     - ⚠️
     - 0 / 1
     - bfloat16: RT-DETR RepVGG fusion accuracy
   * - ``contrib, core, io, onnx, sensors, tracking, utils``
     - ✅
     - ⚠️
     - 0 / 2
     - bfloat16: histogram matching and _torch_svd_cast accuracy

✅ No entries in the CPU known-failure manifest   ⚠️ Has recorded CPU failures.
Neither symbol promises complete dtype support: skips, expected exceptions and source-level xfails are outside
these counts. Some operators compute internally in float32 or float64 and return the input dtype.

**Observed CPU CI outcomes:**

The `scheduled run at 93f1376e, 2026-10-21 <https://github.com/kornia/kornia/actions/runs/39524889979>`_
used Linux x86_64, Python 3.11 and PyTorch 2.14.0, without slow or compile tests:

.. list-table::
   :header-rows: 1

   * - Run
     - Passed
     - Failed
     - Skipped
     - XFAIL (all)
     - XPASS
     - Strict manifest entries
   * - `CPU float16 <https://github.com/kornia/kornia/actions/runs/39524889979/job/118977300500>`_
     - 23905
     - 0
     - 5132
     - 284
     - 4
     - 250
   * - `CPU bfloat16 <https://github.com/kornia/kornia/actions/runs/39524889979/job/118977300410>`_
     - 23726
     - 0
     - 5286
     - 309
     - 4
     - 274

Each job also deselected 3 ONNX conformance tests. XFAIL totals include both the strict manifest entries and
source-level expected failures. The 4 XPASSes in each successful job are non-strict expected failures that passed;
a passing **manifest** entry makes verification fail, as do a changed exception, a skipped entry or a stale node ID.
A green job therefore means the recorded baseline reproduced, even though known failures and skips remain.
These totals describe the linked revision; the module table describes the checked-in manifests.

Verify the complete CPU profiles in the pinned environment with:

.. code-block:: bash

   KORNIA_TEST_OPTIMIZER= pixi run test-module tests/ --verify-known-failures --known-failure-profile=cpu-float16
   KORNIA_TEST_OPTIMIZER= pixi run test-module tests/ --verify-known-failures --known-failure-profile=cpu-bfloat16


``pixi run test-half`` is an unseeded CPU sweep, without the strict profile baseline. See
`TESTING.md <https://github.com/kornia/kornia/blob/main/TESTING.md#7-half-precision-dtypes-float16--bfloat16>`_ for focused checks and baseline maintenance.

**CUDA support and measurement limits:**

Kornia accepts CUDA tensors for GPU acceleration, subject to each operator's dtype and backend limitations.
There is no blocking CUDA CI job or checked-in CUDA known-failure manifest, so the CPU xfail counts above do not
provide a current CUDA pass rate. The previous local measurements are retained here as historical evidence only:

.. list-table::
   :header-rows: 1

   * - Historical run (``f8449854``, 2026-09-23)
     - Passed
     - Failed
     - Skipped
   * - CUDA float32
     - 12193
     - 22
     - 3957
   * - CUDA float16
     - 11738
     - 451
     - 3983
   * - CUDA bfloat16
     - 11736
     - 500
     - 3936

These used an RTX 4090, Python 3.11 and PyTorch 2.14.0+cu130. Source-marked xfails were excluded. Half tests ran
in-process with ``KORNIA_TEST_IN_SUBPROCESS=1``, followed by isolated retries of failures; two homography tests were
deselected. They are not a measurement of the current branch. Those two tests (``TestFindHomographyDLT::test_nocrash``
and ``test_nocrash_lu``) now pass locally in all three CUDA dtypes at ``649b3f86d``; a fresh full CUDA sweep is needed
before quoting new totals.

For CUDA half tests, use ``--isolate-half-precision`` so each test gets a fresh process. They are skipped by default
without it; selecting a half dtype alone does not exercise CUDA half support:

.. code-block:: bash

   pixi run test-module tests/color/ --device=cuda --dtype=float16,bfloat16 --isolate-half-precision
