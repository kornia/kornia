CUDA, float16 and bfloat16 support
==================================

.. meta::
   :description: Kornia CUDA and half-precision support, CPU known failures, CI xfail and xpass outcomes, and a local CUDA test-suite measurement with its CUDA-only failures.

The module table below counts entries in the Linux CPU known-failure manifests
(`cpu_float16.txt <https://github.com/kornia/kornia/blob/main/testing/half_precision_xfails/cpu_float16.txt>`_,
`cpu_bfloat16.txt <https://github.com/kornia/kornia/blob/main/testing/half_precision_xfails/cpu_bfloat16.txt>`_), pinned to Python 3.11 and PyTorch 2.15.0.
There are **243 float16 and 255 bfloat16 recorded failures**. These are parametrized test cases, not counts of
unsupported functions. `#4153 <https://github.com/kornia/kornia/issues/4153>`_ tracked them until it was closed on
2026-10-18; no single issue tracks the remaining entries.
No CI job covers CUDA or MPS half precision; CPU results do not establish support on those backends. A local CUDA
measurement of the whole test suite is in :ref:`cuda-test-suite-measurement`.

.. list-table:: Known failures by module (Linux CPU)
   :header-rows: 1
   :widths: 25 10 10 12 43

   * - Module
     - float16
     - bfloat16
     - Known failures
     - Notes
   * - ``kornia.color``
     - ✅
     - ✅
     - 0 / 0
     - Lab/Luv conversions and ``rgb255_to_normals`` compute half inputs in float32 and return the input dtype
   * - ``kornia.filters``
     - ✅
     - ✅
     - 0 / 0
     - Canny, Gaussian kernels/blur and FFT convolution compute half inputs in float32 and return the input dtype
   * - ``kornia.enhance``
     - ✅
     - ✅
     - 0 / 0
     - ZCA keeps half-input fitted state in float32; whitening and JPEG return the input dtype after float32
       computation
   * - ``kornia.morphology``
     - ✅
     - ✅
     - 0 / 0
     -
   * - ``kornia.augmentation``
     - ⚠️
     - ⚠️
     - 59 / 47
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

.. _cuda-test-suite-measurement:

CUDA test-suite measurement
---------------------------

Kornia accepts CUDA tensors for GPU acceleration, subject to each operator's dtype and backend limitations.
There is no CUDA CI job or checked-in CUDA known-failure manifest, so the totals below are a local measurement of
one revision, not a guarantee for later ones: ``e4a444e9b`` (2026-11-04), an RTX 4090, Python 3.11 and PyTorch
2.15.0+cu132 with TF32 off, over ``tests/`` with ``--device=cuda``. That option deselects the 1673 device-agnostic
tests, which exercise CPU-only code. The half run executed in one process (``KORNIA_TEST_IN_SUBPROCESS=1``), and
every failure without a CPU manifest entry was re-run with ``--isolate-half-precision``. The compile run set
``KORNIA_TEST_OPTIMIZER=inductor`` and selected the directories and ``-k "dynamo or compile"`` of the CI ``dynamo``
job.

.. list-table::
   :header-rows: 1

   * - CUDA run
     - Passed
     - Failed
     - Skipped
     - XFAIL
     - XPASS
   * - float32 + float64
     - 46079
     - 0
     - 9198
     - 46
     - 11
   * - float16 + bfloat16
     - 45024
     - 440
     - 9813
     - 50
     - 7
   * - float32, ``torch.compile`` (inductor) tests
     - 853
     - 6
     - 80
     - 1
     - 1

**float32 and float64.** No failures.

**float16 and bfloat16.** 408 of the 440 failures are the CUDA counterparts of entries in the CPU manifests. Of the
other 32, 21 pass when re-run in an isolated process, and 2 draw unseeded inputs that fail on CPU too
(``TestFindHomographyFromLinesDLT::test_clean_points_iter[float16-5]`` and
``TestQuaternion::test_axis_angle[bfloat16-5]``). 9 fail on CUDA only:

- 5 because CUDA picks kernels by batch size, so a half-precision model gives an image slightly different outputs in
  a batch than alone; on CPU the batched and single outputs are bitwise equal. With cuDNN disabled, SmallSR's are
  bitwise equal on CUDA too. LightGlue has no convolutions, so there the batch-dependent kernels are its matrix
  multiplications or attention. The tests are ``TestLoFTR::test_extract_features_shapes_and_batching`` (float16 and bfloat16),
  ``TestSmallSRNetWrapper::test_feature_consistency_across_batch_sizes`` (float16 and bfloat16) and
  ``TestLightGlueBatchedPairs::test_convention_lightglue_batch_below_the_pruning_threshold_runs_together_5898``
  (bfloat16). With pretrained weights on Photo Tourism images, the change between a batch and a single image is
  smaller than the error of half precision itself, which the "vs float32" columns show:

  .. list-table::
     :header-rows: 1

     * - Model (CUDA)
       - float16: batch vs single
       - float16 vs float32
       - bfloat16: batch vs single
       - bfloat16 vs float32
     * - ``SmallSRNetWrapper`` ×3, 4 images of 160 by 214 pixels: PSNR
       - 88.4 dB
       - 68.7 dB
       - 82.9 dB
       - 50.3 dB
     * - ``LoFTR("outdoor")``, 4 pairs of 480 by 640 pixels: matches identical
       - 99.6 %
       - 79.2 %
       - 95.5 %
       - 91.7 %
     * - ``ALIKED`` (float32, 1024 keypoints) + ``LightGlue("aliked")``, same pairs, early stopping and pruning off:
         matches identical
       - 99.9 %
       - 99.1 %
       - 98.3 %
       - 93.1 %

  The batch held two pairs (LoFTR, LightGlue) or four images (SmallSR). In float32, with TF32 off, the batched and
  single matches and SmallSR outputs were identical, and LoFTR's features differed by at most 3e-5. In half precision,
  SmallSR's batched outputs changed by at most 0.002 (float16) and 0.008 (bfloat16) of the ``[0, 1]`` range.

- 4 are ``TestRandomRotation3DBackward::test_param`` in bfloat16, with ``degrees`` a tensor of 10, 20 and 30
  degrees. This is a bfloat16 limit rather than a CUDA one. The test takes one SGD step with learning rate 10. On
  CUDA the steps are at most 0.054 degrees, each under half a bfloat16 ulp of its angle (0.031 at 10 degrees, 0.0625
  at 20 and 30 degrees), so no angle changes; the bfloat16 steps match the float32 ones within 2 %, apart from one
  of 7e-5 degrees. On CPU the sampled rotations differ, and the step at 10 degrees (0.032 to 0.036 degrees) passes
  half an ulp. The test checks the sum of the changes, so it passes on CPU, although the other two angles do not move
  there either.

**torch.compile.** 6 compiled tests fail on CUDA and pass on CPU. Each also fails when run on its own and on
PyTorch 2.14.0:

- ``TestAugmentationCompile::test_compile_resized_crop_replay[size1-nearest-None]``: 12 of 1080 values differ from
  eager by up to 0.97.
- ``TestOtsuThreshold::test_dynamo_bin_edge_pixels_match_eager_5425``: 2 of 5140 labels differ from eager by one.
- ``TestFitLine::test_dynamo_fit_line_2d_large_offsets``: differs from eager by 1.2e-3 (tolerance 1e-5).
- ``TestHyperplane::test_dynamo_skips_degenerate_checks``: differs from eager by 0.023.
- ``TestKeypointsIndexing::test_dynamo_scalar`` for ``Keypoints`` and ``Keypoints3D``: Dynamo raises
  ``PendingUnbackedSymbolNotFound``.

**Open CUDA-only issues that the test suite does not exercise:**

- `#6278 <https://github.com/kornia/kornia/issues/6278>`_: ``RANSAC(compile=True)`` on CUDA can store a CPU program
  in the inductor cache after a CPU compile, and later CUDA processes that share the cache segfault.
- `#6427 <https://github.com/kornia/kornia/issues/6427>`_: ``OnnxLightGlue`` binds GPU 0 for any CUDA device.
- `#6559 <https://github.com/kornia/kornia/issues/6559>`_: batched SVD of matrices with more than 32 rows makes
  ``solve_pnp_dlt``, ``fit_plane`` and ``find_essential`` up to 200× slower on CUDA than on CPU.

For CUDA half tests, use ``--isolate-half-precision`` so each test gets a fresh process. They are skipped by default
without it; selecting a half dtype alone does not exercise CUDA half support:

.. code-block:: bash

   pixi run test-module tests/color/ --device=cuda --dtype=float16,bfloat16 --isolate-half-precision
