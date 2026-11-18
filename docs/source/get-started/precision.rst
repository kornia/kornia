CUDA, float16 and bfloat16 support
==================================

.. meta::
   :description: Kornia CUDA and half-precision support: CPU known failures by module, CPU and CUDA test-suite totals measured on one revision, and the CUDA-only failures.

The module table below counts entries in the Linux CPU known-failure manifests
(`cpu_float16.txt <https://github.com/kornia/kornia/blob/main/testing/half_precision_xfails/cpu_float16.txt>`_,
`cpu_bfloat16.txt <https://github.com/kornia/kornia/blob/main/testing/half_precision_xfails/cpu_bfloat16.txt>`_), pinned to Python 3.11 and PyTorch 2.15.0.
There are **12 float16 and 17 bfloat16 recorded failures**. These are parametrized test cases, not counts of
unsupported functions. `#4153 <https://github.com/kornia/kornia/issues/4153>`_ tracked them until it was closed on
2026-10-18; no single issue tracks the remaining entries.

CUDA is tested on maintainer hardware rather than in hosted CI, whose runners have no GPU. The whole test suite runs
on an NVIDIA GPU in float32, float64, float16 and bfloat16 and under ``torch.compile`` (totals in
:ref:`measured-test-suite-totals`). It is re-run before releases, and pull requests that touch device-dependent code
are reviewed with their CUDA tests on that GPU. CUDA-only failures are fixed or listed on this page. MPS runs float32
in CI; MPS half precision is not part of CI, and CPU results alone do not establish it.

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
     - ✅
     - ⚠️
     - 0 / 4
     - Crop and perspective generators and hue/saturation adjustment compute half inputs in float32; bfloat16:
       ``RandomRotation3D`` backward, where a learning-rate step is below half an ulp of the angle
   * - ``kornia.geometry.transform``
     - ✅
     - ✅
     - 0 / 0
     - Warps, rotation matrices, homography normalization and ``crop_and_resize`` compute half inputs in float32 and
       return the input dtype
   * - ``kornia.geometry.camera``
     - ⚠️
     - ✅
     - 1 / 0
     - float16: an affine round trip to pixels beyond 65504; ``cam2pixel`` and Kannala-Brandt distortion compute
       half inputs in float32, and a half-precision stereo rig keeps ``Q`` in float32
   * - ``kornia.geometry.calibration``
     - ⚠️
     - ⚠️
     - 7 / 7
     - solve_pnp_dlt, distort_points and undistort_points compute half inputs in float32; the remaining entries are
       solve_pnp_dlt pose checks on a scene whose pixel coordinates, up to 8e4, overflow float16 or round by many
       pixels in both half dtypes
   * - ``kornia.geometry.epipolar``
     - ⚠️
     - ⚠️
     - 4 / 3
     - Essential-matrix decomposition/motion, KRt_from_projection and the fundamental-matrix estimators compute half
       inputs in float32; the remaining entries are Sampson residuals of a pixel-unit F, whose entries span eight
       decades, and float16 projection matrices with entries above 65504
   * - ``kornia.geometry.homography``
     - ✅
     - ✅
     - 0 / 0
     - The point and line DLTs, iterated or not, compute half inputs in float32 and return the input dtype
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
     - ✅
     - ✅
     - 0 / 0
     -
   * - ``kornia.geometry.conversions``
     - ✅
     - ⚠️
     - 0 / 3
     - Rotation conversions compute half inputs in float32 and return the input dtype; bfloat16: the log quaternion
       of a -179.9 degree turn rounds to the other side of the half turn. Includes tests/integration/test_conversions.py
   * - ``kornia.geometry.ransac``
     - ✅
     - ✅
     - 0 / 0
     -
   * - ``kornia.geometry (other)``
     - ✅
     - ✅
     - 0 / 0
     -
   * - ``kornia.image``
     - ✅
     - ✅
     - 0 / 0
     -
   * - ``kornia.losses``
     - ✅
     - ✅
     - 0 / 0
     -
   * - ``kornia.feature``
     - ✅
     - ✅
     - 0 / 0
     - Matching uses a manual cdist fallback for half dtypes; LightGlue's tests do not apply to float16 and are
       deselected
   * - ``kornia.metrics``
     - ✅
     - ✅
     - 0 / 0
     -
   * - ``kornia.models``
     - ✅
     - ✅
     - 0 / 0
     -
   * - ``contrib, core, io, onnx, sensors, tracking, utils``
     - ✅
     - ✅
     - 0 / 0
     -

✅ No entries in the CPU known-failure manifest   ⚠️ Has recorded CPU failures.
Neither symbol promises complete dtype support: skips, expected exceptions and source-level xfails are outside
these counts. Some operators compute internally in float32 or float64 and return the input dtype.

Verify the complete CPU profiles in the pinned environment with:

.. code-block:: bash

   KORNIA_TEST_OPTIMIZER= pixi run test-module tests/ --verify-known-failures --known-failure-profile=cpu-float16
   KORNIA_TEST_OPTIMIZER= pixi run test-module tests/ --verify-known-failures --known-failure-profile=cpu-bfloat16


``pixi run test-half`` is an unseeded CPU sweep, without the strict profile baseline. See
`TESTING.md <https://github.com/kornia/kornia/blob/main/TESTING.md#7-half-precision-dtypes-float16--bfloat16>`_ for focused checks and baseline maintenance.

.. _measured-test-suite-totals:
.. _cuda-test-suite-measurement:

Measured test-suite totals
--------------------------

Every total in this section comes from one revision, measured on one machine on one day, so the rows can be compared
with each other and with the module table above: main ``254d58112`` (2026-11-13), Linux x86_64 (WSL2), an RTX 4090,
Python 3.11 and PyTorch 2.15.0+cu132 with TF32 off. The totals describe that revision; later changes are checked on
the same GPU before release and in review, as described above. When any row is re-measured, re-measure all of them on
the same revision and replace the whole table.

.. list-table::
   :header-rows: 1

   * - Run
     - Passed
     - Failed
     - Errors
     - Skipped
     - Out of scope
     - XFAIL
     - XPASS
   * - CPU float16, strict ``cpu-float16`` profile
     - 29665
     - 0
     - 0
     - 252
     - 1605
     - 45
     - 4
   * - CPU bfloat16, strict ``cpu-bfloat16`` profile
     - 29444
     - 0
     - 0
     - 253
     - 1820
     - 50
     - 4
   * - CUDA float32 + float64
     - 39007
     - 0
     - 0
     - 934
     - 1622
     - 19
     - 5
   * - CUDA float32 with ``--runslow``
     - 21317
     - 0
     - 0
     - 636
     - 867
     - 10
     - 5
   * - CUDA float16 + bfloat16
     - 38546
     - 33
     - 3
     - 315
     - 2685
     - 23
     - 1
   * - CUDA float32, ``torch.compile`` (inductor) tests
     - 910
     - 0
     - 0
     - 38
     - 749
     - 3
     - 1

**How each row was run.** Each row is one command below, run from the repository root of the measured revision. The
CPU rows are the CI half-precision job's command, with one thread and no slow or compile tests; CUDA is hidden from the
process, and the 7 opt-in ONNX and ``torch.compile`` conformance tests are deselected. Pillow was 13.0.0 rather than the
manifests' recorded 12.3.0, a provenance difference that verification reports and accepts. XFAIL counts both the strict
manifest entries and source-level expected failures; XPASS counts non-strict expected failures that passed. A manifest
entry that passes, changes its exception, is skipped or names a stale node ID fails verification, so a clean CPU row
means the checked-in baseline reproduced, even though known failures and skips remain. "Out of scope" counts the cases
deselected because they fall outside a test's declared device, dtype or parameter scope
(``@pytest.mark.applies_to``, `#6750 <https://github.com/kornia/kornia/pull/6750>`_); they are neither run nor
skipped, and pytest-xdist does not report them, so they were counted by collecting each row's selection without
``-n``. ``--device=cuda`` also deselects the device-agnostic tests that exercise CPU-only code (8849 in the CUDA
float32 + float64 row), and runs without an optimizer drop the ``torch.compile`` tests, which the compile row runs. The
half-precision CUDA row runs in one process, and every failure without a CPU manifest entry was re-run three times with
``--isolate-half-precision``. The compile row uses the directories and ``-k`` filter of the CI ``dynamo`` job.

.. code-block:: bash

   # CPU float16 and CPU bfloat16 (replace float16 with bfloat16 in both places)
   MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 KORNIA_TEST_OPTIMIZER= KORNIA_TEST_DEVICE=cpu KORNIA_TEST_DTYPE=float16 \
     python -m pytest tests/ --verify-known-failures --known-failure-profile=cpu-float16 \
     --timeout=600 --timeout-method=thread -o faulthandler_timeout=300
   # CUDA float32 + float64
   OMP_NUM_THREADS=2 KORNIA_TEST_OPTIMIZER= KORNIA_TEST_DEVICE=cuda KORNIA_TEST_DTYPE=float32,float64 \
     python -m pytest tests/ -n 4
   # CUDA float32 with --runslow
   OMP_NUM_THREADS=2 KORNIA_TEST_OPTIMIZER= KORNIA_TEST_DEVICE=cuda KORNIA_TEST_DTYPE=float32 \
     python -m pytest tests/ --runslow -n 4
   # CUDA float16 + bfloat16, in one process
   OMP_NUM_THREADS=2 KORNIA_TEST_IN_SUBPROCESS=1 KORNIA_TEST_OPTIMIZER= KORNIA_TEST_DEVICE=cuda \
     KORNIA_TEST_DTYPE=float16,bfloat16 python -m pytest tests/ -n 4 -m "not cuda_device_assert"
   # CUDA float32, torch.compile (inductor) tests
   OMP_NUM_THREADS=2 KORNIA_TEST_OPTIMIZER=inductor KORNIA_TEST_DEVICE=cuda KORNIA_TEST_DTYPE=float32 \
     python -m pytest tests/augmentation tests/enhance tests/feature tests/filters tests/color tests/geometry \
     tests/losses tests/morphology tests/models tests/core tests/utils -k "dynamo or compile" -n 2

**Why tests are skipped.** Slow tests run only with ``--runslow``: they are 208 and 207 of the skips in the CPU
float16 and bfloat16 rows, 304 in the CUDA float32 + float64 row, 305 in the float16 + bfloat16 row and 28 in the
compile row; in the float32 ``--runslow`` row, all of them pass. A CUDA run without ``--isolate-half-precision`` skips
the tests that pin a half dtype themselves: 623 in each of the CUDA float32 + float64 and float32 ``--runslow`` rows,
and 6 in the compile row. 34 skips in each CPU row need an optional package, tool or local weights, such as OpenCV,
scikit-image or ``uv``; a few do in the CUDA rows. The rest are a handful per row, among them bfloat16 ``So2``/``Se2``,
which PyTorch cannot store without a complex bfloat16 dtype, and two compiled color-conversion checks where CUDA
inductor rounds one pixel below its bound.

**CUDA float32 and float64.** No failures, with and without ``--runslow``.

**CUDA float16 and bfloat16.** Of the 33 failures, 27 are the CUDA counterparts of CPU manifest entries (the other 2
manifest entries pass on CUDA). One failure and the 3 errors are in ``tests/test_conftest.py``, the harness's own tests,
which assume they do not already run inside a half-precision subprocess; they pass in a normal ``--device=cuda`` run.
One is a flaky test rather than a CUDA failure: ``TestHyperplane::test_through_three`` in bfloat16 draws unseeded random
points, and about 2 % of random bfloat16 triples, on CPU and CUDA alike, fall within the collinearity check (a
singular-value ratio of 8 bfloat16 epsilons, 0.0625), so ``Hyperplane.through`` rightly rejects them; it passed in all
three isolated re-runs. The last 4 fail on CUDA only, in every isolated re-run:
``TestRandomRotation3DBackward::test_param`` in bfloat16, with ``degrees`` a tensor of 10, 20 and 30 degrees (the CPU
manifest entries are its ``(10, 20)`` range parameter). This is a bfloat16 limit rather than a CUDA one. The test takes
one SGD step with learning rate 10; its gradient is finite and nonzero, but the range must also move. On CUDA (step
sizes measured on ``26e866e54``) the steps are at most 0.054 degrees, each under half a bfloat16 ulp of its
angle (0.031 at 10 degrees, 0.0625 at 20 and 30 degrees), so no angle changes; the bfloat16 steps match the float32 ones
within 2 %, apart from one of 7e-5 degrees. On CPU the sampled rotations differ, and the step at 10 degrees (0.032 to
0.036 degrees) passes half an ulp. The test checks the sum of the changes, so it passes on CPU, although the other two
angles do not move there either.

**Batch-dependent CUDA kernels.** CUDA picks convolution and matrix-multiply kernels by batch size, so a half-precision
LoFTR, SmallSR or LightGlue gives an image slightly different outputs in a batch than alone; on CPU the batched and
single outputs are bitwise equal. Their batch-consistency tests therefore use a CUDA-only tolerance in half precision
(`#6584 <https://github.com/kornia/kornia/pull/6584>`_). With cuDNN disabled, SmallSR's outputs are bitwise equal on
CUDA too; LightGlue has no convolutions, so there the batch-dependent kernels are its matrix multiplications or
attention. With pretrained weights on Photo Tourism images, the change between a batch and a single image is smaller
than the error of half precision itself, which the "vs float32" columns show (measured on ``e4a444e9b``):

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

**CUDA torch.compile.** No failures. The 6 CUDA-only compile failures measured on ``e4a444e9b`` were fixed by `#6585
<https://github.com/kornia/kornia/pull/6585>`_ and `#6586 <https://github.com/kornia/kornia/pull/6586>`_.

**Open CUDA-only issues that the test suite does not exercise:**

- `#6427 <https://github.com/kornia/kornia/issues/6427>`_: ``OnnxLightGlue`` binds GPU 0 for any CUDA device.
- `#6559 <https://github.com/kornia/kornia/issues/6559>`_: batched SVD of matrices with more than 32 rows makes
  ``solve_pnp_dlt``, ``fit_plane`` and ``find_essential`` up to 200× slower on CUDA than on CPU.

**Fixed upstream:** on PyTorch before 2.15, ``RANSAC(compile=True)`` on CUDA can store a CPU program in the inductor
cache after a CPU compile, and later CUDA processes that share the cache segfault
(`#6278 <https://github.com/kornia/kornia/issues/6278>`_). Reproduced on 2.14.0, not on 2.15.0. On an older
PyTorch, use a separate ``TORCHINDUCTOR_CACHE_DIR`` per device or set ``KORNIA_RANSAC_AOT=0``.

For CUDA half tests, use ``--isolate-half-precision`` so each test gets a fresh process. They are skipped by default
without it; selecting a half dtype alone does not exercise CUDA half support:

.. code-block:: bash

   pixi run test-module tests/color/ --device=cuda --dtype=float16,bfloat16 --isolate-half-precision
