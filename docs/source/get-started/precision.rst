Float16 and bfloat16 support
============================

.. meta::
   :description: Which Kornia modules support float16 and bfloat16 on CPU and CUDA, known limitations, and the latest half-precision test results.

This page documents which kornia modules support half-precision floating-point dtypes
(``torch.float16`` and ``torch.bfloat16``) and what limitations to expect.

The status below comes from the Linux CPU half-precision CI jobs, which run the whole test suite in
``float16`` and in ``bfloat16`` against strict known-failure manifests
(``testing/half_precision_xfails/cpu_float16.txt`` and ``cpu_bfloat16.txt``). The *known failures*
column counts the manifest entries per module (float16 / bfloat16); the remaining work is tracked in
`issue #4153 <https://github.com/kornia/kornia/issues/4153>`_. No CI job covers CUDA or MPS half
precision, so the table says nothing about those backends.

.. list-table:: Half-Precision Support by Module (Linux CPU)
   :header-rows: 1
   :widths: 24 11 11 12 42

   * - Module
     - float16
     - bfloat16
     - Known failures
     - Notes
   * - ``kornia.color``
     - ⚠️ Partial
     - ⚠️ Partial
     - 3 / 9
     - float16: HLS JIT/module and RGB255 round-trip accuracy. bfloat16: Lab, Luv and RGB255 accuracy.
   * - ``kornia.filters``
     - ⚠️ Partial
     - ⚠️ Partial
     - 18 / 9
     - Accuracy misses in Canny magnitudes, discrete Gaussian kernels and Otsu thresholding. On CPU,
       ``fft_conv`` computes its FFTs in float32 and returns the input dtype.
   * - ``kornia.enhance``
     - ✅ Yes
     - ⚠️ Partial
     - 0 / 2
     - bfloat16: ``DiffJPEG`` and ZCA whitening accuracy.
   * - ``kornia.morphology``
     - ✅ Yes
     - ✅ Yes
     - 0 / 0
     -
   * - ``kornia.augmentation``
     - ⚠️ Partial
     - ⚠️ Partial
     - 200 / 56
     - float16: 108 entries are ``CutmixGenerator``, whose Dirichlet sampling rejects float16 parameters; most
       of the rest are ``VideoSequential`` / ``AugmentationSequential`` and 3D-augmentation gradient checks.
       bfloat16: mostly 3D-augmentation gradient checks (28 of 56 entries are ``RandomMotionBlur3D`` /
       ``RandomRotation3D`` backward).
   * - ``kornia.geometry.transform``
     - ⚠️ Partial
     - ⚠️ Partial
     - 43 / 58
     - Linalg steps go through ``_torch_inverse_cast`` / ``_torch_solve_cast``, but rotation matrices,
       affine/perspective warps, the homography warper and 3D crops miss their accuracy checks.
   * - ``kornia.geometry.camera``
     - ⚠️ Partial
     - ⚠️ Partial
     - 13 / 23
     - Pinhole ``cam2pixel`` / ``pixel2cam`` consistency, distortion round trips and ``StereoCamera``
       disparity reprojection. Twelve bfloat16 entries are a test-side assertion that lists only
       float16/float32/float64.
   * - ``kornia.geometry.calibration``
     - ⚠️ Partial
     - ⚠️ Partial
     - 13 / 12
     - ``solve_pnp_dlt()`` explicitly checks that inputs are ``float32`` or ``float64`` and raises otherwise.
       ``undistort_points`` misses its OpenCV reference values.
   * - ``kornia.geometry.epipolar``
     - ⚠️ Partial
     - ⚠️ Partial
     - 58 / 56
     - ``find_fundamental``, ``find_essential``, ``decompose_essential_matrix``, ``motion_from_essential*``
       and ``KRt_from_projection`` raise ``NotImplementedError``: they call ``lu``, ``eigh`` or QR, which have
       no CPU half-precision kernels.
   * - ``kornia.geometry.homography``
     - ⚠️ Partial
     - ⚠️ Partial
     - 11 / 16
     - The DLT solvers run (``_torch_svd_cast`` promotes the SVD to float32) but miss the clean-point accuracy
       checks, including the iterated and line-based variants.
   * - ``kornia.geometry.liegroup``
     - ⚠️ Partial
     - ⚠️ Partial
     - 36 / 130
     - ``So2`` and ``Se2`` use complex tensors: float16 hits missing ``ComplexHalf`` kernels, and most
       bfloat16 ``So2`` / ``Se2`` tests raise because the complex path does not accept bfloat16 (119 entries).
       ``So3`` and ``Se3`` nearly all pass.
   * - ``kornia.geometry.solvers``
     - ⚠️ Partial
     - ⚠️ Partial
     - 2 / 2
     - ``solve_quartic`` accuracy on random quartics and one reference case.
   * - ``kornia.geometry.subpix``
     - ⚠️ Partial
     - ⚠️ Partial
     - 14 / 12
     - ``ConvSoftArgmax3d`` raises because CPU ``avg_pool3d`` has no half-precision kernel; the remaining
       entries are accuracy.
   * - ``kornia.geometry.conversions``
     - ⚠️ Partial
     - ⚠️ Partial
     - 72 / 60
     - Angle-axis, quaternion and rotation-matrix round trips lose accuracy (including ``tests/integration``).
   * - ``kornia.geometry.ransac``
     - ⚠️ Partial
     - ⚠️ Partial
     - 4 / 4
     - The essential and fundamental models raise through the epipolar solvers.
   * - ``kornia.geometry`` (other)
     - ⚠️ Partial
     - ⚠️ Partial
     - 8 / 17
     - Accuracy in boxes, depth and line utilities; bfloat16 ``NamedPose`` construction raises.
   * - ``kornia.image``
     - ⚠️ Partial
     - ⚠️ Partial
     - 4 / 4
     - ``draw_convex_polygon`` fill accuracy.
   * - ``kornia.losses``
     - ⚠️ Partial
     - ⚠️ Partial
     - 3 / 4
     - Dice averaging overflows to inf/NaN in float16; the mutual-information range check fails in both
       dtypes; bfloat16 Dice weighting and total variation miss accuracy. Hausdorff and the photometric losses
       pass.
   * - ``kornia.feature``
     - ✅ Yes
     - ✅ Yes
     - 0 / 0
     - Detectors, descriptors and matchers pass. ``HyNet`` and ``SOSNet`` take their final L2 normalization in
       float32 for a half-precision input, since ``local_response_norm`` has no CPU half kernel for a 4-D
       tensor and its ``1e-10`` guard flushes to zero in float16. Feature *matching* uses a manual ``cdist``
       fallback for both half-precision dtypes. LightGlue's float16 tests are skipped, so that path is not
       measured.
   * - ``kornia.metrics``
     - ✅ Yes
     - ⚠️ Partial
     - 0 / 1
     - bfloat16: ``ssim3d`` accuracy.
   * - ``kornia.models``
     - ⚠️ Partial
     - ⚠️ Partial
     - 9 / 1
     - float16: EfficientViT raises dtype-mismatch errors. bfloat16: RT-DETR RepVGG deployment fusion misses
       accuracy.
   * - ``contrib``, ``core``, ``io``, ``onnx``, ``sensors``, ``tracking``, ``utils``
     - ✅ Yes
     - ⚠️ Partial
     - 0 / 3
     - bfloat16: histogram matching, ``_torch_svd_cast`` and camera-model projection.

Legend
------

- ✅ **Yes** — No known failures in the CPU CI profile for that dtype.
- ⚠️ **Partial** — The module runs, but some tests are known failures. Most are accuracy misses from the limited
  range/precision; the notes name the operations that raise instead.

Test Results
------------

Full test suite (no ``--runslow``). Pass% = passed ÷ (passed + failed); skipped tests and tests marked ``xfail`` in
the source are excluded. The CPU rows come from the scheduled ``main`` CI jobs at commit ``ca5021eb``
(2026-09-14; Linux x86_64, Python 3.11, PyTorch 2.9.1). In the half-precision jobs, *Failed* is the manifest's entry
count: CI reports those tests as strict xfails and turns red if any of them passes or fails differently. The CUDA
rows are a local run at commit ``f8449854`` (2026-09-23; RTX 4090, Python 3.11, PyTorch 2.14.0+cu130), not CI. Every
CUDA failure is re-run on its own (with ``--isolate-half-precision`` for the half dtypes) and counted by that result.
Eight of the CUDA float32 failures are cuDNN TF32 accuracy misses in convolutions
(`#4778 <https://github.com/kornia/kornia/issues/4778>`_); the other 14 are tests that assume CPU behavior
(`#4779 <https://github.com/kornia/kornia/issues/4779>`_).

Reproduce a CPU half-precision row in that environment (the manifest header pins the OS, architecture, Python and
PyTorch versions) with:

.. code-block:: bash

   KORNIA_TEST_OPTIMIZER= pixi run test-module tests/ --verify-known-failures --known-failure-profile=cpu-float16
   KORNIA_TEST_OPTIMIZER= pixi run test-module tests/ --verify-known-failures --known-failure-profile=cpu-bfloat16

The CPU float32 baseline is ``pixi run test-f32``. ``pixi run test-half`` is an unseeded sweep of both dtypes whose
counts can drift slightly from the manifests.

.. list-table::
   :header-rows: 1
   :widths: 32 10 10 10 10

   * - Run
     - Passed
     - Failed
     - Skipped
     - Pass%
   * - CPU float32 *(baseline)*
     - 10398
     - 0
     - 3737
     - **100.0%**
   * - CPU float16
     - 9795
     - 522
     - 3821
     - **94.9%**
   * - CPU bfloat16
     - 9849
     - 512
     - 3774
     - **95.1%**
   * - CUDA float32 *(baseline)*
     - 12193
     - 22
     - 3957
     - **99.8%**
   * - CUDA float16 *(KORNIA_TEST_IN_SUBPROCESS=1)*
     - 11738
     - 451
     - 3983
     - **96.3%**
   * - CUDA bfloat16 *(KORNIA_TEST_IN_SUBPROCESS=1)*
     - 11736
     - 500
     - 3936
     - **95.9%**

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
