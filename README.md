<div align="center">
<p align="center">
  <img width="55%" src="https://github.com/kornia/data/raw/main/kornia_banner_pixie.png" />
</p>

---

English | [简体中文](README_zh-CN.md)

<p align="center">
  <a href="https://www.kornia.org/playground/"><img alt="Try in your browser" src="https://img.shields.io/badge/%E2%96%B6%20Try%20in%20your%20browser-F5B301?style=for-the-badge&labelColor=F5B301"></a>
  <a href="https://www.kornia.org"><img alt="Website" src="https://img.shields.io/badge/Website-0B1A2B?style=for-the-badge"></a>
  <a href="https://kornia.readthedocs.io"><img alt="Docs" src="https://img.shields.io/badge/Docs-0B1A2B?style=for-the-badge"></a>
</p>

<!-- prettier-ignore -->
<a href="https://kornia.readthedocs.io">Docs</a> •
<a href="https://colab.sandbox.google.com/github/kornia/tutorials/blob/master/nbs/hello_world_tutorial.ipynb">Try it Now</a> •
<a href="https://www.kornia.org/tutorials/">Tutorials</a> •
<a href="https://github.com/kornia/kornia-examples">Examples</a> •
<a href="https://www.kornia.org/kornia-blog/">Blog</a> •
<a href="https://discord.gg/HfnywwpBnD">Community</a>

[![PyPI version](https://badge.fury.io/py/kornia.svg)](https://pypi.org/project/kornia)
[![Downloads](https://static.pepy.tech/badge/kornia)](https://pepy.tech/project/kornia)
[![star](https://gitcode.com/kornia/kornia/star/badge.svg)](https://gitcode.com/kornia/kornia)
[![Discord](https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/HfnywwpBnD)
[![Twitter](https://img.shields.io/twitter/follow/kornia_foss?style=social)](https://twitter.com/kornia_foss)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</p>
</div>

**Kornia** is a differentiable computer vision library that provides a rich set of differentiable image processing and geometric vision algorithms. Built on top of [PyTorch](https://pytorch.org), Kornia integrates seamlessly into existing AI workflows, allowing you to leverage powerful [batch transformations](https://kornia.readthedocs.io/en/latest/get-started/gpu-acceleration.html), [auto-differentiation](https://kornia.readthedocs.io/en/latest/get-started/differentiability.html) and [GPU acceleration](https://kornia.readthedocs.io/en/latest/get-started/gpu-acceleration.html). Whether you're working on image transformations, augmentations, or AI-driven image processing, Kornia equips you with the tools you need to bring your ideas to life.

> **📢 Direction**: Kornia is becoming the **reference implementation and executable specification** for differentiable computer vision and geometry in the PyTorch ecosystem — explicit conventions, conformance tests, and honest benchmarks over API growth. Read the [Roadmap](ROADMAP.md).

## When to use Kornia

Kornia pays off where a hand-written version gets long or fragile, or where the alternatives leave the GPU:

- **Breadth.** 1,200+ public differentiable operators, most with no torchvision equivalent: camera models and
  epipolar geometry, minimal solvers and RANSAC, feature detection, description and matching, edge detection,
  morphology, Lab and Bayer color, image losses and metrics. torchvision overlaps mainly on resizing, cropping,
  flipping, color jitter and blur.
- **Per-sample parameters in one batched call.** `gaussian_blur2d(x, (9, 9), sigma)` takes a `(B, 2)` `sigma`;
  `rotate` takes a `(B,)` angle, `warp_affine` / `warp_perspective` take `(B, 2, 3)` / `(B, 3, 3)` matrices and
  `adjust_brightness` a `(B,)` factor. torchvision's functional ops apply one parameter per call.
- **Batched minimal solvers and robust estimation.** `find_fundamental(x1, x2, method="7POINT")` solves `(B, 7, 2)`
  samples and returns every real solution as `(B, 3, 3, 3)`; the 8-point, 5-point essential (`find_essential`), DLT
  homography and PnP (`solve_pnp_dlt`) solvers are there too, and `RANSAC` (homography, fundamental and essential
  models; `compile=True` on PyTorch ≥ 2.14) runs the whole loop on the device.
- **Differentiable geometry.** Depth to 3D and back (`depth_to_3d_v2`, `unproject_points`, `project_points`),
  `warp_frame_depth`, `remap`, camera models with Brown–Conrady and Kannala–Brandt distortion, and Lie groups (`So3`,
  `Se3`) keep gradients with respect to depth, pose and camera parameters.
- **Annotated augmentation.** `AugmentationSequential` applies one sampled transform to images, masks, boxes and
  keypoints together, in 2D and 3D, and can invert it. On CUDA, augmentations move their sampled parameters to the
  GPU without blocking it, and a sync-guard test keeps every public augmentation free of avoidable host
  synchronizations.
- **Local features on the GPU.** SIFT and HardNet, DISK, ALIKED, XFeat, DeDoDe, LoFTR and LightGlue, with
  mutual-nearest-neighbour and ratio-test matchers.
- **Morphology** with arbitrary structuring elements on batched tensors.
- **Compiles and exports.** Nearly all operators run under `torch.compile` and match eager, and over 1,000 of the
  1,215 probed configurations export to ONNX; the result for every operator is on the
  [export support page](https://kornia.readthedocs.io/en/latest/get-started/export-support.html).
- **Fast where it is designed to be: batched tensors on a GPU.** In the published 0.9.0 flagship benchmarks (RTX 4090,
  batch 32), many filters, warps, feature-matching steps and image metrics run 10× to over 100× faster than
  per-image OpenCV or scikit-image on the CPU. On the small overlap with torchvision on the same GPU, Kornia is
  faster for hue adjustment, color jitter and Gaussian blur and slower for flips, crops and brightness; OpenCV wins
  single-image `uint8` CPU work. Both wins and losses are published on the
  [performance page](https://kornia.readthedocs.io/en/latest/get-started/performance.html).

These operate on tensors on the input's device and keep autograd, with no NumPy round trip. For a single
fixed-parameter operation, such as one blur or one resize, a few lines of PyTorch or torchvision are just as good.

## Key Components
1. **Differentiable Image Processing**<br>
  Kornia provides a comprehensive suite of image processing operators, all differentiable and ready to integrate into deep learning pipelines.
    - **Filters**: Gaussian, Sobel, Median, Box Blur, etc.
    - **Transformations**: Affine, Homography, Perspective, etc.
    - **Enhancements**: Histogram Equalization, CLAHE, Gamma Correction, etc.
    - **Edge Detection**: Canny, Laplacian, Sobel, etc.
    - ... check our [docs](https://kornia.readthedocs.io) for more.
2. **Advanced Augmentations**<br>
Perform powerful data augmentation with Kornia’s built-in functions, ideal for training AI models with complex augmentation pipelines.
    - **Augmentation Pipeline**: AugmentationSequential, PatchSequential, VideoSequential, etc.
    - **Automatic Augmentation**: AutoAugment, RandAugment, TrivialAugment.
3. **AI Models**<br>
Leverage pre-trained AI models optimized for a variety of vision tasks, all within the Kornia ecosystem.
    - **Face Detection**: YuNet
    - **Feature Matching**: LoFTR, LightGlue
    - **Feature Descriptor**: DISK, DeDoDe, SOLD2
    - **Segmentation**: SAM
    - **Classification**: MobileViT, VisionTransformer.

<details>
<summary>See here for some of the methods that we support! (>500 ops in total !)</summary>

| **Category**               | **Methods/Models**                                                                                                   |
|----------------------------|---------------------------------------------------------------------------------------------------------------------|
| **Image Processing**        | - Color conversions (RGB, Grayscale, HSV, etc.)<br>- Geometric transformations (Affine, Homography, Resizing, etc.)<br>- Filtering (Gaussian blur, Median blur, etc.)<br>- Edge detection (Sobel, Canny, etc.)<br>- Morphological operations (Erosion, Dilation, etc.)                                 |
| **Augmentation**            | - Random cropping, Erasing<br> - Random geometric transformations (Affine, flipping, Fish Eye, Perspective, Thin plate spline, Elastic)<br>- Random noises (Gaussian, Median, Motion, Box, Rain, Snow, Salt and Pepper)<br>- Random color jittering (Contrast, Brightness, CLAHE, Equalize, Gamma, Hue, Invert, JPEG, Plasma, Posterize, Saturation, Sharpness, Solarize)<br> - Random MixUp, CutMix, Mosaic, Transplantation, etc.                  |
| **Feature Detection**       | - Detector (Harris, GFTT, Hessian, DoG, KeyNet, DISK and DeDoDe)<br> - Descriptor (SIFT, HardNet, TFeat, HyNet, SOSNet, and LAFDescriptor)<br>- Matching (nearest neighbor, mutual nearest neighbor, geometrically aware matching, AdaLAM LightGlue, and LoFTR)                    |
| **Geometry**                | - Camera models and calibration<br>- Stereo vision (epipolar geometry, disparity, etc.)<br>- Homography estimation<br>- Depth estimation from disparity<br>- 3D transformations                |
| **Deep Learning Layers**    | - Custom convolution layers<br>- Recurrent layers for vision tasks<br>- Loss functions (e.g., SSIM, PSNR, etc.)<br>- Vision-specific optimizers                                        |
| **Photometric Functions**   | - Photometric loss functions<br>- Photometric augmentations                                                                                           |
| **Filtering**               | - Bilateral filtering<br>- DexiNed<br>- Dissolving<br>- Guided Blur<br>- Laplacian<br>- Gaussian<br>- Non-local means<br>- Sobel<br>- Unsharp masking                                                                                            |
| **Color**                   | - Color space conversions<br>- Brightness/contrast adjustment<br>- Gamma correction                                                                       |
| **Stereo Vision**           | - Disparity estimation<br>- Depth estimation<br>- Rectification                                                                                           |
| **Image Registration**      | - Affine and homography-based registration<br>- Image alignment using feature matching                                                                     |
| **Pose Estimation**         | - Essential and Fundamental matrix estimation<br>- PnP problem solvers<br>- Pose refinement                                                                |
| **Optical Flow**            | - Farneback optical flow<br>- Dense optical flow<br>- Sparse optical flow                                                                                  |
| **3D Vision**               | - Depth estimation<br>- Point cloud operations<br>                                                                |
| **Image Denoising**         | - Gaussian noise removal<br>- Poisson noise removal                                                                                                        |
| **Edge Detection**          | - Sobel operator<br>- Canny edge detection                                                                                                                 |                                               |
| **Transformations**         | - Rotation<br>- Translation<br>- Scaling<br>- Shearing                                                                                                     |
| **Loss Functions**          | - SSIM (Structural Similarity Index Measure)<br>- PSNR (Peak Signal-to-Noise Ratio)<br>- Cauchy<br>- Charbonnier<br>- Depth Smooth<br>- Dice<br>- Hausdorff<br>- Tversky<br>- Welsch<br>                                   |                                                                                             |
| **Morphological Operations**| - Dilation<br>- Erosion<br>- Opening<br>- Closing                                                                                                          |

</details>

## CUDA and Half-Precision Support

The module table below counts entries in the Linux CPU known-failure manifests
([`cpu_float16.txt`](testing/half_precision_xfails/cpu_float16.txt),
[`cpu_bfloat16.txt`](testing/half_precision_xfails/cpu_bfloat16.txt)), pinned to Python 3.11 and PyTorch 2.14.0.
There are **243 float16 and 256 bfloat16 recorded failures**. These are parametrized test cases, not counts of
unsupported functions. The remaining work is tracked in [#4153](https://github.com/kornia/kornia/issues/4153).
No CI job covers CUDA or MPS half precision; CPU results do not establish support on those backends.

| Module | float16 | bfloat16 | Known failures (fp16 / bf16) | Notes |
|--------|:-------:|:--------:|:----------------------------:|-------|
| `kornia.color` | ✅ | ✅ | 0 / 0 | Lab/Luv conversions and `rgb255_to_normals` compute half inputs in float32 and return the input dtype |
| `kornia.filters` | ✅ | ✅ | 0 / 0 | Canny, Gaussian kernels/blur and CPU FFT convolution compute half inputs in float32 and return the input dtype |
| `kornia.enhance` | ✅ | ✅ | 0 / 0 | ZCA keeps half-input fitted state in float32; whitening and JPEG return the input dtype after float32 computation |
| `kornia.morphology` | ✅ | ✅ | 0 / 0 |  |
| `kornia.augmentation` | ⚠️ | ⚠️ | 59 / 48 | float16: container autocast/runtime errors and accuracy; both dtypes: 3D backward and random-generator accuracy |
| `kornia.geometry.transform` | ⚠️ | ⚠️ | 31 / 31 | Rotation matrices, affine/perspective warps and homography-warper accuracy |
| `kornia.geometry.camera` | ⚠️ | ⚠️ | 13 / 23 | Projection, distortion round trips and stereo reprojection; includes test-side dtype assertions |
| `kornia.geometry.calibration` | ⚠️ | ⚠️ | 13 / 12 | solve_pnp_dlt rejects half inputs; undistort_points accuracy |
| `kornia.geometry.epipolar` | ⚠️ | ⚠️ | 36 / 36 | Essential-matrix decomposition/motion and KRt_from_projection lack CPU half kernels; fundamental-matrix accuracy |
| `kornia.geometry.homography` | ⚠️ | ⚠️ | 10 / 16 | DLT solver accuracy, including iterated and line-based variants |
| `kornia.geometry.liegroup` | ✅ | ✅ | 0 / 0 | So2/Se2 reject bfloat16 rotation construction: PyTorch has no complex bfloat16 dtype; these expected errors/skips are outside the manifests |
| `kornia.geometry.solvers` | ✅ | ✅ | 0 / 0 |  |
| `kornia.geometry.subpix` | ⚠️ | ⚠️ | 6 / 3 | SpatialSoftArgmax2d and ConvQuadInterp3d accuracy; bfloat16: render_gaussian2d accuracy |
| `kornia.geometry.conversions` | ⚠️ | ⚠️ | 65 / 65 | Angle-axis, quaternion and rotation-matrix accuracy; includes tests/integration/test_conversions.py |
| `kornia.geometry.ransac` | ⚠️ | ⚠️ | 2 / 2 | Essential-model fitting raises in epipolar solvers |
| `kornia.geometry (other)` | ⚠️ | ⚠️ | 6 / 13 | Box-transform and depth accuracy |
| `kornia.image` | ⚠️ | ⚠️ | 2 / 2 | Circle/ellipse polygon-fill comparisons use vertices and reference masks computed in half precision |
| `kornia.losses` | ✅ | ⚠️ | 0 / 2 | bfloat16: total-variation accuracy |
| `kornia.feature` | ✅ | ✅ | 0 / 0 | Matching uses a manual cdist fallback for half dtypes; LightGlue's float16 tests are skipped |
| `kornia.metrics` | ✅ | ✅ | 0 / 0 |  |
| `kornia.models` | ✅ | ⚠️ | 0 / 1 | bfloat16: RT-DETR RepVGG fusion accuracy |
| `contrib, core, io, onnx, sensors, tracking, utils` | ✅ | ⚠️ | 0 / 2 | bfloat16: histogram matching and _torch_svd_cast accuracy |

✅ No entries in the CPU known-failure manifest &nbsp; ⚠️ Has recorded CPU failures.
Neither symbol promises complete dtype support: skips, expected exceptions and source-level xfails are outside
these counts. Some operators compute internally in float32 or float64 and return the input dtype.

**Observed CPU CI outcomes:**

The [scheduled run at `93f1376e`, 2026-10-21](https://github.com/kornia/kornia/actions/runs/39524889979)
used Linux x86_64, Python 3.11 and PyTorch 2.14.0, without slow or compile tests:

| Run | Passed | Failed | Skipped | XFAIL (all) | XPASS | Strict manifest entries |
|-----|-------:|-------:|--------:|------------:|------:|------------------------:|
| [CPU float16](https://github.com/kornia/kornia/actions/runs/39524889979/job/118977300500) | 23905 | 0 | 5132 | 284 | 4 | 250 |
| [CPU bfloat16](https://github.com/kornia/kornia/actions/runs/39524889979/job/118977300410) | 23726 | 0 | 5286 | 309 | 4 | 274 |

Each job also deselected 3 ONNX conformance tests. XFAIL totals include both the strict manifest entries and
source-level expected failures. The 4 XPASSes in each successful job are non-strict expected failures that passed;
a passing **manifest** entry makes verification fail, as do a changed exception, a skipped entry or a stale node ID.
A green job therefore means the recorded baseline reproduced, even though known failures and skips remain.
These totals describe the linked revision; the module table describes the checked-in manifests.

Verify the complete CPU profiles in the pinned environment with:

```bash
KORNIA_TEST_OPTIMIZER= pixi run test-module tests/ --verify-known-failures --known-failure-profile=cpu-float16
KORNIA_TEST_OPTIMIZER= pixi run test-module tests/ --verify-known-failures --known-failure-profile=cpu-bfloat16
```

`pixi run test-half` is an unseeded CPU sweep, without the strict profile baseline. See
[TESTING.md](TESTING.md#7-half-precision-dtypes-float16--bfloat16) for focused checks and baseline maintenance.

**CUDA support and measurement limits:**

Kornia accepts CUDA tensors for GPU acceleration, subject to each operator's dtype and backend limitations.
There is no blocking CUDA CI job or checked-in CUDA known-failure manifest, so the CPU xfail counts above do not
provide a current CUDA pass rate. The previous local measurements are retained here as historical evidence only:

| Historical run (`f8449854`, 2026-09-23) | Passed | Failed | Skipped |
|------------------------------------------------|-------:|-------:|--------:|
| CUDA float32 | 12193 | 22 | 3957 |
| CUDA float16 | 11738 | 451 | 3983 |
| CUDA bfloat16 | 11736 | 500 | 3936 |

These used an RTX 4090, Python 3.11 and PyTorch 2.14.0+cu130. Source-marked xfails were excluded. Half tests ran
in-process with `KORNIA_TEST_IN_SUBPROCESS=1`, followed by isolated retries of failures; two homography tests were
deselected. They are not a measurement of the current branch. Those two tests (`TestFindHomographyDLT::test_nocrash`
and `test_nocrash_lu`) now pass locally in all three CUDA dtypes at `649b3f86d`; a fresh full CUDA sweep is needed
before quoting new totals.

For CUDA half tests, use `--isolate-half-precision` so each test gets a fresh process. They are skipped by default
without it; selecting a half dtype alone does not exercise CUDA half support:

```bash
pixi run test-module tests/color/ --device=cuda --dtype=float16,bfloat16 --isolate-half-precision
```

See the [full precision guide](https://kornia.readthedocs.io/en/latest/get-started/precision.html) for details.

## Sponsorship

Kornia is an open-source project that is developed and maintained by volunteers. Whether you're using it for research or commercial purposes, consider sponsoring or collaborating with us. Your support will help ensure Kornia's growth and ongoing innovation. Reach out to us today and be a part of shaping the future of this exciting initiative!

<a href="https://opencollective.com/kornia/donate" target="_blank">
  <img src="https://opencollective.com/webpack/donate/button@2x.png?color=blue" width=300 />
</a>

## Installation

[![PyPI python](https://img.shields.io/pypi/pyversions/kornia)](https://pypi.org/project/kornia)
[![pytorch](https://img.shields.io/badge/PyTorch_2.5.1+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

### From pip

  ```bash
  pip install kornia
  ```

Some features (ONNX, Stable Diffusion dissolving) need extra packages; see [Optional extras](https://kornia.readthedocs.io/en/latest/get-started/installation.html#optional-extras).

<details>
  <summary>Other installation options</summary>

#### From source with editable mode

  ```bash
  pip install -e .
  ```

#### For development with Pixi (Recommended)

For development, Kornia uses [pixi](https://pixi.sh) for fast Python package management and environment management. The project includes a `pixi.toml` configuration file for reproducible dependency management.

  ```bash
  # Install pixi (if not already installed)
  curl -fsSL https://pixi.sh/install.sh | bash

  # Create the Pixi environment and install development dependencies
  pixi install
  pixi run -e default install

  # Run tests
  pixi run test

  # For CUDA development
  pixi run -e cuda install
  pixi run -e cuda test-cuda
  ```

These commands set up a complete development environment with all dependencies. For more details on dependency management and available tasks, see [CONTRIBUTING.md](CONTRIBUTING.md).

#### From Github url (latest version)

  ```bash
  pip install git+https://github.com/kornia/kornia
  ```

</details>

## Quick Start

Kornia is not just another computer vision library — it's your gateway to effortless Computer Vision and AI.

<details>
<summary>Get started with Kornia image transformation and augmentation!</summary>

```python
import numpy as np
import kornia_rs as kr

from kornia.augmentation import AugmentationSequential, RandomAffine, RandomBrightness
from kornia.filters import StableDiffusionDissolving

# Load and prepare your image
img: np.ndarray = kr.read_image_any("img.jpeg")
img = kr.resize(img, (256, 256), interpolation="bilinear")

# alternatively, load image with PIL
# img = Image.open("img.jpeg").resize((256, 256))
# img = np.array(img)

img = np.stack([img] * 2)  # batch images

# Define an augmentation pipeline
augmentation_pipeline = AugmentationSequential(RandomAffine((-45.0, 45.0), p=1.0), RandomBrightness((0.0, 1.0), p=1.0))

# Leveraging StableDiffusion models
dslv_op = StableDiffusionDissolving()

img = augmentation_pipeline(img)
dslv_op(img, step_number=500)

dslv_op.save("Kornia-enhanced.jpg")
```

</details>

<details>
<summary>Find out Kornia ONNX models with ONNXSequential!</summary>

```python
import numpy as np
from kornia.onnx import ONNXSequential

# Chain ONNX models from HuggingFace repo and your own local model together
onnx_seq = ONNXSequential(
    "hf://operators/kornia.geometry.transform.flips.Hflip",
    "hf://models/kornia.models.detection.rtdetr_r18vd_640x640",  # Or you may use "YOUR_OWN_MODEL.onnx"
)
# Prepare some input data
input_data = np.random.randn(1, 3, 384, 512).astype(np.float32)
# Perform inference
outputs = onnx_seq(input_data)
# Print the model outputs
print(outputs)

# Export a new ONNX model that chains up all three models together!
onnx_seq.export("chained_model.onnx")
```
</details>

## Call For Contributors

If kornia is useful to you and you would like to help, contributions of many kinds are welcome: code, bug reports, benchmarks, documentation, questions, answers, and examples. The maintainers have limited time, so we cannot promise that every proposal or pull request will be reviewed or merged.

### Strengthen the Core (Priority)

Kornia's differentiated value is its geometry core: warping and sampling, homographies, cameras, epipolar geometry, rotations and Lie groups, and geometry-consistent augmentation. The highest-impact contributions make that core more trustworthy — see the [Roadmap](ROADMAP.md) for the full picture. Great entry points:

- **Benchmark results from your hardware**: run the benchmark suite with `--contribute` and send the JSON — CUDA numbers from diverse GPUs are especially useful.
- **Convention pinning tests and conformance vectors** for core geometry operations.
- **Corrective error messages**: upgrade bare shape assertions into errors that state what was wrong, what was expected, and which convention applies.
- **Classical vision in the core domain**: camera intrinsic calibration, fiducial markers (ArUco/ChArUco), classical tracking, dense stereo, and Hough transforms. An early design discussion can be useful for work of this size.

See the [Roadmap's contributor areas](ROADMAP.md#areas-seeking-contributors) for more project context.

### AI Models

The model zoo is currently **frozen for expansion** while maintainer bandwidth concentrates on the core. Shipped models (LoFTR, LightGlue, DISK, DeDoDe, SAM, and friends) stay available and maintained, and model work approved before the freeze (Efficient LoFTR, SANDesc) will be completed under its existing scope. New integrations — including VLM/VLA models — require a named maintainer sponsor who accepts ongoing ownership of the integration; a contributor implementation alone cannot reopen the surface. See the [Roadmap](ROADMAP.md#guiding-themes) for the reasoning and the reopen condition.

### Documentation And Tutorial Optimization

Kornia's foundation lies in its extensive collection of classic computer vision operators, providing robust tools for image processing, feature extraction, and geometric transformations. We continuously seek for contributors to help us improve our documentation and present nice tutorials to our users.

## Cite

If you are using kornia in your research-related documents, it is recommended that you cite the paper. See more in [CITATION](./CITATION.md).

  ```bibtex
  @inproceedings{eriba2019kornia,
    author    = {E. Riba, D. Mishkin, D. Ponsa, E. Rublee and G. Bradski},
    title     = {Kornia: an Open Source Differentiable Computer Vision Library for PyTorch},
    booktitle = {Winter Conference on Applications of Computer Vision},
    year      = {2020},
    url       = {https://arxiv.org/pdf/1910.02190.pdf}
  }
  ```

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for our social contract, development setup, and technical guidelines. Participation is subject to the [Code of Conduct](./CODE_OF_CONDUCT.md).

## Community

- **Discord:** talk with people who use and develop kornia. [Join the server](https://discord.gg/HfnywwpBnD)
- **GitHub Issues:** report bugs and propose concrete changes. [Open an issue](https://github.com/kornia/kornia/issues/new/choose)
- **GitHub Discussions:** ask questions and discuss implementations, research, and ideas. [Join a discussion](https://github.com/kornia/kornia/discussions)

<a href="https://github.com/Kornia/kornia/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Kornia/kornia" width="60%" />
</a>

Made with [contrib.rocks](https://contrib.rocks).

## License

Kornia is released under the Apache 2.0 license. See the [LICENSE](./LICENSE) file for more information.
