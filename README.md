<div align="center">
<p align="center">
  <img width="55%" src="https://github.com/kornia/data/raw/main/kornia_banner_pixie.png" />
</p>

---

English | [简体中文](README_zh-CN.md)

<!-- prettier-ignore -->
<a href="https://kornia.readthedocs.io">Docs</a> •
<a href="https://colab.sandbox.google.com/github/kornia/tutorials/blob/master/nbs/hello_world_tutorial.ipynb">Try it Now</a> •
<a href="https://kornia.github.io/tutorials/">Tutorials</a> •
<a href="https://github.com/kornia/kornia-examples">Examples</a> •
<a href="https://kornia.github.io//kornia-blog">Blog</a> •
<a href="https://discord.gg/HfnywwpBnD">Community</a>

[![PyPI version](https://badge.fury.io/py/kornia.svg)](https://pypi.org/project/kornia)
[![Downloads](https://static.pepy.tech/badge/kornia)](https://pepy.tech/project/kornia)
[![star](https://gitcode.com/kornia/kornia/star/badge.svg)](https://gitcode.com/kornia/kornia)
[![Discord](https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/HfnywwpBnD)
[![Twitter](https://img.shields.io/twitter/follow/kornia_foss?style=social)](https://twitter.com/kornia_foss)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</p>
</div>

**Kornia** is a differentiable computer vision library that provides a rich set of differentiable image processing and geometric vision algorithms. Built on top of [PyTorch](https://pytorch.org), Kornia integrates seamlessly into existing AI workflows, allowing you to leverage powerful [batch transformations](), [auto-differentiation]() and [GPU acceleration](). Whether you're working on image transformations, augmentations, or AI-driven image processing, Kornia equips you with the tools you need to bring your ideas to life.

> **📢 Direction**: Kornia is becoming the **reference implementation and executable specification** for differentiable computer vision and geometry in the PyTorch ecosystem — explicit conventions, conformance tests, and honest benchmarks over API growth. Read the [Roadmap](ROADMAP.md).

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
| **Augmentation**            | - Random cropping, Erasing<br> - Random geometric transformations (Affine, flipping, Fish Eye, Perspecive, Thin plate spline, Elastic)<br>- Random noises (Gaussian, Median, Motion, Box, Rain, Snow, Salt and Pepper)<br>- Random color jittering (Contrast, Brightness, CLAHE, Equalize, Gamma, Hue, Invert, JPEG, Plasma, Posterize, Saturation, Sharpness, Solarize)<br> - Random MixUp, CutMix, Mosaic, Transplantation, etc.                  |
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

## Half-Precision Support

| Module | float16 | bfloat16 | Notes |
|--------|:-------:|:--------:|-------|
| `kornia.color` | ⚠️ | ⚠️ | Most conversions work for both; FFT-based ops may fail |
| `kornia.filters` | ⚠️ | ⚠️ | Basic filters work; FFT-based ops may fail on CUDA |
| `kornia.enhance` | ⚠️ | ⚠️ | Histogram eq / gamma / ZCA work (linalg ops use cast helpers) |
| `kornia.morphology` | ✅ | ✅ | Conv/pool ops; `top_hat` / `bottom_hat` / `gradient` also subtract two dilation/erosion results, so bfloat16 loses ~0.4% relative accuracy — within kornia's own bfloat16 tolerance, though 6 tests override it with a tighter one ([#4081](https://github.com/kornia/kornia/issues/4081)) |
| `kornia.augmentation` | ⚠️ | ⚠️ | Most ops work; precision-sensitive transforms may be inaccurate |
| `kornia.geometry.transform` | ⚠️ | ⚠️ | Affine/warp/resize work via cast helpers; thin-plate spline may fail |
| `kornia.geometry.camera` | ⚠️ | ⚠️ | Pinhole model and most camera ops work; `StereoCamera` accepts both |
| `kornia.geometry.calibration` | ❌ | ❌ | Explicitly accepts float32/float64 only (PnP solver) |
| `kornia.geometry.epipolar` | ⚠️ | ⚠️ | SVD/inverse use cast helpers; both dtypes work |
| `kornia.geometry.homography` | ⚠️ | ⚠️ | Uses `_torch_svd_cast` — both dtypes work via casting |
| `kornia.geometry.liegroup` | ⚠️ | ⚠️ | Most ops work via cast helpers; some linalg paths may fail |
| `kornia.geometry.solvers` | ⚠️ | ⚠️ | Uses `_torch_solve_cast` — both dtypes work via casting |
| `kornia.geometry.subpix` | ⚠️ | ⚠️ | Soft-argmax works; precision-sensitive ops may be inaccurate |
| `kornia.losses` | ⚠️ | ⚠️ | Photometric losses work; linalg-based losses may not |
| `kornia.feature` | ⚠️ | ⚠️ | Detectors/descriptors work; matching uses manual cdist fallback |
| `kornia.metrics` | ⚠️ | ⚠️ | Pixel-level metrics work; linalg-based metrics may not |
| `kornia.models` | ⚠️ | ⚠️ | Conv-based models work; attention-based models may have dtype mismatches |

✅ Supported &nbsp; ⚠️ Partial &nbsp; ❌ Not supported

**Test results:**

| Run | Passed | Failed | Skipped | Pass% | Measured |
|-----|-------:|-------:|--------:|------:|----------|
| CPU float32 *(baseline)* | 8499 | 0 | 3535 | **100.0%** | `4ab79c78`, 2026-08-29 |
| CPU float16 | 7751 | 689 | 3595 | **91.8%** | `4ab79c78`, 2026-08-29 |
| CPU bfloat16 | 7794 | 695 | 3545 | **91.8%** | `4ab79c78`, 2026-08-29 |
| CUDA float32 *(baseline)* | 7634 | 3 | 3280 | **99.9%** | `6131e98`, 2026-03-21 |
| CUDA float16 *(KORNIA_TEST_IN_SUBPROCESS=1)* | 6727 | 643 | 3556 | **91.3%** | `6131e98`, 2026-03-21 |
| CUDA bfloat16 *(KORNIA_TEST_IN_SUBPROCESS=1)* | 6695 | 713 | 3518 | **90.4%** | `6131e98`, 2026-03-21 |

Reproduce the two CPU half rows with `pixi run test-half` and the CPU float32 baseline with `pixi run test-f32`
(`test-half` pins `KORNIA_TEST_DTYPE` to `float16,bfloat16`, so it cannot produce the baseline). The half-precision
suite is not run in CI (see [#4070](https://github.com/kornia/kornia/issues/4070)), so these numbers are refreshed
by hand.

See the [full precision guide](https://kornia.readthedocs.io/en/stable/get-started/precision.html) for details.

## Sponsorship

Kornia is an open-source project that is developed and maintained by volunteers. Whether you're using it for research or commercial purposes, consider sponsoring or collaborating with us. Your support will help ensure Kornia's growth and ongoing innovation. Reach out to us today and be a part of shaping the future of this exciting initiative!

<a href="https://opencollective.com/kornia/donate" target="_blank">
  <img src="https://opencollective.com/webpack/donate/button@2x.png?color=blue" width=300 />
</a>

## Installation

[![PyPI python](https://img.shields.io/pypi/pyversions/kornia)](https://pypi.org/project/kornia)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

### From pip

  ```bash
  pip install kornia
  ```

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
  pixi run install

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

## Multi-framework support

You can now use Kornia with [TensorFlow](https://www.tensorflow.org/), [JAX](https://jax.readthedocs.io/en/latest/index.html), and [NumPy](https://numpy.org/). See [Multi-Framework Support](docs/source/get-started/multi-framework-support.rst) for more details.

```python
import kornia

tf_kornia = kornia.to_tensorflow()
```

<p align="center">
  Powered by
  <a href="https://github.com/ivy-llc/ivy" target="_blank">
    <div class="dark-light" style="display: block;" align="center">
      <img class="dark-light" width="15%" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/ivy-long.svg"/>
    </div>
  </a>
</p>

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
