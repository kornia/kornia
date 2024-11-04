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
<a href="https://join.slack.com/t/kornia/shared_invite/zt-csobk21g-CnydWe5fmvkcktIeRFGCEQ">Community</a>

[![PyPI version](https://badge.fury.io/py/kornia.svg)](https://pypi.org/project/kornia)
[![Downloads](https://static.pepy.tech/badge/kornia)](https://pepy.tech/project/kornia)
[![Slack](https://img.shields.io/badge/Slack-4A154B?logo=slack&logoColor=white)](https://join.slack.com/t/kornia/shared_invite/zt-csobk21g-2AQRi~X9Uu6PLMuUZdvfjA)
[![Twitter](https://img.shields.io/twitter/follow/kornia_foss?style=social)](https://twitter.com/kornia_foss)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENCE)

</p>
</div>

**Kornia** is a differentiable computer vision library that provides a rich set of differentiable image processing and geometric vision algorithms. Built on top of [PyTorch](https://pytorch.org), Kornia integrates seamlessly into existing AI workflows, allowing you to leverage powerful [batch transformations](), [auto-differentiation]() and [GPU acceleration](). Whether you’re working on image transformations, augmentations, or AI-driven image processing, Kornia equips you with the tools you need to bring your ideas to life.

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
| **3D Vision**               | - Depth estimation<br>- Point cloud operations<br>- Nerf<br>                                                                |
| **Image Denoising**         | - Gaussian noise removal<br>- Poisson noise removal                                                                                                        |
| **Edge Detection**          | - Sobel operator<br>- Canny edge detection                                                                                                                 |                                               |
| **Transformations**         | - Rotation<br>- Translation<br>- Scaling<br>- Shearing                                                                                                     |
| **Loss Functions**          | - SSIM (Structural Similarity Index Measure)<br>- PSNR (Peak Signal-to-Noise Ratio)<br>- Cauchy<br>- Charbonnier<br>- Depth Smooth<br>- Dice<br>- Hausdorff<br>- Tversky<br>- Welsch<br>                                   |                                                                                             |
| **Morphological Operations**| - Dilation<br>- Erosion<br>- Opening<br>- Closing                                                                                                          |

</details>

## Sponsorship

Kornia is an open-source project that is developed and maintained by volunteers. Whether you're using it for research or commercial purposes, consider sponsoring or collaborating with us. Your support will help ensure Kornia's growth and ongoing innovation. Reach out to us today and be a part of shaping the future of this exciting initiative!

<a href="https://opencollective.com/kornia/donate" target="_blank">
  <img src="https://opencollective.com/webpack/donate/button@2x.png?color=blue" width=300 />
</a>

## Installation

[![PyPI python](https://img.shields.io/pypi/pyversions/kornia)](https://pypi.org/project/kornia)
[![pytorch](https://img.shields.io/badge/PyTorch_1.9.1+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

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
augmentation_pipeline = AugmentationSequential(
    RandomAffine((-45., 45.), p=1.),
    RandomBrightness((0.,1.), p=1.)
)

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

Are you passionate about computer vision, AI, and open-source development? Join us in shaping the future of Kornia! We are actively seeking contributors to help expand and enhance our library, making it even more powerful, accessible, and versatile. Whether you're an experienced developer or just starting, there's a place for you in our community.

### Accessible AI Models

We are excited to announce our latest advancement: a new initiative designed to seamlessly integrate lightweight AI models into Kornia.
We aim to run any models as smooth as big models such as StableDiffusion, to support them well in many perspectives.
We have already included a selection of lightweight AI models like [YuNet (Face Detection)](), [Loftr (Feature Matching)](), and [SAM (Segmentation)](). Now, we're looking for contributors to help us:

- Expand the Model Selection: Import decent models into our library. If you are a researcher, Kornia is an excellent place for you to promote your model!
- Model Optimization: Work on optimizing models to reduce their computational footprint while maintaining accuracy and performance. You may start from offering ONNX support!
- Model Documentation: Create detailed guides and examples to help users get the most out of these models in their projects.


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

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us. Please, consider reading the [CONTRIBUTING](./CONTRIBUTING.md) notes. The participation in this open source project is subject to [Code of Conduct](./CODE_OF_CONDUCT.md).

## Community

- **Forums:** discuss implementations, research, etc. [GitHub Forums](https://github.com/kornia/kornia/discussions)
- **GitHub Issues:** bug reports, feature requests, install issues, RFCs, thoughts, etc. [OPEN](https://github.com/kornia/kornia/issues/new/choose)
- **Slack:** Join our workspace to keep in touch with our core contributors and be part of our community. [JOIN HERE](https://join.slack.com/t/kornia/shared_invite/zt-csobk21g-2AQRi~X9Uu6PLMuUZdvfjA)

<a href="https://github.com/Kornia/kornia/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Kornia/kornia" width="60%" />
</a>

Made with [contrib.rocks](https://contrib.rocks).

## License

Kornia is released under the Apache 2.0 license. See the [LICENSE](./LICENSE) file for more information.
