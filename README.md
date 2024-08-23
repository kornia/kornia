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

**Kornia** is a differentiable computer vision library that provides a rich set of differentiable image processing and vision algorithms. Built on [PyTorch](https://pytorch.org),Kornia integrates seamlessly into existing AI workflows, allowing you to leverage powerful [batch transformations](), [auto-differentiation]() and [GPU acceleration](). Whether you’re working on image transformations, augmentations, or AI-driven image processing, Kornia equips you with the tools you need to bring your ideas to life.

## Key Components
1. **Differentiable Image Processing**<br>
  Kornia provides a comprehensive suite of image processing operators, all differentiable and ready to integrate into deep learning pipelines.
    - **Filters**: Gaussian, Sobel, Median, Box Blur, etc.
    - **Transformations**: Affine, Homography, Perspective, etc.
    - **Enhancements**: Histogram Equalization, CLAHE, Gamma Correction, etc.
    - **Edge Detection**: Canny, Laplacian, Sobel, etc.
    - ... check our [docs](https://kornia.readthedocs.io) for more.
2. **Advanced Augmentations**<br>
Perform powerful data augmentation with Kornia’s built-in functions, ideal for training robust AI models.
    - **Augmentation Pipeline**: AugmentationSequential, PatchSequential, VideoSequential, etc.
    - **Automatic Augmentation**: AutoAugment, RandAugment, TrivialAugment.
3. **AI Models**<br>
Leverage pre-trained AI models optimized for a variety of vision tasks, all within the Kornia ecosystem.
    - **Face Detection**: YuNet
    - **Feature Matching**: LoFTR, etc.
    - **Segmentation**: SAM, DeepLabV3
    - **Classification**: MobileViT, VisionTransformer.

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

Kornia is not just another computer vision library — it's your gateway to effortless image transformation and augmentation.

```python
import numpy as np
from PIL import Image
from kornia.augmentation import AugmentationSequential, RandomAffine, RandomBrightness
from kornia.filters import StableDiffusionDissolving

# Load and prepare your image
img = Image.open("img.jpeg").resize((256, 256))
img = np.stack([np.array(img)] * 2)  # Example for numpy input

# Define an augmentation pipeline
augmentation_pipeline = AugmentationSequential(
    RandomAffine((-45., 45.), p=1.),
    RandomBrightness((0.,1.), p=1.)
)

img = augmentation_pipeline(img)

# Leveraging StableDiffusion models
output = StableDiffusionDissolving()(img, step_number=500)
output.save("Kornia-enhanced.jpg")
```

In addition, Kornia offers lots of

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

Kornia's foundation lies in its extensive collection of classic computer vision operators, providing robust tools for image processing, feature extraction, and geometric transformations. We continuously seek for contributors to help us improve our documentation and present nice tutorials for our users.


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
