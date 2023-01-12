<div align="center">
<p align="center">
  <img width="75%" src="https://github.com/kornia/data/raw/main/kornia_banner_pixie.png" />
</p>

---

English | [简体中文](README_zh-CN.md)

<!-- prettier-ignore -->
<a href="https://kornia.org">Website</a> •
<a href="https://kornia.readthedocs.io">Docs</a> •
<a href="https://colab.research.google.com/github/kornia/tutorials/blob/master/source/hello_world_tutorial.ipynb">Try it Now</a> •
<a href="https://kornia-tutorials.readthedocs.io">Tutorials</a> •
<a href="https://github.com/kornia/kornia-examples">Examples</a> •
<a href="https://kornia.github.io//kornia-blog">Blog</a> •
<a href="https://join.slack.com/t/kornia/shared_invite/zt-csobk21g-CnydWe5fmvkcktIeRFGCEQ">Community</a>

[![PyPI python](https://img.shields.io/pypi/pyversions/kornia)](https://pypi.org/project/kornia)
[![PyPI version](https://badge.fury.io/py/kornia.svg)](https://pypi.org/project/kornia)
[![Downloads](https://pepy.tech/badge/kornia)](https://pepy.tech/project/kornia)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENCE)
[![Slack](https://img.shields.io/badge/Slack-4A154B?logo=slack&logoColor=white)](https://join.slack.com/t/kornia/shared_invite/zt-csobk21g-2AQRi~X9Uu6PLMuUZdvfjA)
[![Twitter](https://img.shields.io/twitter/follow/kornia_foss?style=social)](https://twitter.com/kornia_foss)

[![tests-cpu](https://github.com/kornia/kornia/actions/workflows/scheduled_test_cpu.yml/badge.svg?event=schedule&&branch=master)](https://github.com/kornia/kornia/actions/workflows/scheduled_test_cpu.yml)
[![tests-cpu-nightly](https://github.com/kornia/kornia/actions/workflows/scheduled_test_nightly.yml/badge.svg?event=schedule&&branch=master)](https://github.com/kornia/kornia/actions/workflows/scheduled_test_nightly.yml)
[![tests-cuda](https://github.com/kornia/kornia/actions/workflows/tests_cuda.yml/badge.svg)](https://github.com/kornia/kornia/actions/workflows/tests_cuda.yml)
[![codecov](https://codecov.io/gh/kornia/kornia/branch/master/graph/badge.svg?token=FzCb7e0Bso)](https://codecov.io/gh/kornia/kornia)
[![Documentation Status](https://readthedocs.org/projects/kornia/badge/?version=latest)](https://kornia.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/kornia/kornia/master.svg)](https://results.pre-commit.ci/latest/github/kornia/kornia/master)

<a href="https://www.producthunt.com/posts/kornia?utm_source=badge-featured&utm_medium=badge&utm_souce=badge-kornia" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=306439&theme=light" alt="Kornia - Computer vision library for deep learning | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>

</p>
</div>

*Kornia* is a differentiable computer vision library for [PyTorch](https://pytorch.org).

It consists of a set of routines and differentiable modules to solve generic computer vision problems. At its core, the package uses *PyTorch* as its main backend both for efficiency and to take advantage of the reverse-mode auto-differentiation to define and compute the gradient of complex functions.

<div align="center">
  <img src="https://github.com/kornia/kornia/raw/master/docs/source/_static/img/hakuna_matata.gif" width="75%" height="75%">
</div>

<!--<div align="center">
  <img src="http://drive.google.com/uc?export=view&id=1KNwaanUdY1MynF0EYfyXjDM3ti09tzaq">
</div>-->

## Overview

Inspired by existing packages, this library is composed by a subset of packages containing operators that can be inserted within neural networks to train models to perform image transformations, epipolar geometry, depth estimation, and low-level image processing such as filtering and edge detection that operate directly on tensors.

At a granular level, Kornia is a library that consists of the following components:

| **Component**                                                                    | **Description**                                                                                                                       |
|----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| [kornia](https://kornia.readthedocs.io/en/latest/index.html)                     | a Differentiable Computer Vision library, with strong GPU support                                                                     |
| [kornia.augmentation](https://kornia.readthedocs.io/en/latest/augmentation.html) | a module to perform data augmentation in the GPU                                                                                      |
| [kornia.color](https://kornia.readthedocs.io/en/latest/color.html)               | a set of routines to perform color space conversions                                                                                  |
| [kornia.contrib](https://kornia.readthedocs.io/en/latest/contrib.html)           | a compilation of user contrib and experimental operators                                                                              |
| [kornia.enhance](https://kornia.readthedocs.io/en/latest/enhance.html)           | a module to perform normalization and intensity transformation                                                                        |
| [kornia.feature](https://kornia.readthedocs.io/en/latest/feature.html)           | a module to perform feature detection                                                                                                 |
| [kornia.filters](https://kornia.readthedocs.io/en/latest/filters.html)           | a module to perform image filtering and edge detection                                                                                |
| [kornia.geometry](https://kornia.readthedocs.io/en/latest/geometry.html)         | a geometric computer vision library to perform image transformations, 3D linear algebra and conversions using different camera models |
| [kornia.losses](https://kornia.readthedocs.io/en/latest/losses.html)             | a stack of loss functions to solve different vision tasks                                                                             |
| [kornia.morphology](https://kornia.readthedocs.io/en/latest/morphology.html)     | a module to perform morphological operations                                                                                          |
| [kornia.utils](https://kornia.readthedocs.io/en/latest/utils.html)               | image to tensor utilities and metrics for vision problems                                                                             |

## Installation

### From pip:

  ```bash
  pip install kornia
  pip install kornia[x]  # to get the training API !
  ```

<details>
  <summary>Other installation options</summary>

  #### From source:

  ```bash
  python setup.py install
  ```

  #### From source with symbolic links:

  ```bash
  pip install -e .
  ```

  #### From source using pip:

  ```bash
  pip install git+https://github.com/kornia/kornia
  ```

</details>


## Examples

Run our Jupyter notebooks [tutorials](https://kornia-tutorials.readthedocs.io/en/latest/) to learn to use the library.

<div align="center">
  <a href="https://colab.research.google.com/github/kornia/tutorials/blob/master/source/hello_world_tutorial.ipynb" target="_blank">
    <img src="https://raw.githubusercontent.com/kornia/data/main/hello_world_arturito.png" width="75%" height="75%">
  </a>
</div>

:triangular_flag_on_post: **Updates**
- :white_check_mark: [Image Matching](https://kornia.readthedocs.io/en/latest/applications/image_matching.html) Integrated to [Huggingface Spaces](https://huggingface.co/spaces). See [Gradio Web Demo](https://huggingface.co/spaces/akhaliq/Kornia-LoFTR).
- :white_check_mark: [Face Detection](https://kornia.readthedocs.io/en/latest/applications/face_detection.html) Integrated to [Huggingface Spaces](https://huggingface.co/spaces). See [Gradio Web Demo](https://huggingface.co/spaces/frapochetti/blurry-faces).

## Cite

If you are using kornia in your research-related documents, it is recommended that you cite the paper. See more in [CITATION](https://github.com/kornia/kornia/blob/master/CITATION.md).

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
We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us. Please, consider reading the [CONTRIBUTING](https://github.com/kornia/kornia/blob/master/CONTRIBUTING.rst) notes. The participation in this open source project is subject to [Code of Conduct](https://github.com/kornia/kornia/blob/master/CODE_OF_CONDUCT.md).


## Community
- **Forums:** discuss implementations, research, etc. [GitHub Forums](https://github.com/kornia/kornia/discussions)
- **GitHub Issues:** bug reports, feature requests, install issues, RFCs, thoughts, etc. [OPEN](https://github.com/kornia/kornia/issues/new/choose)
- **Slack:** Join our workspace to keep in touch with our core contributors and be part of our community. [JOIN HERE](https://join.slack.com/t/kornia/shared_invite/zt-csobk21g-2AQRi~X9Uu6PLMuUZdvfjA)
- For general information, please visit our website at www.kornia.org

<a href="https://github.com/Kornia/kornia/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Kornia/kornia" width="75%" height="75%" />
</a>

Made with [contrib.rocks](https://contrib.rocks).
