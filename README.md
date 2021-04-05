<p align="center">
  <img width="50%" src="https://github.com/kornia/kornia/blob/master/docs/source/_static/img/kornia_logo.svg" />
</p>

--------------------------------------------------------------------------------

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![tests-cpu-versions](https://github.com/kornia/kornia/actions/workflows/tests_cpu_versions.yml/badge.svg)](https://github.com/kornia/kornia/actions/workflows/tests_cpu_versions.yml)
[![tests-cuda-versions](https://github.com/kornia/kornia/actions/workflows/tests_cuda_versions.yml/badge.svg)](https://github.com/kornia/kornia/actions/workflows/tests_cuda_versions.yml)
[![codecov](https://codecov.io/gh/kornia/kornia/branch/master/graph/badge.svg?token=FzCb7e0Bso)](https://codecov.io/gh/kornia/kornia)
[![PyPI version](https://badge.fury.io/py/kornia.svg)](https://badge.fury.io/py/kornia)
[![Documentation Status](https://readthedocs.org/projects/kornia/badge/?version=latest)](https://kornia.readthedocs.io/en/latest/?badge=latest)

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
    <img src="https://kornia-tutorials.readthedocs.io/en/latest/_images/hello_world_tutorial_17_0.png" width="75%" height="75%">
  </a>
</div>

## Cite

If you are using kornia in your research-related documents, it is recommended that you cite the paper. Se more in [CITATION](https://github.com/kornia/kornia/blob/master/CITATION.md).

  ```bash
  @inproceedings{eriba2019kornia,
    author    = {E. Riba, D. Mishkin, D. Ponsa, E. Rublee and G. Bradski},
    title     = {Kornia: an Open Source Differentiable Computer Vision Library for PyTorch},
    booktitle = {Winter Conference on Applications of Computer Vision},
    year      = {2020},
    url       = {https://arxiv.org/pdf/1910.02190.pdf}
  }
  ```

## Contributing
We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us. Please, consider reading the [CONTRIBUTING](https://github.com/arraiyopensource/kornia/blob/master/CONTRIBUTING.rst) notes. The participation in this open source project is subject to [Code of Conduct](https://github.com/arraiyopensource/kornia/blob/master/CODE_OF_CONDUCT.md).


## Community
- **forums:** discuss implementations, research, etc. [GitHub Forums](https://github.com/kornia/kornia/discussions)
- **GitHub issues:** bug reports, feature requests, install issues, RFCs, thoughts, etc. [OPEN](https://github.com/kornia/kornia/issues/new/choose)
- **Slack:** Join our workspace to keep in touch with our core contributors and be part of our community. [JOIN HERE](https://join.slack.com/t/kornia/shared_invite/zt-csobk21g-CnydWe5fmvkcktIeRFGCEQ)
- for general information, please visit our website at www.kornia.org
