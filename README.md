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

**Kornia** is a differentiable computer vision library for [PyTorch](https://pytorch.org).

It consists of a set of routines and differentiable modules to solve generic computer vision problems. At its core, the package uses *PyTorch* as its main backend both for efficiency and to take advantage of the reverse-mode auto-differentiation to define and compute the gradient of complex functions.

Inspired by existing packages, this library is composed by a subset of packages containing operators that can be inserted within neural networks to train models to perform image transformations, epipolar geometry, depth estimation, and low-level image processing such as filtering and edge detection that operate directly on tensors.

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
