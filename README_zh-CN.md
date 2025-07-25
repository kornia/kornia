<div align="center">
<p align="center">
  <img width="75%" src="https://github.com/kornia/data/raw/main/kornia_banner_pixie.png" />
</p>

**The open-source and Computer Vision 2.0 library**

---

[English](README.md) | 简体中文

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
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENCE)

</p>
</div>

*Kornia* 是一款基于 [PyTorch](https://pytorch.org) 的可微分的计算机视觉库。

它由一组用于解决通用计算机视觉问题的操作模块和可微分模块组成。其核心使用 *PyTorch* 作为主要后端，以提高效率并利用反向模式自动微分来定义和计算复杂函数的梯度。

<div align="center">
  <img src="https://github.com/kornia/kornia/raw/main/docs/source/_static/img/hakuna_matata.gif" width="75%" height="75%">
</div>

<!--<div align="center">
  <img src="http://drive.google.com/uc?export=view&id=1KNwaanUdY1MynF0EYfyXjDM3ti09tzaq">
</div>-->

## 概览

受现有开源库的启发，Kornia可以由包含各种可以嵌入神经网络的操作符组成，并可以训练模型来执行图像变换、对极几何、深度估计和低级图像处理，例如过滤和边缘检测。此外，整个库都可以直接对张量进行操作。

详细来说，Kornia 是一个包含以下组件的库：

| **Component**                                                                    | **Description**                                                                                                                       |
|----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| [kornia](https://kornia.readthedocs.io/en/latest/index.html)                     | 具有强大 GPU 支持的可微计算机视觉库                                                                   |
| [kornia.augmentation](https://kornia.readthedocs.io/en/latest/augmentation.html) | 在 GPU 中执行数据增强的模块                                                                                     |
| [kornia.color](https://kornia.readthedocs.io/en/latest/color.html)               | 执行色彩空间转换的模块                                                                                  |
| [kornia.contrib](https://kornia.readthedocs.io/en/latest/contrib.html)           | 未进入稳定版本的实验性模块                                                                              |
| [kornia.enhance](https://kornia.readthedocs.io/en/latest/enhance.html)           | 执行归一化和像素强度变换的模块                                                                        |
| [kornia.feature](https://kornia.readthedocs.io/en/latest/feature.html)           | 执行特征检测的模块                                                                                                 |
| [kornia.filters](https://kornia.readthedocs.io/en/latest/filters.html)           | 执行图像滤波和边缘检测的模块                                                                                |
| [kornia.geometry](https://kornia.readthedocs.io/en/latest/geometry.html)         | 执行几何计算的模块，用于使用不同的相机模型执行图像变换、3D线性代数和转换 |
| [kornia.losses](https://kornia.readthedocs.io/en/latest/losses.html)             | 损失函数模块                                                                             |
| [kornia.morphology](https://kornia.readthedocs.io/en/latest/morphology.html)     | 执行形态学操作的模块                                                                                          |
| [kornia.utils](https://kornia.readthedocs.io/en/latest/utils.html)               | 图像/张量常用工具以及metrics                                                                             |

## 支持我们

<a href="https://opencollective.com/kornia/donate" target="_blank">
  <img src="https://opencollective.com/webpack/donate/button@2x.png?color=blue" width=300 />
</a>

## 安装说明

### 通过 pip 安装:

  ```bash
  pip install kornia
  ```

<details>
  <summary>其他安装方法</summary>

  #### 通过源码安装（软链接至当前路径）:

  ```bash
  pip install -e .
  ```

  #### 通过源码安装（从GIT自动下载最新代码）:

  ```bash
  pip install git+https://github.com/kornia/kornia
  ```
</details>


## 例子

可以尝试通过这些 [教程](https://kornia.github.io/tutorials/) 来学习和使用这个库。

<div align="center">
  <a href="https://colab.sandbox.google.com/github/kornia/tutorials/blob/master/nbs/hello_world_tutorial.ipynb" target="_blank">
    <img src="https://raw.githubusercontent.com/kornia/data/main/hello_world_arturito.png" width="75%" height="75%">
  </a>
</div>

:triangular_flag_on_post: **Updates**
- :white_check_mark: 现已通过 [Gradio](https://github.com/gradio-app/gradio) 将Kornia集成进 [Huggingface Spaces](https://huggingface.co/spaces). 可以尝试 [Gradio 在线Demo](https://huggingface.co/spaces/akhaliq/Kornia-LoFTR).

## 引用

如果您在与研究相关的文档中使用 Kornia，您可以引用我们的论文。更多信息可以在 [CITATION](https://github.com/kornia/kornia/blob/main/CITATION.md) 看到。

  ```bibtex
  @inproceedings{eriba2019kornia,
    author    = {E. Riba, D. Mishkin, D. Ponsa, E. Rublee and G. Bradski},
    title     = {Kornia: an Open Source Differentiable Computer Vision Library for PyTorch},
    booktitle = {Winter Conference on Applications of Computer Vision},
    year      = {2020},
    url       = {https://arxiv.org/pdf/1910.02190.pdf}
  }
  ```

## 贡献
我们感谢所有的贡献者为改进和提升 Kornia 所作出的努力。您可以直接修复一个已知的BUG而无需进一步讨论；如果您想要添加一个任何新的或者扩展功能，请务必先通过提交一个Issue来与我们讨论。详情请阅读 [贡献指南](https://github.com/kornia/kornia/blob/main/CONTRIBUTING.md)。开源项目的参与者请务必了解如下 [规范](https://github.com/kornia/kornia/blob/main/CODE_OF_CONDUCT.md)。

## 社区
- **论坛:** 讨论代码实现，学术研究等。[GitHub Forums](https://github.com/kornia/kornia/discussions)
- **GitHub Issues:** bug reports, feature requests, install issues, RFCs, thoughts, etc. [OPEN](https://github.com/kornia/kornia/issues/new/choose)
- **Slack:** 加入我们的Slack社区，与我们的核心贡献者保持联系。 [JOIN HERE](https://join.slack.com/t/kornia/shared_invite/zt-csobk21g-2AQRi~X9Uu6PLMuUZdvfjA)
- 常见信息请访问我们的网站 www.kornia.org

## 中文社区
扫描下方的二维码可关注 Kornia 的官方交流QQ群（679683070）以及Kornia知乎账号。

<div align="center">
  <img src="https://github.com/kornia/kornia/raw/main/docs/source/_static/img/cn_community_qq.jpg" height="700px">
  <img src="https://github.com/kornia/kornia/raw/main/docs/source/_static/img/cn_community_zhihu.jpg" height="700px">
</div>

我们会在 Kornia 交流社区为大家

- 📢 更新 Kornia 的最新动态
- 📘 进行更高效的答疑解惑以及意见反馈
- 💻 提供与行业大牛的充分交流的平台
