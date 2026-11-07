<div align="center">
<p align="center">
  <img width="55%" src="https://github.com/kornia/data/raw/main/kornia_banner_pixie.png" />
</p>

---

[English](README.md) | 简体中文

<p align="center">
  <a href="https://www.kornia.org/playground/"><img alt="在浏览器中试用" src="https://img.shields.io/badge/%E2%96%B6%20Try%20in%20your%20browser-F5B301?style=for-the-badge&labelColor=F5B301"></a>
  <a href="https://www.kornia.org"><img alt="网站" src="https://img.shields.io/badge/Website-0B1A2B?style=for-the-badge"></a>
  <a href="https://kornia.readthedocs.io"><img alt="文档" src="https://img.shields.io/badge/Docs-0B1A2B?style=for-the-badge"></a>
</p>

<!-- prettier-ignore -->
<a href="https://kornia.readthedocs.io">文档</a> •
<a href="https://colab.sandbox.google.com/github/kornia/tutorials/blob/master/nbs/hello_world_tutorial.ipynb">立即试用</a> •
<a href="https://www.kornia.org/tutorials/">教程</a> •
<a href="https://github.com/kornia/kornia-examples">示例</a> •
<a href="https://www.kornia.org/kornia-blog/">博客</a> •
<a href="https://discord.gg/HfnywwpBnD">社区</a>

[![PyPI version](https://badge.fury.io/py/kornia.svg)](https://pypi.org/project/kornia)
[![Downloads](https://static.pepy.tech/badge/kornia)](https://pepy.tech/project/kornia)
[![star](https://gitcode.com/kornia/kornia/star/badge.svg)](https://gitcode.com/kornia/kornia)
[![Discord](https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/HfnywwpBnD)
[![Twitter](https://img.shields.io/twitter/follow/kornia_foss?style=social)](https://twitter.com/kornia_foss)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</p>
</div>

**Kornia** 是一个可微分计算机视觉库，提供丰富的可微分图像处理和几何视觉算法。Kornia 基于 [PyTorch](https://pytorch.org) 构建，可无缝融入现有的 AI 工作流程，支持[批量变换](https://kornia.readthedocs.io/en/latest/get-started/gpu-acceleration.html)、[自动微分](https://kornia.readthedocs.io/en/latest/get-started/differentiability.html)和 [GPU 加速](https://kornia.readthedocs.io/en/latest/get-started/gpu-acceleration.html)。无论您需要图像变换、数据增强还是 AI 驱动的图像处理，Kornia 都能提供实现这些想法所需的工具。

> **📢 发展方向**：Kornia 正在成为 PyTorch 生态中可微分计算机视觉与几何算法的**参考实现和可执行规范**——相比增加 API 数量，我们更重视明确的约定、一致性测试以及如实报告的性能基准。详情请阅读[路线图](ROADMAP.md)。

## 何时使用 Kornia

当手写实现变得冗长或容易出错，或者替代方案需要将数据移出 GPU 时，Kornia 尤其适用：

- **覆盖面广。** 提供 1,200 多个公开算子，大多数在 torchvision 中没有对应实现：相机模型与对极几何、最小求解器与 RANSAC、特征检测、描述与匹配、边缘检测、形态学、Lab 与 Bayer 颜色处理，以及图像损失函数和评估指标。与 torchvision 重叠的功能主要是缩放、裁剪、翻转、颜色扰动和模糊。
- **一次批量调用即可为每个样本使用不同参数。** `gaussian_blur2d(x, (9, 9), sigma)` 接受形状为 `(B, 2)` 的 `sigma`；`rotate` 接受 `(B,)` 的角度，`warp_affine` / `warp_perspective` 接受 `(B, 2, 3)` / `(B, 3, 3)` 的矩阵，`adjust_brightness` 接受 `(B,)` 的系数。torchvision 的函数式算子在每次调用中使用同一组参数。从 Kornia 0.9.1 起，`sigma` 张量以及图像增强函数的取值检查（gamma、contrast、solarize）均在设备上完成，因此 CUDA 调用以及 PyTorch ≥ 2.13 上的 MPS 调用无需为这些检查与主机同步。
- **批量最小求解器与鲁棒估计。** `find_fundamental(x1, x2, method="7POINT")` 求解形状为 `(B, 7, 2)` 的样本，将全部实数解返回为 `(B, 3, 3, 3)`，未使用的位置填充零矩阵；还提供 8 点法、5 点本质矩阵求解器（`find_essential`）、DLT 单应性求解器和 PnP 求解器（`solve_pnp_dlt`）。`RANSAC`（支持单应矩阵、基础矩阵和本质矩阵模型）在设备上采样和评分候选模型；默认情况下，细化与最终选择在主机上以 float64 进行，而 `compile=True`（PyTorch ≥ 2.14，CPU 和 CUDA）将整个估计过程作为一个编译图在设备上执行。
- **可微分几何。** 深度与三维坐标之间的转换（`depth_to_3d_v2`、`unproject_points`、`project_points`）、`warp_frame_depth`、`remap`、具有 Brown–Conrady 和 Kannala–Brandt 畸变的相机模型，以及李群（`So3`、`Se3`）均保留相对于深度、位姿和相机参数的梯度。
- **带标注的数据增强。** 在二维场景中，`AugmentationSequential` 将同一次采样得到的变换同步应用于图像、掩码、边界框和关键点，并可执行逆变换；三维增强用于变换体数据，不支持标注或逆变换。在 CUDA 上，增强操作会将采样参数传到 GPU，而不阻塞 GPU；同步检查测试确保每个公开增强操作都不会产生可避免的主机同步。
- **GPU 上的局部特征。** 支持 SIFT 和 HardNet、DISK、ALIKED、XFeat、DeDoDe、LoFTR 与 LightGlue，以及互为最近邻和比率测试匹配器。
- **形态学。** 支持在批量张量上使用任意结构元素。
- **编译与导出。** 几乎所有已探测的算子配置都可在 `torch.compile` 下运行，并与 eager 执行结果一致；超过 1,000 个配置可导出为 ONNX。每个算子的结果都列在[导出支持页面](https://kornia.readthedocs.io/en/latest/get-started/export-support.html)。
- **在设计目标场景中速度快：GPU 上的批量张量。** 在已发布的 0.9.0 旗舰基准测试中（RTX 4090，批量大小 32），许多滤波器、图像变换、特征匹配步骤和图像指标在 eager 模式下，比 CPU 上逐张执行的 OpenCV 或 scikit-image 快约 10× 至 70×；使用 `torch.compile` 后，其中一些超过 100×。在同一 GPU 上与 torchvision 重叠的少量操作中，eager Kornia 较慢；编译后的 Kornia 在色调调整、颜色扰动和高斯模糊上较快，在翻转、裁剪和亮度调整上仍较慢。对于 CPU 上的单张 `uint8` 图像处理，OpenCV 更快。结果会随硬件和工作负载而变化；所有数值、优势和劣势都公开在[性能页面](https://kornia.readthedocs.io/en/latest/get-started/performance.html)。

这些算子在输入所在的设备上运行，无需经由 NumPy 往返转换，且大多数可微分；鲁棒估计（`RANSAC`）以及 DeDoDe、XFeat 等学习型特征的提取方法不计算梯度。对于参数固定的单个操作，例如一次模糊或缩放，几行 PyTorch 或 torchvision 代码也同样合适。

## 主要组件

1. **可微分图像处理**<br>
   Kornia 提供全面的图像处理算子，支持微分，可集成到深度学习流水线中。
    - **滤波器**：高斯、Sobel、中值、方框模糊等。
    - **变换**：仿射、单应性、透视等。
    - **图像增强**：直方图均衡化、CLAHE、伽马校正等。
    - **边缘检测**：Canny、拉普拉斯、Sobel 等。
    - 更多内容请参阅[文档](https://kornia.readthedocs.io)。
2. **高级数据增强**<br>
   使用 Kornia 内置函数执行数据增强，适用于包含复杂增强流水线的 AI 模型训练。
    - **增强流水线**：AugmentationSequential、PatchSequential、VideoSequential 等。
    - **自动增强**：AutoAugment、RandAugment、TrivialAugment。
3. **AI 模型**<br>
   在 Kornia 生态中使用针对不同视觉任务优化的预训练 AI 模型。
    - **人脸检测**：YuNet
    - **特征匹配**：LoFTR、LightGlue
    - **特征描述子**：DISK、DeDoDe、SOLD2
    - **分割**：SAM
    - **分类**：MobileViT、VisionTransformer。

<details>
<summary>查看部分支持的方法（总计超过 500 个算子！）</summary>

| **类别** | **方法／模型** |
|----------|---------------|
| **图像处理** | - 颜色转换（RGB、灰度、HSV 等）<br>- 几何变换（仿射、单应性、缩放等）<br>- 滤波（高斯模糊、中值模糊等）<br>- 边缘检测（Sobel、Canny 等）<br>- 形态学操作（腐蚀、膨胀等） |
| **数据增强** | - 随机裁剪、擦除<br>- 随机几何变换（仿射、翻转、鱼眼、透视、薄板样条、弹性变换）<br>- 随机噪声（高斯、中值、运动、方框、雨、雪、椒盐）<br>- 随机颜色扰动（对比度、亮度、CLAHE、均衡化、Gamma、色调、反色、JPEG、Plasma、色调分离、饱和度、锐度、Solarize）<br>- 随机 MixUp、CutMix、Mosaic、Transplantation 等 |
| **特征检测** | - 检测器（Harris、GFTT、Hessian、DoG、KeyNet、DISK 和 DeDoDe）<br>- 描述子（SIFT、HardNet、TFeat、HyNet、SOSNet 和 LAFDescriptor）<br>- 匹配（最近邻、互为最近邻、几何感知匹配、AdaLAM、LightGlue 和 LoFTR） |
| **几何** | - 相机模型与标定<br>- 双目视觉（对极几何、视差等）<br>- 单应性估计<br>- 从视差估计深度<br>- 三维变换 |
| **深度学习层** | - 自定义卷积层<br>- 面向视觉任务的循环层<br>- 损失函数（如 SSIM、PSNR 等）<br>- 视觉专用优化器 |
| **光度函数** | - 光度损失函数<br>- 光度增强 |
| **滤波** | - 双边滤波<br>- DexiNed<br>- Dissolving<br>- 引导模糊<br>- 拉普拉斯<br>- 高斯<br>- 非局部均值<br>- Sobel<br>- 非锐化掩蔽 |
| **颜色** | - 色彩空间转换<br>- 亮度／对比度调整<br>- 伽马校正 |
| **双目视觉** | - 视差估计<br>- 深度估计<br>- 校正 |
| **图像配准** | - 基于仿射和单应性的配准<br>- 基于特征的图像对齐 |
| **位姿估计** | - 本质矩阵与基础矩阵估计<br>- PnP 问题求解器<br>- 位姿细化 |
| **光流** | - Farneback 光流<br>- 稠密光流<br>- 稀疏光流 |
| **三维视觉** | - 深度估计<br>- 点云操作 |
| **图像去噪** | - 去除高斯噪声<br>- 去除泊松噪声 |
| **边缘检测** | - Sobel 算子<br>- Canny 边缘检测 |
| **变换** | - 旋转<br>- 平移<br>- 缩放<br>- 剪切 |
| **损失函数** | - SSIM（结构相似性指标）<br>- PSNR（峰值信噪比）<br>- Cauchy<br>- Charbonnier<br>- 深度平滑<br>- Dice<br>- Hausdorff<br>- Tversky<br>- Welsch |
| **形态学操作** | - 膨胀<br>- 腐蚀<br>- 开运算<br>- 闭运算 |

</details>

## CUDA 与半精度支持

下表按模块统计 Linux CPU 已知失败清单中的条目（[`cpu_float16.txt`](testing/half_precision_xfails/cpu_float16.txt)、[`cpu_bfloat16.txt`](testing/half_precision_xfails/cpu_bfloat16.txt)），对应固定的 Python 3.11 和 PyTorch 2.15.0 环境。
清单记录了 **239 个 float16 失败和 251 个 bfloat16 失败**。这些数量指参数化测试用例，并不代表不受支持的函数数量。[#4153](https://github.com/kornia/kornia/issues/4153) 曾跟踪这些失败，直到该议题于 2026-10-18 关闭；目前没有单个议题跟踪所有剩余条目。
没有 CI 任务覆盖 CUDA 或 MPS 半精度；CPU 结果不能证明这些后端的支持情况。完整测试套件的一次本地 CUDA 测量见[下文](#cuda-测试套件测量)。

| 模块 | float16 | bfloat16 | 已知失败（fp16 / bf16） | 说明 |
|------|:-------:|:--------:|:----------------------:|------|
| `kornia.color` | ✅ | ✅ | 0 / 0 | Lab/Luv 转换和 `rgb255_to_normals` 对半精度输入使用 float32 计算，并返回输入 dtype |
| `kornia.filters` | ✅ | ✅ | 0 / 0 | Canny、高斯核／模糊和 FFT 卷积对半精度输入使用 float32 计算，并返回输入 dtype |
| `kornia.enhance` | ✅ | ✅ | 0 / 0 | ZCA 对半精度输入保留 float32 拟合状态；白化和 JPEG 以 float32 计算后返回输入 dtype |
| `kornia.morphology` | ✅ | ✅ | 0 / 0 |  |
| `kornia.augmentation` | ⚠️ | ⚠️ | 59 / 47 | float16：容器 autocast／运行时错误和精度问题；两种 dtype：三维反向传播和随机生成器精度问题 |
| `kornia.geometry.transform` | ⚠️ | ⚠️ | 31 / 30 | 旋转矩阵、仿射／透视变换和单应性变换器的精度问题 |
| `kornia.geometry.camera` | ⚠️ | ⚠️ | 13 / 23 | 投影、畸变往返变换和双目重投影；包括测试中的 dtype 断言 |
| `kornia.geometry.calibration` | ⚠️ | ⚠️ | 12 / 12 | solve_pnp_dlt 拒绝半精度输入；undistort_points 精度问题 |
| `kornia.geometry.epipolar` | ⚠️ | ⚠️ | 36 / 36 | 本质矩阵分解／运动恢复和 KRt_from_projection 缺少 CPU 半精度内核；基础矩阵精度问题 |
| `kornia.geometry.homography` | ⚠️ | ⚠️ | 10 / 16 | DLT 求解器精度问题，包括迭代版本和基于直线的版本 |
| `kornia.geometry.liegroup` | ✅ | ✅ | 0 / 0 | So2/Se2 拒绝用 bfloat16 构造旋转：PyTorch 没有复数 bfloat16 dtype；这些预期错误／跳过项不计入清单 |
| `kornia.geometry.solvers` | ✅ | ✅ | 0 / 0 |  |
| `kornia.geometry.subpix` | ⚠️ | ⚠️ | 6 / 3 | SpatialSoftArgmax2d 和 ConvQuadInterp3d 精度问题；bfloat16：render_gaussian2d 精度问题 |
| `kornia.geometry.conversions` | ⚠️ | ⚠️ | 65 / 65 | 轴角、四元数和旋转矩阵的精度问题；包括 tests/integration/test_conversions.py |
| `kornia.geometry.ransac` | ⚠️ | ⚠️ | 2 / 2 | 本质矩阵模型拟合在对极几何求解器中抛出异常 |
| `kornia.geometry (other)` | ⚠️ | ⚠️ | 3 / 10 | 边界框变换和深度的精度问题 |
| `kornia.image` | ⚠️ | ⚠️ | 2 / 2 | 圆／椭圆多边形填充的比较使用半精度计算的顶点和参考掩码 |
| `kornia.losses` | ✅ | ⚠️ | 0 / 2 | bfloat16：总变分精度问题 |
| `kornia.feature` | ✅ | ✅ | 0 / 0 | 特征匹配在半精度 dtype 下使用手动实现的 cdist 回退路径；LightGlue 的 float16 测试被跳过 |
| `kornia.metrics` | ✅ | ✅ | 0 / 0 |  |
| `kornia.models` | ✅ | ⚠️ | 0 / 1 | bfloat16：RT-DETR RepVGG 融合精度问题 |
| `contrib, core, io, onnx, sensors, tracking, utils` | ✅ | ⚠️ | 0 / 2 | bfloat16：直方图匹配和 _torch_svd_cast 精度问题 |

✅ CPU 已知失败清单中没有条目 &nbsp; ⚠️ 存在已记录的 CPU 失败。
这两个符号都不承诺完整的 dtype 支持：跳过的测试、预期异常以及源码中的 xfail 不在这些计数内。一些算子内部使用 float32 或 float64 计算，再返回输入 dtype。

**观测到的 CPU CI 结果：**

[2026-10-21 在 `93f1376e` 上的定时运行](https://github.com/kornia/kornia/actions/runs/39524889979)使用 Linux x86_64、Python 3.11 和 PyTorch 2.14.0，不包含慢速测试或编译测试：

| 运行 | 通过 | 失败 | 跳过 | XFAIL（全部） | XPASS | 严格清单条目 |
|------|-----:|-----:|-----:|-------------:|------:|-------------:|
| [CPU float16](https://github.com/kornia/kornia/actions/runs/39524889979/job/118977300500) | 23905 | 0 | 5132 | 284 | 4 | 250 |
| [CPU bfloat16](https://github.com/kornia/kornia/actions/runs/39524889979/job/118977300410) | 23726 | 0 | 5286 | 309 | 4 | 274 |

每个任务还取消选择了 3 个 ONNX 一致性测试。XFAIL 总数包括严格清单条目和源码中的预期失败。每个成功任务中的 4 个 XPASS，是原本非严格预期失败但实际通过的测试；**清单**中的条目一旦通过，则会使验证失败，异常类型改变、条目被跳过或节点 ID 过期也同样如此。因此，CI 任务变绿意味着记录的基线得到复现，即便仍有已知失败和跳过项。这些总数描述的是所链接的版本；模块表描述的是仓库中提交的清单。

在固定环境中验证完整 CPU 配置：

```bash
KORNIA_TEST_OPTIMIZER= pixi run test-module tests/ --verify-known-failures --known-failure-profile=cpu-float16
KORNIA_TEST_OPTIMIZER= pixi run test-module tests/ --verify-known-failures --known-failure-profile=cpu-bfloat16
```

`pixi run test-half` 是一次未固定随机种子的 CPU 测试运行，不使用严格配置基线。定向检查和基线维护方法请参阅 [TESTING.md](TESTING.md#7-half-precision-dtypes-float16--bfloat16)。

### CUDA 测试套件测量

Kornia 接受 CUDA 张量以使用 GPU 加速，但受各算子的 dtype 和后端限制约束。目前没有 CUDA CI 任务，也没有提交到仓库的 CUDA 已知失败清单，因此下列总数只是某一版本的本地测量，不保证后续版本具有相同结果：版本 `e4a444e9b`（2026-11-04），RTX 4090，Python 3.11，PyTorch 2.15.0+cu132，关闭 TF32，对 `tests/` 使用 `--device=cuda`：

| CUDA 运行 | 通过 | 失败 | 跳过 | XFAIL | XPASS |
|-----------|-----:|-----:|-----:|------:|------:|
| float32 + float64 | 46079 | 0 | 9198 | 46 | 11 |
| float16 + bfloat16 | 45024 | 440 | 9813 | 50 | 7 |
| float32，`torch.compile`（inductor）测试 | 853 | 6 | 80 | 1 | 1 |

- **float32 和 float64：** 没有失败。
- **float16 和 bfloat16：** 440 个失败中，408 个是 CPU 清单条目在 CUDA 上的对应用例。其余 32 个中，21 个在独立进程中通过，2 个使用未固定随机种子的输入，在 CPU 上也会失败。有 9 个仅在 CUDA 上失败：其中 5 个是因为 CUDA 按批量大小选择卷积和矩阵乘法内核，使半精度 LoFTR、SmallSR 或 LightGlue 对同一图像的批量输出与单独输出略有不同（在 CPU 上逐位相同；使用预训练权重时，这一差异小于半精度误差本身）；另外 4 个是 `RandomRotation3D` 的 bfloat16 反向传播用例，原因在于 bfloat16 的限制，而非 CUDA：测试中的 SGD 步长小于每个角度的半个 bfloat16 ulp，因此角度没有变化。
- **`torch.compile`：** 6 个编译测试仅在 CUDA 上失败（缩放裁剪的重放、Otsu 分箱边界、`fit_line`、`Hyperplane` 和 `Keypoints` 的标量索引）。

[精度指南](https://kornia.readthedocs.io/en/latest/get-started/precision.html#cuda-test-suite-measurement)列出了每项失败，给出了批量与单张处理的测量结果，并列出测试套件未覆盖的、仍未解决的 CUDA 专属问题。

运行 CUDA 半精度测试时，请使用 `--isolate-half-precision`，使每个测试都在新进程中运行。缺少此选项时，这些测试默认被跳过；仅选择半精度 dtype 并不能测试 CUDA 半精度支持：

```bash
pixi run test-module tests/color/ --device=cuda --dtype=float16,bfloat16 --isolate-half-precision
```

详情请参阅[完整精度指南](https://kornia.readthedocs.io/en/latest/get-started/precision.html)。

## 赞助

Kornia 是一个由志愿者开发和维护的开源项目。无论您是将其用于研究还是商业用途，都欢迎考虑赞助我们或与我们合作。您的支持将有助于确保 Kornia 的成长与持续创新。立即联系我们，成为塑造这一令人振奋项目未来的一员！

<a href="https://opencollective.com/kornia/donate" target="_blank">
  <img src="https://opencollective.com/webpack/donate/button@2x.png?color=blue" width=300 />
</a>

## 安装

[![PyPI python](https://img.shields.io/pypi/pyversions/kornia)](https://pypi.org/project/kornia)
[![pytorch](https://img.shields.io/badge/PyTorch_2.5.1+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

### 使用 pip 安装

  ```bash
  pip install kornia
  ```

部分功能（ONNX、Stable Diffusion 溶解）需要额外的软件包；请参阅[可选扩展](https://kornia.readthedocs.io/en/latest/get-started/installation.html#optional-extras)。

<details>
  <summary>其他安装选项</summary>

#### 从源码以可编辑模式安装

  ```bash
  pip install -e .
  ```

#### 使用 Pixi 进行开发（推荐）

开发 Kornia 时，我们使用 [pixi](https://pixi.sh) 进行快速的 Python 包与环境管理。项目包含一个 `pixi.toml` 配置文件，以实现可复现的依赖管理。

  ```bash
  # 安装 pixi（如果尚未安装）
  curl -fsSL https://pixi.sh/install.sh | bash

  # 创建 Pixi 环境并安装开发依赖
  pixi install
  pixi run -e default install

  # 运行测试
  pixi run test

  # 用于 CUDA 开发
  pixi run -e cuda install
  pixi run -e cuda test-cuda
  ```

这些命令会设置包含所有依赖项的完整开发环境。有关依赖管理和可用任务的更多详情，请参阅 [TESTING.md](TESTING.md#environment-and-validation-tasks)。

#### 从 Github URL 安装（最新版本）

  ```bash
  pip install git+https://github.com/kornia/kornia
  ```

</details>

## 快速开始

Kornia 不只是另一个计算机视觉库——它是您轻松进入计算机视觉和 AI 世界的入口。

<details>
<summary>开始使用 Kornia 进行图像变换和增强！</summary>

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
<summary>使用 ONNXSequential 探索 Kornia ONNX 模型！</summary>

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

## 征集贡献者

如果 Kornia 对您有用，且您愿意提供帮助，我们欢迎各种形式的贡献：代码、错误报告、基准测试、文档、问题、解答和示例。维护者的时间有限，因此我们无法承诺审核或合并每一个提案或拉取请求。

### 强化核心（优先事项）

Kornia 的差异化价值在于其几何核心：图像变换与采样、单应性、相机、对极几何、旋转与李群，以及几何一致的增强。最具影响力的贡献能让这一核心更值得信赖——完整概览请参阅[路线图](ROADMAP.md)。可以从以下方面入手：

- **来自您硬件的基准测试结果**：使用 `--contribute` 运行基准测试套件并发送 JSON——来自不同 GPU 的 CUDA 数据尤其有价值。
- 为核心几何操作编写**锁定约定行为的测试和一致性测试向量**。
- **有助于纠正问题的错误信息**：将简单的形状断言升级为说明错误之处、预期内容及适用约定的错误信息。
- **已记录的 CUDA 和半精度失败**：根据[正确性策略](ROADMAP.md#correctness-and-backend-support)修复现有 CUDA 报告和未关闭的半精度议题中的问题，以及 [`float16`](testing/half_precision_xfails/cpu_float16.txt)/[`bfloat16`](testing/half_precision_xfails/cpu_bfloat16.txt) 已知失败清单中的条目。仅在有 CUDA 设备可用于复现和验证时开展 CUDA 修复；在验证修复后，移除相应的 xfail 或清单条目。

更多项目背景请参阅[路线图中面向贡献者的领域](ROADMAP.md#areas-seeking-contributors)。

### AI 模型

在维护者的精力集中于核心期间，模型库目前**暂停扩展**。已发布的模型（LoFTR、LightGlue、DISK、DeDoDe、SAM 等）将保持可用并继续维护，而在暂停前获批的模型工作（Efficient LoFTR、SANDesc）将在其既有范围内完成。新的集成——包括 VLM/VLA 模型——必须由明确指定的维护者支持，并由其承担长期维护责任；仅有贡献者的实现不足以重新开放模型扩展。有关原因和重新开放条件，请参阅[路线图](ROADMAP.md#guiding-themes)。

### 文档与教程优化

Kornia 的基础在于其丰富的经典计算机视觉算子集合，为图像处理、特征提取和几何变换提供了强大的工具。我们持续寻找贡献者，帮助我们改进文档，并为用户提供优秀的教程。

## 引用

如果您在研究相关文档中使用 Kornia，建议引用该论文。更多信息请参阅 [CITATION](./CITATION.md)。

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

有关我们的社会契约、开发环境设置和技术指南，请参阅 [CONTRIBUTING.md](./CONTRIBUTING.md)。参与即表示您同意遵守[行为准则](./CODE_OF_CONDUCT.md)。

## 社区

- **Discord**：与使用和开发 Kornia 的人交流。[加入服务器](https://discord.gg/HfnywwpBnD)
- **GitHub Issues**：报告错误并提出具体更改。[创建 Issue](https://github.com/kornia/kornia/issues/new/choose)
- **GitHub Discussions**：提出问题并讨论实现、研究和想法。[参与讨论](https://github.com/kornia/kornia/discussions)

<a href="https://github.com/Kornia/kornia/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Kornia/kornia" width="60%" />
</a>

由 [contrib.rocks](https://contrib.rocks) 制作。

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

## 许可证

Kornia 采用 Apache 2.0 许可证发布。更多信息请参阅 [LICENSE](./LICENSE) 文件。
