"""Upload kornia pretrained model weights as individual repos under the kornia HF org.

Creates one repo per model (e.g. kornia/xfeat, kornia/lightglue, …) and uploads
the weights plus a per-model README/model-card to each.

Usage:
    python scripts/upload_weights_to_hf.py [--dry-run] [--model MODEL [MODEL …]]

Requirements:
    pip install huggingface_hub requests tqdm

Environment:
    HF_TOKEN  — HuggingFace write token (must have write access to kornia org)
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import requests
from huggingface_hub import HfApi
from tqdm import tqdm

HF_ORG = "kornia"
REPO_TYPE = "model"

# ---------------------------------------------------------------------------
# Source URL constants
# ---------------------------------------------------------------------------
LG_VERSION = "v0.1_arxiv"
LG_BASE = f"https://github.com/cvg/LightGlue/releases/download/{LG_VERSION}"
MISHK = "http://cmp.felk.cvut.cz/~mishkdmy/models"


# ---------------------------------------------------------------------------
# Registry: repo_name -> list of (filename_in_repo, source_url)
# ---------------------------------------------------------------------------
WEIGHTS: dict[str, list[tuple[str, str]]] = {
    # ------------------------------------------------------------------
    "xfeat": [
        ("xfeat.pt", "https://github.com/verlab/accelerated_features/raw/main/weights/xfeat.pt"),
    ],
    # ------------------------------------------------------------------
    "lightglue": [
        ("superpoint_lightglue.pth", f"{LG_BASE}/superpoint_lightglue.pth"),
        ("disk_lightglue.pth", f"{LG_BASE}/disk_lightglue.pth"),
        ("aliked_lightglue.pth", f"{LG_BASE}/aliked_lightglue.pth"),
        ("raco_aliked_lightglue.pth", f"{LG_BASE}/raco_aliked_lightglue.pth"),
        ("sift_lightglue.pth", f"{LG_BASE}/sift_lightglue.pth"),
        ("doghardnet_lightglue.pth", f"{LG_BASE}/doghardnet_lightglue.pth"),
        ("keynet_affnet_hardnet_lightglue.pth", f"{MISHK}/keynet_affnet_hardnet_lightlue.pth"),
        ("dedodeb_lightglue.pth", f"{MISHK}/dedodeb_lightglue.pth"),
        ("dedodeg_lightglue.pth", f"{MISHK}/dedodeg_lightglue.pth"),
        ("xfeat-lighterglue.pt", "https://github.com/verlab/accelerated_features/raw/main/weights/xfeat-lighterglue.pt"),
    ],
    # ------------------------------------------------------------------
    "aliked": [
        ("aliked-t16.pth", "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-t16.pth"),
        ("aliked-n16.pth", "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth"),
        ("aliked-n16rot.pth", "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16rot.pth"),
        ("aliked-n32.pth", "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n32.pth"),
    ],
    # ------------------------------------------------------------------
    "disk": [
        ("depth-save.pth", "https://raw.githubusercontent.com/cvlab-epfl/disk/master/depth-save.pth"),
        ("epipolar-save.pth", "https://raw.githubusercontent.com/cvlab-epfl/disk/master/epipolar-save.pth"),
    ],
    # ------------------------------------------------------------------
    "sold2": [
        ("sold2_wireframe.pth", f"{MISHK}/sold2_wireframe.pth"),
    ],
    # ------------------------------------------------------------------
    "defmo": [
        ("encoder_best.pt", "http://ptak.felk.cvut.cz/personal/rozumden/defmo_saved_models/encoder_best.pt"),
        ("rendering_best.pt", "http://ptak.felk.cvut.cz/personal/rozumden/defmo_saved_models/rendering_best.pt"),
    ],
    # ------------------------------------------------------------------
    "dedode": [
        ("dedode_detector_L.pth", "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_detector_L.pth"),
        ("dedode_detector_C4.pth", "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/dedode_detector_C4.pth"),
        ("dedode_detector_SO2.pth", "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/dedode_detector_SO2.pth"),
        ("dedode_detector_L_v2.pth", "https://github.com/Parskatt/DeDoDe/releases/download/v2/dedode_detector_L_v2.pth"),
        ("dedode_descriptor_B.pth", "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_B.pth"),
        ("dedode_descriptor_G.pth", "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_G.pth"),
        ("B_C4_Perm_descriptor_setting_C.pth", "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/B_C4_Perm_descriptor_setting_C.pth"),
        ("B_SO2_Spread_descriptor_setting_C.pth", "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/B_SO2_Spread_descriptor_setting_C.pth"),
        ("G_C4_Perm_descriptor_setting_C.pth", "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/G_C4_Perm_descriptor_setting_C.pth"),
        ("G_SO2_Spread_descriptor_setting_C.pth", "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/G_SO2_Spread_descriptor_setting_C.pth"),
    ],
    # ------------------------------------------------------------------
    "loftr": [
        ("loftr_outdoor.ckpt", f"{MISHK}/loftr_outdoor.ckpt"),
        ("loftr_indoor_ds_new.ckpt", f"{MISHK}/loftr_indoor_ds_new.ckpt"),
        ("loftr_indoor.ckpt", f"{MISHK}/loftr_indoor.ckpt"),
    ],
    # ------------------------------------------------------------------
    "hardnet": [
        ("HardNetPP.pth", "https://github.com/DagnyT/hardnet/raw/master/pretrained/pretrained_all_datasets/HardNet%2B%2B.pth"),
        ("checkpoint_liberty_with_aug.pth", "https://github.com/DagnyT/hardnet/raw/master/pretrained/train_liberty_with_aug/checkpoint_liberty_with_aug.pth"),
        ("hardnet8v2.pt", f"{MISHK}/hardnet8v2.pt"),
    ],
    # ------------------------------------------------------------------
    "hynet": [
        ("HyNet_LIB.pth", "https://github.com/ducha-aiki/Key.Net-Pytorch/raw/main/model/HyNet/weights/HyNet_LIB.pth"),
        ("HyNet_ND.pth", "https://github.com/ducha-aiki/Key.Net-Pytorch/raw/main/model/HyNet/weights/HyNet_ND.pth"),
        ("HyNet_YOS.pth", "https://github.com/ducha-aiki/Key.Net-Pytorch/raw/main/model/HyNet/weights/HyNet_YOS.pth"),
    ],
    # ------------------------------------------------------------------
    "keynet": [
        ("keynet_pytorch.pth", "https://github.com/axelBarroso/Key.Net-Pytorch/raw/main/model/weights/keynet_pytorch.pth"),
    ],
    # ------------------------------------------------------------------
    "affnet": [
        ("AffNet.pth", "https://github.com/ducha-aiki/affnet/raw/master/pretrained/AffNet.pth"),
    ],
    # ------------------------------------------------------------------
    "orinet": [
        ("OriNet.pth", "https://github.com/ducha-aiki/affnet/raw/master/pretrained/OriNet.pth"),
    ],
    # ------------------------------------------------------------------
    "sosnet": [
        ("sosnet_32x32_liberty.pth", "https://github.com/yuruntian/SOSNet/raw/master/sosnet-weights/sosnet_32x32_liberty.pth"),
        ("sosnet_32x32_hpatches_a.pth", "https://github.com/yuruntian/SOSNet/raw/master/sosnet-weights/sosnet_32x32_hpatches_a.pth"),
    ],
    # ------------------------------------------------------------------
    "tfeat": [
        ("tfeat-liberty.params", "https://github.com/vbalnt/tfeat/raw/master/pretrained-models/tfeat-liberty.params"),
        ("tfeat-notredame.params", "https://github.com/vbalnt/tfeat/raw/master/pretrained-models/tfeat-notredame.params"),
        ("tfeat-yosemite.params", "https://github.com/vbalnt/tfeat/raw/master/pretrained-models/tfeat-yosemite.params"),
    ],
    # ------------------------------------------------------------------
    "yunet": [
        ("yunet_final.pth", "https://github.com/kornia/data/raw/main/yunet_final.pth"),
    ],
    # ------------------------------------------------------------------
    "dexined": [
        ("DexiNed_BIPED_10.pth", f"{MISHK}/DexiNed_BIPED_10.pth"),
    ],
    # ------------------------------------------------------------------
    "mobile_sam": [
        ("mobile_sam.pt", "https://github.com/ChaoningZhang/MobileSAM/raw/a509aac54fdd7af59f843135f2f7cee307283c88/weights/mobile_sam.pt"),
    ],
    # ------------------------------------------------------------------
    "rt_detr": [
        ("rtdetr_r18vd_dec3_6x_coco_from_paddle.pth", "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"),
        ("rtdetr_r34vd_dec4_6x_coco_from_paddle.pth", "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth"),
        ("rtdetr_r50vd_m_6x_coco_from_paddle.pth", "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth"),
        ("rtdetr_r50vd_6x_coco_from_paddle.pth", "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth"),
        ("rtdetr_r101vd_6x_coco_from_paddle.pth", "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth"),
    ],
    # ------------------------------------------------------------------
    "tiny_vit": [
        ("tiny_vit_5m_22k_distill.pth", "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_22k_distill.pth"),
        ("tiny_vit_5m_22kto1k_distill.pth", "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_22kto1k_distill.pth"),
        ("tiny_vit_11m_22k_distill.pth", "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_22k_distill.pth"),
        ("tiny_vit_11m_22kto1k_distill.pth", "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_22kto1k_distill.pth"),
        ("tiny_vit_21m_22k_distill.pth", "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22k_distill.pth"),
        ("tiny_vit_21m_22kto1k_distill.pth", "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_distill.pth"),
        ("tiny_vit_21m_22kto1k_384_distill.pth", "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_384_distill.pth"),
        ("tiny_vit_21m_22kto1k_512_distill.pth", "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_512_distill.pth"),
    ],
}


# ---------------------------------------------------------------------------
# Per-model README cards
# ---------------------------------------------------------------------------
MODEL_CARDS: dict[str, str] = {
    "xfeat": """\
---
license: apache-2.0
tags:
  - kornia
  - feature-detection
  - feature-description
  - image-matching
---

# kornia/xfeat

Pretrained weights for **XFeat** (Accelerated Features for Lightweight Image Matching),
used by [`kornia.feature.XFeat`](https://kornia.readthedocs.io/en/latest/feature.html).

XFeat is a lightweight keypoint detector and descriptor optimised for real-time image
matching. It detects sparse keypoints and computes 64-dim descriptors in a single
forward pass with a fully-convolutional architecture. CVPR 2024.

**Original repo:** [verlab/accelerated_features](https://github.com/verlab/accelerated_features)

## Weights

| File | Description |
|------|-------------|
| `xfeat.pt` | Default checkpoint |

## Citation

```bibtex
@inproceedings{potje2024xfeat,
    author    = {Guilherme {Potje} and Felipe {Cadar} and Andre {Araujo}
                 and Renato {Martins} and Erickson R. {Nascimento}},
    title     = {{XFeat}: Accelerated Features for Lightweight Image Matching},
    booktitle = {CVPR},
    year      = {2024}
}
```
""",

    "lightglue": """\
---
license: apache-2.0
tags:
  - kornia
  - feature-matching
---

# kornia/lightglue

Pretrained weights for **LightGlue** (Local Feature Matching at Light Speed),
used by [`kornia.feature.LightGlue`](https://kornia.readthedocs.io/en/latest/feature.html).

LightGlue is a sparse feature matcher built as a pruned transformer that early-exits
unpromising keypoint pairs at each layer, achieving near-SuperGlue accuracy at a
fraction of the latency. ICCV 2023.

**Original repo:** [cvg/LightGlue](https://github.com/cvg/LightGlue)

## Weights

| File | Descriptor |
|------|------------|
| `superpoint_lightglue.pth` | SuperPoint |
| `disk_lightglue.pth` | DISK |
| `aliked_lightglue.pth` | ALIKED |
| `raco_aliked_lightglue.pth` | RaCo-ALIKED |
| `sift_lightglue.pth` | SIFT |
| `doghardnet_lightglue.pth` | DoG-AffNet-HardNet |
| `keynet_affnet_hardnet_lightglue.pth` | Key.Net-AffNet-HardNet |
| `dedodeb_lightglue.pth` | DeDoDe-B |
| `dedodeg_lightglue.pth` | DeDoDe-G |
| `xfeat-lighterglue.pt` | XFeat (LighterGlue) |

## Citation

```bibtex
@article{LightGlue2023,
    author  = {Philipp Lindenberger and Paul-Edouard Sarlin and Marc Pollefeys},
    title   = {{LightGlue}: Local Feature Matching at Light Speed},
    journal = {ICCV},
    year    = {2023}
}
```
""",

    "aliked": """\
---
license: other
tags:
  - kornia
  - feature-detection
  - feature-description
  - image-matching
---

# kornia/aliked

Pretrained weights for **ALIKED** (A Lighter Keypoint and Descriptor Extraction Network
via Deformable Transformation), used by
[`kornia.feature.ALIKED`](https://kornia.readthedocs.io/en/latest/feature.html).

ALIKED uses deformable convolutional heads to detect keypoints that are inherently
invariant to affine deformations, with a sparse descriptor extraction step.
IEEE Transactions on Instrumentation and Measurement, 2023.

**Original repo:** [Shiaoming/ALIKED](https://github.com/Shiaoming/ALIKED)
**Original license:** BSD-3-Clause

## Weights

| File | Variant |
|------|---------|
| `aliked-t16.pth` | Tiny, 16-dim |
| `aliked-n16.pth` | Normal, 16-dim |
| `aliked-n16rot.pth` | Normal, 16-dim, rotation-robust |
| `aliked-n32.pth` | Normal, 32-dim |

## Citation

```bibtex
@article{zhao2023aliked,
    title   = {{ALIKED}: A Lighter Keypoint and Descriptor Extraction Network
               via Deformable Transformation},
    author  = {Xiaoming Zhao and Xingming Wu and Weihai Chen and Peter C.Y. Chen
               and Qingsong Xu and Zhengguo Li},
    journal = {IEEE Transactions on Instrumentation and Measurement},
    year    = {2023}
}
```
""",

    "disk": """\
---
license: mit
tags:
  - kornia
  - feature-detection
  - feature-description
  - image-matching
---

# kornia/disk

Pretrained weights for **DISK** (Learning local features with policy gradient),
used by [`kornia.feature.DISK`](https://kornia.readthedocs.io/en/latest/feature.html).

DISK is trained with policy gradient and produces dense heatmaps of keypoint
probability and per-pixel 128-dim descriptors via a U-Net backbone. NeurIPS 2020.

**Original repo:** [cvlab-epfl/disk](https://github.com/cvlab-epfl/disk)

## Weights

| File | Training supervision |
|------|----------------------|
| `depth-save.pth` | Depth |
| `epipolar-save.pth` | Epipolar geometry |

## Citation

```bibtex
@article{tyszkiewicz2020disk,
    title   = {{DISK}: Learning local features with policy gradient},
    author  = {Tyszkiewicz, Micha{\\l} and Fua, Pascal and Trulls, Eduard},
    journal = {Advances in Neural Information Processing Systems},
    volume  = {33},
    pages   = {14254--14265},
    year    = {2020}
}
```
""",

    "sold2": """\
---
license: mit
tags:
  - kornia
  - line-detection
  - feature-description
---

# kornia/sold2

Pretrained weights for **SOLD²** (Self-supervised Occlusion-aware Line Description
and Detection), used by
[`kornia.feature.SOLD2`](https://kornia.readthedocs.io/en/latest/feature.html).

SOLD² detects line segments and computes semi-dense descriptors along them using a
shared encoder. Trained on wireframe and outdoor datasets. CVPR 2021.

**Original repo:** [cvg/SOLD2](https://github.com/cvg/SOLD2)

## Weights

| File | Training data |
|------|--------------|
| `sold2_wireframe.pth` | Wireframe dataset |

## Citation

```bibtex
@inproceedings{SOLD22021,
    author    = {Pautrat*, Rémi and Lin*, Juan-Ting and Larsson, Viktor
                 and Oswald, Martin R. and Pollefeys, Marc},
    title     = {{SOLD2}: Self-supervised Occlusion-aware Line Description and Detection},
    booktitle = {CVPR},
    year      = {2021}
}
```
""",

    "defmo": """\
---
license: mit
tags:
  - kornia
  - deblurring
  - motion-estimation
---

# kornia/defmo

Pretrained weights for **DeFMO** (Deblurring and Shape Recovery of Fast Moving Objects),
used by [`kornia.feature.DeFMO`](https://kornia.readthedocs.io/en/latest/feature.html).

DeFMO is an encoder–renderer network that takes a blurred image and background frame
as input and predicts the sharp object appearance and trajectory over time. CVPR 2021.

Weights redistributed with permission from Denys Rozumnyi.

**Original repo:** [rozumden/DeFMO](https://github.com/rozumden/DeFMO)

## Weights

| File | Component |
|------|-----------|
| `encoder_best.pt` | Encoder |
| `rendering_best.pt` | Renderer |

## Citation

```bibtex
@inproceedings{DeFMO2021,
    title     = {{DeFMO}: Deblurring and Shape Recovery of Fast Moving Objects},
    author    = {Rozumnyi, Denys and Oswald, Martin R. and Ferrari, Vittorio
                 and Matas, Jiri and Pollefeys, Marc},
    booktitle = {CVPR},
    year      = {2021}
}
```
""",

    "dedode": """\
---
license: mit
tags:
  - kornia
  - feature-detection
  - feature-description
  - image-matching
---

# kornia/dedode

Pretrained weights for **DeDoDe** (Detect, Don't Describe — Describe, Don't Detect),
used by [`kornia.feature.DeDoDe`](https://kornia.readthedocs.io/en/latest/feature.html).

DeDoDe trains detection and description independently: the detector is trained to
find repeatable 3D points, the descriptor is trained separately for matchability.
Supports upright and rotation-equivariant (C4 / SO2) variants. 3DV 2024.

**Original repo:** [Parskatt/DeDoDe](https://github.com/Parskatt/DeDoDe)
**Rotation-equivariant variants:** [georg-bn/rotation-steerers](https://github.com/georg-bn/rotation-steerers)

## Weights

| File | Type | Variant |
|------|------|---------|
| `dedode_detector_L.pth` | Detector | L-upright |
| `dedode_detector_C4.pth` | Detector | L-C4 |
| `dedode_detector_SO2.pth` | Detector | L-SO2 |
| `dedode_detector_L_v2.pth` | Detector | L-upright v2 |
| `dedode_descriptor_B.pth` | Descriptor | B-upright |
| `dedode_descriptor_G.pth` | Descriptor | G-upright |
| `B_C4_Perm_descriptor_setting_C.pth` | Descriptor | B-C4 |
| `B_SO2_Spread_descriptor_setting_C.pth` | Descriptor | B-SO2 |
| `G_C4_Perm_descriptor_setting_C.pth` | Descriptor | G-C4 |
| `G_SO2_Spread_descriptor_setting_C.pth` | Descriptor | G-SO2 |

## Citation

```bibtex
@inproceedings{edstedt2024dedode,
    title     = {{DeDoDe}: Detect, Don't Describe --- Describe, Don't Detect
                 for Local Feature Matching},
    author    = {Johan Edstedt and Georg Bökman and Mårten Wadenbäck
                 and Michael Felsberg},
    booktitle = {3DV},
    year      = {2024}
}
```
""",

    "loftr": """\
---
license: mit
tags:
  - kornia
  - feature-matching
---

# kornia/loftr

Pretrained weights for **LoFTR** (Detector-Free Local Feature Matching with Transformers),
used by [`kornia.feature.LoFTR`](https://kornia.readthedocs.io/en/latest/feature.html).

LoFTR builds dense coarse-to-fine correspondences directly from feature maps using
linear attention, without any keypoint detection step. Trained on outdoor (MegaDepth)
and indoor (ScanNet) datasets. CVPR 2021.

**Original repo:** [zju3dv/LoFTR](https://github.com/zju3dv/LoFTR)

## Weights

| File | Scene type |
|------|-----------|
| `loftr_outdoor.ckpt` | Outdoor (MegaDepth) |
| `loftr_indoor_ds_new.ckpt` | Indoor (ScanNet, updated) |
| `loftr_indoor.ckpt` | Indoor (ScanNet) |

## Citation

```bibtex
@inproceedings{LoFTR2021,
    title     = {{LoFTR}: Detector-Free Local Feature Matching with Transformers},
    author    = {Sun, Jiaming and Shen, Zehong and Wang, Yuang
                 and Bao, Hujun and Zhou, Xiaowei},
    booktitle = {CVPR},
    year      = {2021}
}
```
""",

    "hardnet": """\
---
license: mit
tags:
  - kornia
  - feature-description
  - patch-descriptor
---

# kornia/hardnet

Pretrained weights for **HardNet** and **HardNet++**,
used by [`kornia.feature.HardNet`](https://kornia.readthedocs.io/en/latest/feature.html).

HardNet is a 128-dimensional descriptor for 32×32 grayscale patches, trained with a
hard-negative mining triplet loss. NeurIPS 2017.

**Original repo:** [DagnyT/hardnet](https://github.com/DagnyT/hardnet)

## Weights

| File | Training data |
|------|--------------|
| `HardNetPP.pth` | All Brown datasets (HardNet++) |
| `checkpoint_liberty_with_aug.pth` | Liberty with photometric augmentation |
| `hardnet8v2.pt` | HardNet8 v2, trained on HPatches |

## Citation

```bibtex
@inproceedings{HardNet2017,
    title     = {Working hard to know your neighbor's margins:
                 Local descriptor learning loss},
    author    = {Anastasiya Mishchuk and Dmytro Mishkin
                 and Filip Radenovic and Jiri Matas},
    booktitle = {NeurIPS},
    year      = {2017}
}

@article{HardNet2020,
    title   = {Improving the HardNet Descriptor},
    author  = {Milan Pultar},
    journal = {arXiv ePrint 2007.09699},
    year    = {2020}
}
```
""",

    "hynet": """\
---
license: mit
tags:
  - kornia
  - feature-description
  - patch-descriptor
---

# kornia/hynet

Pretrained weights for **HyNet**,
used by [`kornia.feature.HyNet`](https://kornia.readthedocs.io/en/latest/feature.html).

HyNet is a 128-dimensional patch descriptor that replaces batch normalisation with
Filter Response Normalisation and trains with a hybrid similarity measure combining
L2 distance and cosine similarity under a triplet margin loss. NeurIPS 2020.

**Original repo:** [yuruntian/HyNet](https://github.com/yuruntian/HyNet)

## Weights

| File | Training data |
|------|--------------|
| `HyNet_LIB.pth` | Liberty |
| `HyNet_ND.pth` | Notre Dame |
| `HyNet_YOS.pth` | Yosemite |

## Citation

```bibtex
@inproceedings{hynet2020,
    author    = {Tian, Yurun and Barroso Laguna, Axel and Ng, Tony
                 and Balntas, Vassileios and Mikolajczyk, Krystian},
    title     = {{HyNet}: Learning Local Descriptor with Hybrid Similarity
                 Measure and Triplet Loss},
    booktitle = {NeurIPS},
    year      = {2020}
}
```
""",

    "keynet": """\
---
license: mit
tags:
  - kornia
  - feature-detection
---

# kornia/keynet

Pretrained weights for **Key.Net**,
used by [`kornia.feature.KeyNet`](https://kornia.readthedocs.io/en/latest/feature.html).

Key.Net is a keypoint detector that combines handcrafted (Difference of Gaussians) filters
with learned filters inside a shallow multi-scale CNN, trained to maximise repeatability
under homographic transformations. ICCV 2019.

**Original repo:** [axelBarroso/Key.Net-Pytorch](https://github.com/axelBarroso/Key.Net-Pytorch)

## Weights

| File | Description |
|------|-------------|
| `keynet_pytorch.pth` | Default checkpoint |

## Citation

```bibtex
@inproceedings{KeyNet2019,
    author    = {Barroso-Laguna, Axel and Riba, Edgar
                 and Ponsa, Daniel and Mikolajczyk, Krystian},
    title     = {{Key.Net}: Keypoint Detection by Handcrafted and Learned CNN Filters},
    booktitle = {ICCV},
    year      = {2019}
}
```
""",

    "affnet": """\
---
license: mit
tags:
  - kornia
  - affine-shape-estimation
  - feature-detection
---

# kornia/affnet

Pretrained weights for **AffNet**,
used by [`kornia.feature.LAFAffineShapeEstimator`](https://kornia.readthedocs.io/en/latest/feature.html).

AffNet is a CNN that estimates the 2×2 affine shape matrix for a detected keypoint patch,
making subsequent description invariant to affine transformations. Trained jointly with
HardNet to maximise discriminability of affine-normalised patches. ECCV 2018.

**Original repo:** [ducha-aiki/affnet](https://github.com/ducha-aiki/affnet)

## Weights

| File | Description |
|------|-------------|
| `AffNet.pth` | Default checkpoint |

## Citation

```bibtex
@inproceedings{AffNet2018,
    author    = {D. Mishkin and F. Radenovic and J. Matas},
    title     = {{Repeatability is Not Enough}: Learning Affine Regions
                 via Discriminability},
    booktitle = {ECCV},
    year      = {2018}
}
```
""",

    "orinet": """\
---
license: mit
tags:
  - kornia
  - orientation-estimation
  - feature-detection
---

# kornia/orinet

Pretrained weights for **OriNet**,
used by [`kornia.feature.LAFOrienter`](https://kornia.readthedocs.io/en/latest/feature.html).

OriNet is a CNN that regresses the dominant gradient orientation for a keypoint patch,
trained with a Siamese setup to produce consistent orientations across viewpoints.
Companion to AffNet. ECCV 2018.

**Original repo:** [ducha-aiki/affnet](https://github.com/ducha-aiki/affnet)

## Weights

| File | Description |
|------|-------------|
| `OriNet.pth` | Default checkpoint |

## Citation

```bibtex
@inproceedings{AffNet2018,
    author    = {D. Mishkin and F. Radenovic and J. Matas},
    title     = {{Repeatability is Not Enough}: Learning Affine Regions
                 via Discriminability},
    booktitle = {ECCV},
    year      = {2018}
}
```
""",

    "sosnet": """\
---
license: mit
tags:
  - kornia
  - feature-description
  - patch-descriptor
---

# kornia/sosnet

Pretrained weights for **SOSNet**,
used by [`kornia.feature.SOSNet`](https://kornia.readthedocs.io/en/latest/feature.html).

SOSNet is a 128-dimensional patch descriptor that adds a second-order similarity
regularisation term to the HardNet loss, enforcing that the structure of the descriptor
space reflects patch similarity at multiple scales. CVPR 2019.

**Original repo:** [yuruntian/SOSNet](https://github.com/yuruntian/SOSNet)

## Weights

| File | Training data |
|------|--------------|
| `sosnet_32x32_liberty.pth` | Liberty |
| `sosnet_32x32_hpatches_a.pth` | HPatches-A |

## Citation

```bibtex
@InProceedings{sosnet2019,
    author    = {Tian, Yurong and Yu, Xin and Fan, Bin and Wu, Fuchao
                 and Heijnen, Huub and Balntas, Vassileios},
    title     = {{SOSNet}: Second Order Similarity Regularization for
                 Local Descriptor Learning},
    booktitle = {CVPR},
    year      = {2019}
}
```
""",

    "tfeat": """\
---
license: other
tags:
  - kornia
  - feature-description
  - patch-descriptor
---

# kornia/tfeat

Pretrained weights for **TFeat**,
used by [`kornia.feature.TFeat`](https://kornia.readthedocs.io/en/latest/feature.html).

TFeat is a 128-dimensional patch descriptor trained with a triplet loss and a shallow
two-branch CNN. One of the first learning-based descriptors to outperform SIFT on
standard patch benchmarks. BMVC 2016.

**Original repo:** [vbalnt/tfeat](https://github.com/vbalnt/tfeat)
**Original license:** BSD

## Weights

| File | Training data |
|------|--------------|
| `tfeat-liberty.params` | Liberty |
| `tfeat-notredame.params` | Notre Dame |
| `tfeat-yosemite.params` | Yosemite |

## Citation

```bibtex
@inproceedings{TFeat2016,
    author    = {Vassileios Balntas and Edgar Riba
                 and Daniel Ponsa and Krystian Mikolajczyk},
    title     = {Learning local feature descriptors with triplets
                 and shallow convolutional neural networks},
    booktitle = {BMVC},
    year      = {2016}
}
```
""",

    "yunet": """\
---
license: mit
tags:
  - kornia
  - face-detection
  - object-detection
---

# kornia/yunet

Pretrained weights for **YuNet**,
used by [`kornia.models.FaceDetector`](https://kornia.readthedocs.io/en/latest/models.html).

YuNet is a lightweight face detection CNN based on depthwise-separable convolutions
and a multi-scale FPN head. Runs in real time on CPU. IEEE TBIOM 2021.

**Original repo:** [ShiqiYu/libfacedetection.train](https://github.com/ShiqiYu/libfacedetection.train)

## Weights

| File | Description |
|------|-------------|
| `yunet_final.pth` | Default checkpoint |

## Citation

```bibtex
@article{facedetect-yu,
    author  = {Yuantao Feng and Shiqi Yu and Hanyang Peng
               and Yan-ran Li and Jianguo Zhang},
    title   = {Detect Faces Efficiently: A Survey and Evaluations},
    journal = {IEEE Transactions on Biometrics, Behavior, and Identity Science},
    year    = {2021}
}
```
""",

    "dexined": """\
---
license: mit
tags:
  - kornia
  - edge-detection
---

# kornia/dexined

Pretrained weights for **DexiNed** (Dense Extreme Inception Network for Edge Detection),
used by [`kornia.models.DexiNed`](https://kornia.readthedocs.io/en/latest/models.html).

DexiNed is a fully convolutional network with Inception-inspired blocks that predicts
multi-scale edge maps without pre-training or post-processing. Trained on BIPED.
WACV 2020.

**Original repo:** [xavysp/DexiNed](https://github.com/xavysp/DexiNed)

## Weights

| File | Training data |
|------|--------------|
| `DexiNed_BIPED_10.pth` | BIPED (epoch 10) |

## Citation

```bibtex
@INPROCEEDINGS{xsoria2020dexined,
    author    = {X. Soria and E. Riba and A. Sappa},
    title     = {Dense Extreme Inception Network: Towards a Robust
                 CNN Model for Edge Detection},
    booktitle = {WACV},
    year      = {2020}
}
```
""",

    "mobile_sam": """\
---
license: apache-2.0
tags:
  - kornia
  - image-segmentation
---

# kornia/mobile_sam

Pretrained weights for **MobileSAM**,
used by [`kornia.models.SegmentAnything`](https://kornia.readthedocs.io/en/latest/models.html).

MobileSAM replaces the heavy ViT-H image encoder of SAM with a distilled TinyViT-5M
encoder, reducing model size by 60× while retaining most of SAM's segmentation quality.
arXiv 2023.

**Original repo:** [ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)

## Weights

| File | Description |
|------|-------------|
| `mobile_sam.pt` | TinyViT-5M encoder + SAM decoder |

## Citation

```bibtex
@article{zhang2023mobilesam,
    title   = {Faster Segment Anything: Towards Lightweight SAM
               for Mobile Applications},
    author  = {Zhang, Chaoning and Han, Dongshen and Qiao, Yu
               and Kim, Jung Uk and Bae, Sung-Ho and Lee, Seungkyu
               and Hong, Choong Seon},
    journal = {arXiv preprint arXiv:2306.14289},
    year    = {2023}
}
```
""",

    "rt_detr": """\
---
license: apache-2.0
tags:
  - kornia
  - object-detection
---

# kornia/rt_detr

Pretrained weights for **RT-DETR** (Real-Time Detection Transformer),
used by [`kornia.models.RTDETRDetector`](https://kornia.readthedocs.io/en/latest/models.html).

RT-DETR replaces traditional NMS post-processing with a transformer decoder and an
efficient hybrid encoder combining a multi-scale convolutional backbone with an
intra-scale interaction module. Weights converted from PaddleDetection. CVPR 2024.

**Original repo:** [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)

## Weights

| File | Backbone | COCO AP |
|------|----------|---------|
| `rtdetr_r18vd_dec3_6x_coco_from_paddle.pth` | ResNet-18D | 46.4 |
| `rtdetr_r34vd_dec4_6x_coco_from_paddle.pth` | ResNet-34D | 48.9 |
| `rtdetr_r50vd_m_6x_coco_from_paddle.pth` | ResNet-50D-M | 51.3 |
| `rtdetr_r50vd_6x_coco_from_paddle.pth` | ResNet-50D | 53.1 |
| `rtdetr_r101vd_6x_coco_from_paddle.pth` | ResNet-101D | 54.3 |

## Citation

```bibtex
@inproceedings{zhao2024rtdetr,
    title     = {{DETRs} Beat {YOLOs} on Real-time Object Detection},
    author    = {Zhao, Yian and Lv, Wenyu and Xu, Shangliang
                 and Wei, Jinman and Wang, Guanzhong and Dang, Qingqing
                 and Liu, Yi and Chen, Jie},
    booktitle = {CVPR},
    year      = {2024}
}
```
""",

    "tiny_vit": """\
---
license: mit
tags:
  - kornia
  - image-classification
  - backbone
---

# kornia/tiny_vit

Pretrained weights for **TinyViT**,
used as the encoder backbone in
[`kornia.models.SegmentAnything`](https://kornia.readthedocs.io/en/latest/models.html)
(MobileSAM) and available via
[`kornia.models.TinyViT`](https://kornia.readthedocs.io/en/latest/models.html).

TinyViT is a small Vision Transformer trained with knowledge distillation from large
teacher models on ImageNet-22K. ECCV 2022.

**Original repo:** [microsoft/Cream/TinyViT](https://github.com/microsoft/Cream/tree/main/TinyViT)

## Weights

| File | Params | Pre-training | Fine-tuning |
|------|--------|-------------|-------------|
| `tiny_vit_5m_22k_distill.pth` | 5M | ImageNet-22K | — |
| `tiny_vit_5m_22kto1k_distill.pth` | 5M | ImageNet-22K | ImageNet-1K 224 |
| `tiny_vit_11m_22k_distill.pth` | 11M | ImageNet-22K | — |
| `tiny_vit_11m_22kto1k_distill.pth` | 11M | ImageNet-22K | ImageNet-1K 224 |
| `tiny_vit_21m_22k_distill.pth` | 21M | ImageNet-22K | — |
| `tiny_vit_21m_22kto1k_distill.pth` | 21M | ImageNet-22K | ImageNet-1K 224 |
| `tiny_vit_21m_22kto1k_384_distill.pth` | 21M | ImageNet-22K | ImageNet-1K 384 |
| `tiny_vit_21m_22kto1k_512_distill.pth` | 21M | ImageNet-22K | ImageNet-1K 512 |

## Citation

```bibtex
@inproceedings{wu2022tinyvit,
    title     = {{TinyViT}: Fast Pretraining Distillation for Small Vision Transformers},
    author    = {Wu, Kan and Zhang, Jinnian and Peng, Houwen and Liu, Mengchen
                 and Xiao, Bin and Fu, Jianlong and Yuan, Lu},
    booktitle = {ECCV},
    year      = {2022}
}
```
""",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Stream-download *url* into *dest*, showing a tqdm progress bar."""
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0)) or None
    with dest.open("wb") as fh, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=dest.name,
        leave=False,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            fh.write(chunk)
            bar.update(len(chunk))


def upload_weights(
    models: list[str],
    *,
    dry_run: bool = False,
    token: str | None = None,
) -> None:
    token = token or os.environ.get("HF_TOKEN")
    if not token and not dry_run:
        sys.exit("HF_TOKEN not set. Export it or pass --token.")

    api = HfApi(token=token)

    for model in models:
        if model not in WEIGHTS:
            print(f"[WARN] Unknown model '{model}', skipping.")
            continue

        repo_id = f"{HF_ORG}/{model}"
        files = WEIGHTS[model]
        card = MODEL_CARDS.get(model, "")

        print(f"\n{'='*60}")
        print(f"Repo: {repo_id}  ({len(files)} file(s))")
        print(f"{'='*60}")

        if not dry_run:
            api.create_repo(repo_id, repo_type=REPO_TYPE, exist_ok=True, private=False)

            # Upload/refresh the model card
            with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as tmp:
                tmp.write(card)
                tmp_path = tmp.name
            api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type=REPO_TYPE,
                commit_message="Update README",
            )
            Path(tmp_path).unlink()

            # Check existing files to allow resume
            try:
                existing = {f.rfilename for f in api.list_repo_tree(repo_id, repo_type=REPO_TYPE)}
            except Exception:
                existing = set()
        else:
            existing = set()

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, (filename, url) in enumerate(files, 1):
                prefix = f"  [{i}/{len(files)}] {filename}"

                if filename in existing:
                    print(f"{prefix}  — already exists, skipping")
                    continue

                if dry_run:
                    print(f"{prefix}  — DRY RUN  ({url})")
                    continue

                local = Path(tmpdir) / filename
                print(f"{prefix}  — downloading …")
                try:
                    _download(url, local)
                except Exception as exc:
                    print(f"    ERROR downloading: {exc}")
                    continue

                print(f"{prefix}  — uploading …")
                try:
                    api.upload_file(
                        path_or_fileobj=str(local),
                        path_in_repo=filename,
                        repo_id=repo_id,
                        repo_type=REPO_TYPE,
                        commit_message=f"Add {filename}",
                    )
                    print(f"{prefix}  — done")
                except Exception as exc:
                    print(f"    ERROR uploading: {exc}")
                finally:
                    local.unlink(missing_ok=True)

    print("\nAll done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    all_models = sorted(WEIGHTS)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        nargs="+",
        metavar="MODEL",
        default=all_models,
        help=f"Which model(s) to upload (default: all). Choices: {all_models}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually doing it.",
    )
    parser.add_argument("--token", default=None, help="HuggingFace write token (falls back to $HF_TOKEN).")
    args = parser.parse_args()

    upload_weights(args.model, dry_run=args.dry_run, token=args.token)


if __name__ == "__main__":
    main()
