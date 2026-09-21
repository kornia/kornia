# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""ONNX export survey — kornia.augmentation (deterministic params-fed + random-mode).

PART 1 (group augmentation*, deterministic): ``aug(img, params=params)`` with the sampled parameter
tensors fed as ONNX inputs (wrapper ``Det``); containers/auto are called once eagerly and then
exported with ``params=seq._params`` held constant.
PART 2 (group *.random): ``aug(img)`` in train mode, randomness inside the graph, ``check=False``.
"""

from __future__ import annotations

import sys
import traceback

import torch
from harness import case, run_cases
from torch import nn

import kornia as K
import kornia.augmentation as KA
from kornia.augmentation.auto import AutoAugment, RandAugment, TrivialAugment

torch.manual_seed(0)

IMG = torch.rand(2, 3, 32, 40)
IMG1 = torch.rand(1, 3, 32, 40)
VOL = torch.rand(2, 1, 8, 16, 16)
LABEL = torch.tensor([0, 1])
MASK = (torch.rand(2, 1, 32, 40) > 0.5).float()
MASK_INT = torch.randint(0, 3, (2, 32, 40))
MASK3D_INT = torch.randint(0, 3, (2, 8, 16, 16))
KPTS = torch.stack([torch.rand(2, 5) * 39, torch.rand(2, 5) * 31], -1)  # (B, N, 2) inside the image
BBOX_XYXY = torch.tensor([[[2.0, 3.0, 20.0, 25.0], [10.0, 5.0, 38.0, 30.0]]]).repeat(2, 1, 1)  # (B, 2, 4)
VIDEO = torch.rand(2, 4, 3, 16, 20)  # BTCHW

NONPARAM_KEYS = ("batch_prob", "forward_input_shape")


class Det(nn.Module):
    """Call ``aug(*imgs, params=p)`` with the tensor-valued params fed as positional inputs."""

    def __init__(self, aug: nn.Module, params: dict, keys: list[str], n_data: int = 1) -> None:
        super().__init__()
        self.aug = aug
        self.const = {k: v for k, v in params.items() if k not in keys}
        self.keys = keys
        self.n_data = n_data

    def forward(self, *xs):
        data, ps = xs[: self.n_data], xs[self.n_data :]
        p = dict(self.const)
        p.update(zip(self.keys, ps))
        return self.aug(*data, params=p)


class Rand(nn.Module):
    """Call ``aug(*data)`` — parameters are sampled inside the graph (train mode kept)."""

    def __init__(self, aug: nn.Module, n_data: int = 1) -> None:
        super().__init__()
        self.aug = aug
        self.n_data = n_data

    def forward(self, *xs):
        out = self.aug(*xs[: self.n_data])
        return out

    def eval(self):  # keep the augmentation in train mode (harness calls .eval())
        return self


CASES: list = []


def build(
    name: str,
    group: str,
    ctor,
    data: list[torch.Tensor],
    *,
    note: str = "",
    tags=(),
    det: bool = True,
    rand: bool = True,
    rand_note: str = "",
    direct: bool = False,
    atol: float = 2e-4,
    check_det: bool = True,
    skip_rand: str | None = None,
    const_keys: tuple = (),
):
    """Add a deterministic (params-fed) case and a random-mode case for ``ctor()``.

    ``direct=True``: no random params -> deterministic case is a plain ``aug(*data)`` call.
    """
    tags = tuple(tags)
    if det:
        try:
            aug = ctor()
            if direct:
                CASES.append(
                    case(
                        name,
                        group,
                        Rand(aug, len(data)),
                        list(data),
                        note=note or "no random params; aug(img) direct",
                        tags=tags,
                        atol=atol,
                    )
                )
            else:
                torch.manual_seed(7)
                shape = data[0].shape
                params = aug.forward_parameters(shape)
                if hasattr(aug, "data_keys") and "dtype" not in params:  # mix augmentations record dtype
                    from kornia.constants import DType

                    params.update({"dtype": torch.tensor(DType.get(data[0].dtype).value)})
                keys = [
                    k
                    for k in params
                    if k not in NONPARAM_KEYS and torch.is_tensor(params[k]) and k != "dtype" and k not in const_keys
                ]
                n = f"params fed as inputs: {keys}" if keys else "no tensor params (only batch_prob/shape, constant)"
                CASES.append(
                    case(
                        name,
                        group,
                        Det(aug, params, keys, len(data)),
                        list(data) + [params[k] for k in keys],
                        note=(note + "; " if note else "") + n,
                        tags=tags,
                        atol=atol,
                        check=check_det,
                    )
                )
        except Exception as e:
            CASES.append(
                case(
                    name,
                    group,
                    None,
                    list(data),
                    skip=f"CONSTRUCTION/PARAM FAILURE (spec): {type(e).__name__}: {e}"[:300],
                    note=note,
                    tags=tags,
                )
            )
            traceback.print_exc()
    if rand:
        try:
            aug = ctor()
            CASES.append(
                case(
                    f"{name}[random]",
                    group + ".random",
                    Rand(aug, len(data)),
                    list(data),
                    check=False,
                    note=rand_note or "aug(img) in train mode, randomness inside the graph",
                    tags=tags,
                    skip=skip_rand,
                )
            )
        except Exception as e:
            CASES.append(
                case(
                    f"{name}[random]",
                    group + ".random",
                    None,
                    list(data),
                    skip=f"CONSTRUCTION FAILURE (spec): {type(e).__name__}: {e}"[:300],
                    tags=tags,
                )
            )


def seq_build(
    name: str, group: str, ctor, data: list[torch.Tensor], *, note: str = "", tags=(), rand: bool = True, call=None
):
    """Container recipe: run once eagerly to populate ``_params``; export with params constant."""
    tags = tuple(tags)
    try:
        seq = ctor()
        torch.manual_seed(7)
        with torch.no_grad():
            seq(*data)
        params = seq._params

        class SeqDet(nn.Module):
            def __init__(self):
                super().__init__()
                self.seq = seq

            def forward(self, *xs):
                return self.seq(*xs, params=params)

        CASES.append(
            case(
                name,
                group,
                SeqDet(),
                list(data),
                tags=tags,
                note=(note + "; " if note else "") + "params=seq._params held CONSTANT (only image is a live input)",
            )
        )
    except Exception as e:
        CASES.append(
            case(
                name,
                group,
                None,
                list(data),
                skip=f"CONSTRUCTION/EAGER FAILURE (spec): {type(e).__name__}: {e}"[:300],
                note=note,
                tags=tags,
            )
        )
        traceback.print_exc()
    if rand:
        try:
            seq = ctor()
            CASES.append(
                case(
                    f"{name}[random]",
                    group + ".random",
                    Rand(seq, len(data)),
                    list(data),
                    check=False,
                    tags=tags,
                    note="container called directly; parameters sampled inside the graph",
                )
            )
        except Exception as e:
            CASES.append(
                case(
                    f"{name}[random]",
                    group + ".random",
                    None,
                    list(data),
                    skip=f"CONSTRUCTION FAILURE (spec): {type(e).__name__}: {e}"[:300],
                    tags=tags,
                )
            )


# ----------------------------------------------------------------------------- PART 1+2: 2D not yet covered
G2 = "augmentation"
build(
    "Normalize",
    G2,
    lambda: KA.Normalize(mean=torch.tensor([0.5, 0.4, 0.3]), std=torch.tensor([0.2, 0.25, 0.3])),
    [IMG],
    direct=True,
)
build(
    "Denormalize",
    G2,
    lambda: KA.Denormalize(mean=torch.tensor([0.5, 0.4, 0.3]), std=torch.tensor([0.2, 0.25, 0.3])),
    [IMG],
    direct=True,
)
build("Resize", G2, lambda: KA.Resize((24, 36)), [IMG], direct=True, note="size (24,36) baked")
build(
    "Resize[antialias]",
    G2,
    lambda: KA.Resize((16, 20), antialias=True),
    [IMG],
    direct=True,
    note="antialias=True",
    rand=False,
)
build("LongestMaxSize", G2, lambda: KA.LongestMaxSize(60), [IMG], direct=True, note="max_size=60 baked")
build("SmallestMaxSize", G2, lambda: KA.SmallestMaxSize(24), [IMG], direct=True, note="max_size=24 baked")
build(
    "LongestMaxSize[const-size]",
    G2,
    lambda: KA.LongestMaxSize(60),
    [IMG],
    rand=False,
    const_keys=("output_size",),
    note="output_size param held constant (not a graph input)",
)
build(
    "SmallestMaxSize[const-size]",
    G2,
    lambda: KA.SmallestMaxSize(24),
    [IMG],
    rand=False,
    const_keys=("output_size",),
    note="output_size param held constant (not a graph input)",
)
build("PadTo", G2, lambda: KA.PadTo((40, 48)), [IMG], direct=True, note="size (40,48) baked")
build(
    "PadTo[reflect]",
    G2,
    lambda: KA.PadTo((40, 48), pad_mode="reflect"),
    [IMG],
    direct=True,
    note="pad_mode=reflect",
    rand=False,
)
build("RandomRotation90", G2, lambda: KA.RandomRotation90(times=(1, 3), p=1.0), [IMG])
build("RandomShear", G2, lambda: KA.RandomShear(shear=10.0, p=1.0), [IMG])
build("RandomTranslate", G2, lambda: KA.RandomTranslate((0.1, 0.2), (0.1, 0.2), p=1.0), [IMG])
build("RandomChannelDropout[b1]", G2, lambda: KA.RandomChannelDropout(p=1.0), [IMG1], rand=False, note="batch 1")

# ----------------------------------------------------------------------------- PART 2 only: 2D from the
# previous pass (random mode)
PREV2D = {
    "ColorJiggle": lambda: KA.ColorJiggle(0.3, 0.3, 0.3, 0.3, p=1.0),
    "ColorJitter": lambda: KA.ColorJitter(0.3, 0.3, 0.3, 0.3, p=1.0),
    "RandomAffine": lambda: KA.RandomAffine((-15.0, 20.0), (0.1, 0.1), (0.7, 1.3), 20, p=1.0),
    "RandomBoxBlur": lambda: KA.RandomBoxBlur((7, 7), p=1.0),
    "RandomBrightness": lambda: KA.RandomBrightness((0.0, 1.0), p=1.0),
    "RandomContrast": lambda: KA.RandomContrast((0.0, 1.0), p=1.0),
    "RandomChannelDropout": lambda: KA.RandomChannelDropout(p=1.0),
    "RandomChannelShuffle": lambda: KA.RandomChannelShuffle(p=1.0),
    "RandomElasticTransform": lambda: KA.RandomElasticTransform((63, 63), (32, 32), (2.0, 2.0), p=1.0),
    "RandomEqualize": lambda: KA.RandomEqualize(p=1.0),
    "RandomErasing": lambda: KA.RandomErasing((0.2, 0.4), (0.3, 1 / 0.3), p=1.0),
    "RandomFisheye": lambda: KA.RandomFisheye(
        torch.tensor([-0.3, 0.3]), torch.tensor([-0.3, 0.3]), torch.tensor([0.9, 1.0]), p=1.0
    ),
    "RandomGamma": lambda: KA.RandomGamma((0.0, 1.0), p=1.0),
    "RandomGaussianBlur": lambda: KA.RandomGaussianBlur((3, 3), (0.1, 2.0), p=1.0),
    "RandomGaussianIllumination": lambda: KA.RandomGaussianIllumination(
        (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (-1.0, 1.0), p=1.0
    ),
    "RandomGaussianNoise": lambda: KA.RandomGaussianNoise(0.0, 0.05, p=1.0),
    "RandomGrayscale": lambda: KA.RandomGrayscale(p=1.0),
    "RandomHue": lambda: KA.RandomHue((-0.5, 0.5), p=1.0),
    "RandomHorizontalFlip": lambda: KA.RandomHorizontalFlip(p=1.0),
    "RandomVerticalFlip": lambda: KA.RandomVerticalFlip(p=1.0),
    "RandomInvert": lambda: KA.RandomInvert(p=1.0),
    "RandomJPEG": lambda: KA.RandomJPEG((1.0, 5.0), p=1.0),
    "RandomLinearCornerIllumination": lambda: KA.RandomLinearCornerIllumination((0.5, 0.5), (-1.0, 1.0), p=1.0),
    "RandomLinearIllumination": lambda: KA.RandomLinearIllumination((0.5, 0.5), (-1.0, 1.0), p=1.0),
    "RandomMedianBlur": lambda: KA.RandomMedianBlur((3, 3), p=1.0),
    "RandomMotionBlur": lambda: KA.RandomMotionBlur(7, 35.0, 0.5, p=1.0),
    "RandomPerspective": lambda: KA.RandomPerspective(0.2, p=1.0),
    "RandomPlanckianJitter": lambda: KA.RandomPlanckianJitter(p=1.0),
    "RandomPlasmaShadow": lambda: KA.RandomPlasmaShadow((0.2, 0.5), p=1.0),
    "RandomPlasmaBrightness": lambda: KA.RandomPlasmaBrightness(p=1.0),
    "RandomPlasmaContrast": lambda: KA.RandomPlasmaContrast(p=1.0),
    "RandomPosterize": lambda: KA.RandomPosterize((1, 4), p=1.0),
    "RandomRotation": lambda: KA.RandomRotation(45.0, p=1.0),
    "RandomSaltAndPepperNoise": lambda: KA.RandomSaltAndPepperNoise((0.05, 0.5), (0.1, 0.7), p=1.0),
    "RandomSaturation": lambda: KA.RandomSaturation((0.5, 5.0), p=1.0),
    "RandomSharpness": lambda: KA.RandomSharpness(16.0, p=1.0),
    "RandomSolarize": lambda: KA.RandomSolarize(0.2, 0.2, p=1.0),
    "RandomThinPlateSpline": lambda: KA.RandomThinPlateSpline(p=1.0),
    "RandomCrop": lambda: KA.RandomCrop((24, 32), p=1.0),
    "RandomResizedCrop": lambda: KA.RandomResizedCrop((24, 32), p=1.0),
    "CenterCrop": lambda: KA.CenterCrop((24, 32)),
    "RandomRain": lambda: KA.RandomRain(p=1.0),
    "RandomSnow": lambda: KA.RandomSnow(p=1.0),
    "RandomRGBShift": lambda: KA.RandomRGBShift(p=1.0),
    "RandomAutoContrast": lambda: KA.RandomAutoContrast(p=1.0),
    "RandomClahe": lambda: KA.RandomClahe(p=1.0),
    "RandomDissolving": lambda: KA.RandomDissolving(p=1.0),
}
for _n, _c in PREV2D.items():
    build(_n, G2, _c, [IMG], det=False)
build(
    "ColorJitter[fixed order]",
    G2,
    lambda: KA.ColorJitter(0.3, 0.3, 0.3, 0.3, p=1.0, order=(0, 1, 2, 3)),
    [IMG],
    det=False,
    rand_note="order=(0, 1, 2, 3): the per-call random operation order is the only export blocker",
)

# default p=0.5 for 5 representative ones (batch_prob branching becomes live)
P05 = {
    "RandomHorizontalFlip": lambda: KA.RandomHorizontalFlip(p=0.5),
    "ColorJiggle": lambda: KA.ColorJiggle(0.3, 0.3, 0.3, 0.3, p=0.5),
    "RandomAffine": lambda: KA.RandomAffine((-15.0, 20.0), (0.1, 0.1), (0.7, 1.3), 20, p=0.5),
    "RandomGaussianBlur": lambda: KA.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.5),
    "RandomCrop": lambda: KA.RandomCrop((24, 32), p=0.5),
}
for _n, _c in P05.items():
    build(f"{_n}[p=0.5]", G2, _c, [IMG], det=False, rand_note="default p=0.5: per-sample batch_prob gate live in graph")

# ----------------------------------------------------------------------------- 3D
G3 = "augmentation3d"
build(
    "RandomAffine3D",
    G3,
    lambda: KA.RandomAffine3D(15.0, translate=(0.1, 0.1, 0.1), scale=(0.8, 1.2), p=1.0),
    [VOL],
    tags=("3d",),
)
build("RandomCrop3D", G3, lambda: KA.RandomCrop3D((6, 12, 12), p=1.0), [VOL], tags=("3d",), note="size (6,12,12) baked")
build("RandomRotation3D", G3, lambda: KA.RandomRotation3D(15.0, p=1.0), [VOL], tags=("3d",))
build("RandomPerspective3D", G3, lambda: KA.RandomPerspective3D(0.2, p=1.0), [VOL], tags=("3d",))
build("RandomMotionBlur3D", G3, lambda: KA.RandomMotionBlur3D(3, 35.0, 0.5, p=1.0), [VOL], tags=("3d",))
build("RandomEqualize3D", G3, lambda: KA.RandomEqualize3D(p=1.0), [VOL], tags=("3d",))
build("RandomDepthicalFlip3D", G3, lambda: KA.RandomDepthicalFlip3D(p=1.0), [VOL], tags=("3d",))
build("RandomHorizontalFlip3D", G3, lambda: KA.RandomHorizontalFlip3D(p=1.0), [VOL], tags=("3d",))
build("RandomVerticalFlip3D", G3, lambda: KA.RandomVerticalFlip3D(p=1.0), [VOL], tags=("3d",))
build("CenterCrop3D", G3, lambda: KA.CenterCrop3D((6, 12, 12)), [VOL], direct=True, tags=("3d",), note="size baked")
build(
    "RandomTransplantation3D",
    G3,
    lambda: KA.RandomTransplantation3D(p=1.0),
    [VOL, MASK3D_INT],
    tags=("3d", "mix"),
    note="inputs (volume, int mask)",
)

# ----------------------------------------------------------------------------- mix
GM = "augmentation.mix"
build("RandomMixUpV2", GM, lambda: KA.RandomMixUpV2(p=1.0), [IMG, LABEL], tags=("mix",), note="inputs (image, label)")
build("RandomCutMixV2", GM, lambda: KA.RandomCutMixV2(p=1.0), [IMG, LABEL], tags=("mix",), note="inputs (image, label)")
build(
    "RandomMosaic",
    GM,
    lambda: KA.RandomMosaic(output_size=(32, 40), p=1.0, data_keys=["input"]),
    [IMG],
    tags=("mix",),
    note="output_size (32,40) baked; data_keys=[input]",
)
build(
    "RandomMosaic[bbox]",
    GM,
    lambda: KA.RandomMosaic(output_size=(32, 40), p=1.0, data_keys=["input", "bbox_xyxy"]),
    [IMG, BBOX_XYXY],
    tags=("mix",),
    note="data_keys=[input,bbox_xyxy]",
)
build("RandomJigsaw", GM, lambda: KA.RandomJigsaw(grid=(2, 2), p=1.0), [IMG], tags=("mix",), note="grid (2,2) baked")
build(
    "RandomTransplantation",
    GM,
    lambda: KA.RandomTransplantation(p=1.0),
    [IMG, MASK_INT],
    tags=("mix",),
    note="inputs (image, int mask)",
)
build("PatchMix", GM, lambda: KA.PatchMix(patch_size=8, p=1.0), [IMG, LABEL], tags=("mix",), note="patch_size=8 baked")

# ----------------------------------------------------------------------------- containers
GC = "augmentation.container"


def _augs():
    # children that export on their own (ColorJiggle does not: `order.tolist()` indexing) -> container
    # mechanics are what is tested
    return [KA.RandomHorizontalFlip(p=1.0), KA.RandomBrightness((0.8, 1.2), p=1.0), KA.RandomAffine(15.0, p=1.0)]


seq_build(
    "AugmentationSequential",
    GC,
    lambda: KA.AugmentationSequential(*_augs(), data_keys=["input"]),
    [IMG],
    note="HFlip(p=1)+RandomBrightness(p=1)+RandomAffine(p=1)",
)
seq_build(
    "AugmentationSequential[ColorJiggle]",
    GC,
    lambda: KA.AugmentationSequential(
        KA.RandomHorizontalFlip(p=1.0),
        KA.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
        KA.RandomAffine(15.0, p=1.0),
        data_keys=["input"],
    ),
    [IMG],
    note="same with a ColorJiggle child (expected to inherit ColorJiggle's order.tolist() blocker)",
)
seq_build(
    "AugmentationSequential[p=0.5]",
    GC,
    lambda: KA.AugmentationSequential(
        KA.RandomHorizontalFlip(p=0.5),
        KA.RandomBrightness((0.8, 1.2), p=0.5),
        KA.RandomAffine(15.0, p=0.5),
        data_keys=["input"],
    ),
    [IMG],
    note="same pipeline with p=0.5 (batch_prob gates live)",
)
seq_build(
    "AugmentationSequential[mask]",
    GC,
    lambda: KA.AugmentationSequential(*_augs(), data_keys=["input", "mask"]),
    [IMG, MASK],
    note="data_keys=[input, mask]",
)
seq_build(
    "AugmentationSequential[keypoints]",
    GC,
    lambda: KA.AugmentationSequential(*_augs(), data_keys=["input", "keypoints"]),
    [IMG, KPTS],
    note="data_keys=[input, keypoints]",
    tags=("points",),
)
seq_build(
    "AugmentationSequential[bbox_xyxy]",
    GC,
    lambda: KA.AugmentationSequential(*_augs(), data_keys=["input", "bbox_xyxy"]),
    [IMG, BBOX_XYXY],
    note="data_keys=[input, bbox_xyxy]",
)
seq_build(
    "AugmentationSequential[crop]",
    GC,
    lambda: KA.AugmentationSequential(
        KA.RandomResizedCrop((24, 32), p=1.0), KA.RandomHorizontalFlip(p=1.0), data_keys=["input"]
    ),
    [IMG],
    note="shape-changing child (RandomResizedCrop)",
)
seq_build(
    "ImageSequential",
    GC,
    lambda: KA.ImageSequential(
        KA.RandomBrightness((0.8, 1.2), p=1.0), K.color.RgbToBgr(), KA.RandomHorizontalFlip(p=1.0)
    ),
    [IMG],
    note="RandomBrightness + kornia.color.RgbToBgr + HFlip",
)
seq_build(
    "ImageSequential[random_apply]",
    GC,
    lambda: KA.ImageSequential(
        KA.RandomBrightness((0.8, 1.2), p=1.0),
        KA.RandomHorizontalFlip(p=1.0),
        KA.RandomGrayscale(p=1.0),
        random_apply=2,
    ),
    [IMG],
    note="random_apply=2 (random subset/order of children)",
)
seq_build(
    "PatchSequential",
    GC,
    lambda: KA.PatchSequential(
        KA.RandomBrightness((0.8, 1.2), p=1.0),
        KA.RandomHorizontalFlip(p=1.0),
        KA.RandomGrayscale(p=1.0),
        KA.RandomSolarize(0.1, 0.1, p=1.0),
        grid_size=(2, 2),
        patchwise_apply=True,
    ),
    [IMG],
    note="grid_size (2,2), patchwise_apply=True (one module per patch)",
)
seq_build(
    "PatchSequential[not-patchwise]",
    GC,
    lambda: KA.PatchSequential(
        KA.RandomBrightness((0.8, 1.2), p=1.0), KA.RandomHorizontalFlip(p=1.0), grid_size=(2, 2), patchwise_apply=False
    ),
    [IMG],
    note="patchwise_apply=False",
)
seq_build(
    "VideoSequential",
    GC,
    lambda: KA.VideoSequential(
        KA.RandomAffine(15.0, p=1.0), KA.RandomBrightness((0.8, 1.2), p=1.0), data_format="BTCHW", same_on_frame=True
    ),
    [VIDEO],
    note="BTCHW (2,4,3,16,20), same_on_frame=True",
)
seq_build(
    "VideoSequential[BCTHW]",
    GC,
    lambda: KA.VideoSequential(
        KA.RandomAffine(15.0, p=1.0), KA.RandomBrightness((0.8, 1.2), p=1.0), data_format="BCTHW", same_on_frame=False
    ),
    [VIDEO.permute(0, 2, 1, 3, 4).contiguous()],
    note="BCTHW, same_on_frame=False",
)


def _disp_seq():
    return KA.AugmentationSequential(
        KA.RandomBrightness((0.8, 1.2), p=1.0), KA.RandomAffine(15.0, p=1.0), data_keys=["input", "mask"]
    )


def _many2many():
    d = KA.ManyToManyAugmentationDispather(_disp_seq(), _disp_seq())
    torch.manual_seed(7)
    with torch.no_grad():
        d((IMG, MASK), (IMG1.repeat(2, 1, 1, 1), MASK))
    ps = [a._params for a in d.augmentations]

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.d = d

        def forward(self, a, b, c, e):
            return [aug(*inp, params=p) for inp, aug, p in zip(((a, b), (c, e)), self.d.augmentations, ps)]

    return M()


def _many2one():
    d = KA.ManyToOneAugmentationDispather(_disp_seq(), _disp_seq())
    torch.manual_seed(7)
    with torch.no_grad():
        d(IMG, MASK)
    ps = [a._params for a in d.augmentations]

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.d = d

        def forward(self, a, b):
            return [aug(a, b, params=p) for aug, p in zip(self.d.augmentations, ps)]

    return M()


CASES.append(
    case(
        "ManyToManyAugmentationDispather",
        GC,
        _many2many(),
        [IMG, MASK, IMG1.repeat(2, 1, 1, 1), MASK],
        note="two AugmentationSequential(RandomBrightness, RandomAffine; input+mask); dispatcher forward has no params "
        "arg, so each child is called with its own _params constant (public API replayed manually)",
    )
)
CASES.append(
    case(
        "ManyToManyAugmentationDispather[random]",
        GC + ".random",
        (lambda d: lambda a, b, c, e: d((a, b), (c, e)))(KA.ManyToManyAugmentationDispather(_disp_seq(), _disp_seq())),
        [IMG, MASK, IMG1.repeat(2, 1, 1, 1), MASK],
        check=False,
        note="dispatcher called directly",
    )
)
CASES.append(
    case(
        "ManyToOneAugmentationDispather",
        GC,
        _many2one(),
        [IMG, MASK],
        note="two AugmentationSequential on the same (input, mask); children replayed with constant _params",
    )
)
CASES.append(
    case(
        "ManyToOneAugmentationDispather[random]",
        GC + ".random",
        (lambda d: lambda a, b: d(a, b))(KA.ManyToOneAugmentationDispather(_disp_seq(), _disp_seq())),  # noqa: PLW0108
        [IMG, MASK],
        check=False,
        note="dispatcher called directly",
    )
)

# ----------------------------------------------------------------------------- auto
GA = "augmentation.auto"
seq_build(
    "RandAugment",
    GA,
    lambda: RandAugment(n=2, m=10),
    [IMG],
    note="n=2, m=10; policy chosen at eager call, then constant",
)
seq_build(
    "AutoAugment",
    GA,
    AutoAugment,
    [IMG],
    note="imagenet policy; sub-policy chosen at eager call, then constant",
)
seq_build("TrivialAugment", GA, TrivialAugment, [IMG], note="one op chosen at eager call, then constant")
seq_build(
    "AugmentationSequential[RandAugment]",
    GA,
    lambda: KA.AugmentationSequential(RandAugment(n=2, m=10)),
    [IMG],
    note="RandAugment wrapped in AugmentationSequential (doc recipe)",
)

if __name__ == "__main__":
    run_cases(CASES, sys.argv[1], only=sys.argv[2:] or None)
