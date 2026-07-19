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
"""``torch.export`` tracking test for deterministic kornia.augmentation transforms.

Distinct from ``torch.onnx.export`` (see ``test_onnx_export.py``): ``torch.export.export``
is the newer full-graph capture that downstream projects rely on to ship a model together
with its preprocessing (Normalize / Resize / CenterCrop / Pad) as a single exported program.
TorchGeo migrated its inference-time transforms off kornia onto ``torchvision.transforms.v2``
citing exactly this gap ("kornia augmentations can't be torch.exported").

The augmentation base ``forward`` stashes per-call state on ``self`` (``_params`` and the lazy
transform matrix), which ``torch.export`` on torch <= 2.9 rejects ("attrs created in forward" /
pytree "Node arity mismatch"). Those side effects are now skipped under ``torch.export`` (the
captured image output is unchanged), so the deterministic transforms export cleanly and
numerically match eager. This pins that so it does not regress.

Only *deterministic single* transforms are covered — random augmentations sample parameters
during capture in ways that don't line up with a separate eager call (RNG bookkeeping), and
``AugmentationSequential`` containers still stash their own per-call state and are not export-clean
yet.
"""

from __future__ import annotations

from typing import Callable, Tuple

import pytest
import torch

import kornia.augmentation as K


def _normalize() -> torch.nn.Module:
    return K.Normalize(mean=torch.zeros(3), std=torch.ones(3))


def _denormalize() -> torch.nn.Module:
    return K.Denormalize(mean=torch.zeros(3), std=torch.ones(3))


def _resize() -> torch.nn.Module:
    return K.Resize((24, 24))


def _center_crop() -> torch.nn.Module:
    return K.CenterCrop(24)


def _pad_to() -> torch.nn.Module:
    return K.PadTo((40, 40))


TORCH_EXPORT_DETERMINISTIC: list[Tuple[str, Callable[[], torch.nn.Module]]] = [
    ("Normalize", _normalize),
    ("Denormalize", _denormalize),
    ("Resize", _resize),
    ("CenterCrop", _center_crop),
    ("PadTo", _pad_to),
]


@pytest.mark.skipif(not hasattr(torch, "export"), reason="torch.export requires torch>=2.1")
@pytest.mark.parametrize("name,factory", TORCH_EXPORT_DETERMINISTIC, ids=[n for n, _ in TORCH_EXPORT_DETERMINISTIC])
def test_torch_export_deterministic(name: str, factory: Callable[[], torch.nn.Module]) -> None:
    """Deterministic inference transform captures via ``torch.export`` and matches eager."""
    torch.manual_seed(0)
    x = torch.randn(1, 3, 32, 32)
    aug = factory()
    aug.eval()
    eager = aug(x)
    exported = torch.export.export(aug, (x,))
    out = exported.module()(x)
    assert out.shape == eager.shape, f"{name}: shape {out.shape} vs {eager.shape}"
    torch.testing.assert_close(out, eager, atol=1e-5, rtol=1e-5)


def _seq_pointwise() -> torch.nn.Module:
    return K.AugmentationSequential(_normalize(), K.RandomBrightness((1.2, 1.2), p=1.0), data_keys=["input"])


def _seq_resize() -> torch.nn.Module:
    return K.AugmentationSequential(_normalize(), K.Resize((16, 16)), data_keys=["input"])


def _seq_crop_pad() -> torch.nn.Module:
    return K.AugmentationSequential(_normalize(), K.CenterCrop(20), K.PadTo((24, 24)), data_keys=["input"])


TORCH_EXPORT_CONTAINERS: list[Tuple[str, Callable[[], torch.nn.Module]]] = [
    ("Seq[Norm,Bright]", _seq_pointwise),
    ("Seq[Norm,Resize]", _seq_resize),
    ("Seq[Norm,CenterCrop,PadTo]", _seq_crop_pad),
]


@pytest.mark.skipif(not hasattr(torch, "export"), reason="torch.export requires torch>=2.1")
@pytest.mark.parametrize("name,factory", TORCH_EXPORT_CONTAINERS, ids=[n for n, _ in TORCH_EXPORT_CONTAINERS])
def test_torch_export_container(name: str, factory: Callable[[], torch.nn.Module]) -> None:
    """A deterministic ``AugmentationSequential`` pipeline captures via ``torch.export`` and matches eager."""
    torch.manual_seed(0)
    x = torch.randn(1, 3, 32, 32)
    seq = factory()
    eager = seq(x)
    exported = torch.export.export(seq, (x,))
    out = exported.module()(x)
    assert out.shape == eager.shape, f"{name}: shape {out.shape} vs {eager.shape}"
    torch.testing.assert_close(out, eager, atol=1e-5, rtol=1e-5)
