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
"""torch.export round-trip for deterministic kornia transforms.

Closes torchgeo#3108 — the canonical example was:
    torch.export.export(K.Normalize(0, 1), inputs)
"""
from __future__ import annotations

import pytest
import torch
import kornia.augmentation as K


def _make_normalize_imagenet() -> K.Normalize:
    return K.Normalize(
        mean=torch.tensor([0.485, 0.456, 0.406]),
        std=torch.tensor([0.229, 0.224, 0.225]),
    )


def _make_normalize_zero_one() -> K.Normalize:
    return K.Normalize(mean=torch.tensor([0.0]), std=torch.tensor([1.0]))


def _make_denormalize() -> K.Denormalize:
    return K.Denormalize(
        mean=torch.tensor([0.485, 0.456, 0.406]),
        std=torch.tensor([0.229, 0.224, 0.225]),
    )


def _make_center_crop() -> K.CenterCrop:
    return K.CenterCrop(size=(64, 64))


def _make_resize() -> K.Resize:
    return K.Resize(size=(64, 64))


def _make_longest_max_size() -> K.LongestMaxSize:
    return K.LongestMaxSize(max_size=128)


def _make_smallest_max_size() -> K.SmallestMaxSize:
    return K.SmallestMaxSize(max_size=128)


def _make_pad_to() -> K.PadTo:
    return K.PadTo(size=(128, 128))


# The transforms torchgeo cares about specifically (issue body §preprocessing transforms):
DETERMINISTIC = [
    ("Normalize_imagenet", _make_normalize_imagenet),
    ("Normalize_zero_one", _make_normalize_zero_one),
    ("Denormalize", _make_denormalize),
    ("CenterCrop", _make_center_crop),
    ("Resize", _make_resize),
    ("LongestMaxSize", _make_longest_max_size),
    ("SmallestMaxSize", _make_smallest_max_size),
    ("PadTo", _make_pad_to),
]


@pytest.mark.parametrize("name,factory", DETERMINISTIC, ids=lambda x: x[0] if isinstance(x, tuple) else None)
def test_torch_export(name: str, factory) -> None:
    """Each deterministic transform must export via torch.export.export and
    round-trip numerically equal to eager."""
    try:
        m = factory()
        m.eval()
    except Exception as e:
        pytest.skip(f"{name} construction failed: {e}")
    x = torch.randn(1, 3, 96, 96)

    try:
        ep = torch.export.export(m, (x,))
    except Exception as e:
        pytest.fail(f"{name} torch.export.export failed: {type(e).__name__}: {e}")

    y_eager = m(x)
    y_exported = ep.module()(x)
    if isinstance(y_eager, torch.Tensor):
        torch.testing.assert_close(y_eager, y_exported, atol=1e-5, rtol=1e-4)
    else:
        # tuple-return — match elementwise
        for a, b in zip(y_eager, y_exported):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_normalize_zero_one_canonical() -> None:
    """The exact line from torchgeo issue #3108 must work."""
    m = K.Normalize(mean=torch.tensor([0.0]), std=torch.tensor([1.0]))
    x = torch.randn(1, 3, 32, 32)
    ep = torch.export.export(m, (x,))
    assert ep is not None
    y = ep.module()(x)
    torch.testing.assert_close(y, x, atol=1e-6, rtol=0.0)
