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

import pytest
import torch
import torch.nn.functional as F

from kornia.morphology import (
    dilation,
    dilation1d,
    erosion,
    erosion1d,
    separable_dilation2d,
    separable_erosion2d,
)


def test_dilation1d_numerical_equivalence():
    torch.manual_seed(42)
    B, C, L = 2, 3, 50
    x = torch.randn(B, C, L, dtype=torch.float32)

    for k in [3, 5, 7, 11]:
        pad = k // 2
        out_ours = dilation1d(x, kernel_size=k, padding=pad)
        out_ref = F.max_pool1d(x, kernel_size=k, stride=1, padding=pad)
        assert torch.allclose(out_ours, out_ref, atol=1e-5), f"Failed for kernel_size={k}"


def test_erosion1d_numerical_equivalence():
    torch.manual_seed(42)
    B, C, L = 2, 3, 50
    x = torch.randn(B, C, L, dtype=torch.float32)

    for k in [3, 5, 7, 11]:
        pad = k // 2
        out_ours = erosion1d(x, kernel_size=k, padding=pad)
        out_ref = -F.max_pool1d(-x, kernel_size=k, stride=1, padding=pad)
        assert torch.allclose(out_ours, out_ref, atol=1e-5), f"Failed for kernel_size={k}"


def test_separable_dilation2d_numerical_equivalence():
    torch.manual_seed(42)
    B, C, H, W = 2, 2, 32, 32
    x = torch.randn(B, C, H, W, dtype=torch.float32)

    for k in [3, 5, 7]:
        pad = k // 2
        out_ours = separable_dilation2d(x, kernel_size=k, padding=pad)
        out_ref = F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)
        assert torch.allclose(out_ours, out_ref, atol=1e-5), f"2D Dilation mismatch at k={k}"


def test_separable_erosion2d_numerical_equivalence():
    torch.manual_seed(42)
    B, C, H, W = 2, 2, 32, 32
    x = torch.randn(B, C, H, W, dtype=torch.float32)

    for k in [3, 5, 7]:
        pad = k // 2
        out_ours = separable_erosion2d(x, kernel_size=k, padding=pad)
        out_ref = -F.max_pool2d(-x, kernel_size=k, stride=1, padding=pad)
        assert torch.allclose(out_ours, out_ref, atol=1e-5), f"2D Erosion mismatch at k={k}"


def test_dilation_monotonic_engine():
    torch.manual_seed(42)
    B, C, H, W = 2, 2, 32, 32
    x = torch.randn(B, C, H, W, dtype=torch.float32)
    kernel = torch.ones(5, 5, dtype=torch.float32)

    out_mono = dilation(x, kernel, engine="monotonic")
    out_unfold = dilation(x, kernel, engine="unfold")
    assert torch.allclose(out_mono, out_unfold, atol=1e-5)


def test_erosion_monotonic_engine():
    torch.manual_seed(42)
    B, C, H, W = 2, 2, 32, 32
    x = torch.randn(B, C, H, W, dtype=torch.float32)
    kernel = torch.ones(5, 5, dtype=torch.float32)

    out_mono = erosion(x, kernel, engine="monotonic")
    out_unfold = erosion(x, kernel, engine="unfold")
    assert torch.allclose(out_mono, out_unfold, atol=1e-5)


def test_gradcheck_dilation1d():
    torch.manual_seed(42)
    x = torch.randn(2, 2, 16, dtype=torch.float64, requires_grad=True)
    x = x + torch.linspace(0, 1, 16, dtype=torch.float64).reshape(1, 1, -1)

    assert torch.autograd.gradcheck(
        dilation1d,
        (x, 3, 1, 1),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        raise_exception=True,
    )
