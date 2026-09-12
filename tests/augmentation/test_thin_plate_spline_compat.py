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

from __future__ import annotations

import pytest
import torch

from kornia.augmentation import RandomThinPlateSpline
from kornia.geometry.transform import get_tps_transform, warp_image_tps


def test_corrected_grid_opt_in(device: torch.device, dtype: torch.dtype) -> None:
    src = torch.tensor(
        [[[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]],
        device=device,
        dtype=dtype,
    )
    params = {"src": src, "dst": src}
    image = torch.arange(16.0, device=device, dtype=dtype).view(1, 1, 4, 4)

    legacy = RandomThinPlateSpline(p=1.0)
    with pytest.warns(DeprecationWarning, match="use_correct_grid=True"):
        legacy_output = legacy.apply_transform(image, params, legacy.flags)
    assert not torch.allclose(legacy_output, image, atol=1e-2, rtol=1e-2)

    corrected = RandomThinPlateSpline(p=1.0, use_correct_grid=True)
    corrected_output = corrected.apply_transform(image, params, corrected.flags)
    torch.testing.assert_close(corrected_output, image, atol=1e-2, rtol=1e-2)


def test_compile_legacy_and_corrected_grids() -> None:
    src = torch.tensor([[[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]])
    image = torch.arange(16.0).view(1, 1, 4, 4)
    kernel, affine = get_tps_transform(src, src)

    def legacy(
        x: torch.Tensor, points: torch.Tensor, weights: torch.Tensor, affine_weights: torch.Tensor
    ) -> torch.Tensor:
        return warp_image_tps(x, points, weights, affine_weights)

    def corrected(
        x: torch.Tensor, points: torch.Tensor, weights: torch.Tensor, affine_weights: torch.Tensor
    ) -> torch.Tensor:
        return warp_image_tps(x, points, weights, affine_weights, use_correct_grid=True)

    with pytest.warns(DeprecationWarning, match="use_correct_grid=True"):
        expected_legacy = legacy(image, src, kernel, affine)

    compiled_legacy = torch.compile(legacy, backend="eager", fullgraph=True)(image, src, kernel, affine)
    compiled_corrected = torch.compile(corrected, backend="eager", fullgraph=True)(image, src, kernel, affine)

    torch.testing.assert_close(compiled_legacy, expected_legacy)
    torch.testing.assert_close(compiled_corrected, image, atol=1e-2, rtol=1e-2)
