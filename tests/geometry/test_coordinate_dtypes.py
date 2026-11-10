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

from functools import partial

import pytest
import torch

from kornia.geometry.boxes import Boxes, Boxes3D, VideoBoxes
from kornia.geometry.keypoints import Keypoints, Keypoints3D

from testing.base import BaseTester

_BOX = [[[2049, 2051], [2051, 2051], [2051, 2053], [2049, 2053]]]
_CUBE = [
    [
        [2049, 2051, 257],
        [2051, 2051, 257],
        [2051, 2053, 257],
        [2049, 2053, 257],
        [2049, 2051, 259],
        [2051, 2051, 259],
        [2051, 2053, 259],
        [2049, 2053, 259],
    ]
]


@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
class TestCoordinateDtypes(BaseTester):
    @pytest.mark.parametrize(
        "constructor, coordinates",
        [
            pytest.param(partial(Keypoints, raise_if_not_floating_point=False), [[2049, 2051]], id="keypoints"),
            pytest.param(
                partial(Keypoints3D, raise_if_not_floating_point=False), [[2049, 2051, 257]], id="keypoints3d"
            ),
            pytest.param(partial(Boxes, raise_if_not_floating_point=False), _BOX, id="boxes"),
            pytest.param(partial(Boxes.from_tensor, mode="vertices_plus"), _BOX, id="boxes-from-tensor"),
            pytest.param(partial(Boxes3D, raise_if_not_floating_point=False), _CUBE, id="boxes3d"),
            pytest.param(
                partial(Boxes3D.from_tensor, mode="xyzxyz_plus"),
                [[2049, 2051, 257, 2051, 2053, 259]],
                id="boxes3d-from-tensor",
            ),
            pytest.param(partial(VideoBoxes, raise_if_not_floating_point=False), [_BOX], id="video-boxes"),
            pytest.param(VideoBoxes.from_tensor, [[_BOX]], id="video-boxes-from-tensor"),
        ],
    )
    @pytest.mark.parametrize("input_dtype", [torch.int64, torch.float16, torch.bfloat16, torch.float32, torch.float64])
    def test_coordinate_dtype(self, device, default_dtype, input_dtype, constructor, coordinates):
        expected_dtype = (
            (torch.float64 if default_dtype == torch.float64 else torch.float32)
            if input_dtype == torch.int64
            else input_dtype
        )
        if device.type == "mps" and torch.float64 in (input_dtype, expected_dtype):
            pytest.skip("MPS does not support float64")
        source = torch.tensor(coordinates, device=device, dtype=input_dtype)
        original_default = torch.get_default_dtype()
        try:
            torch.set_default_dtype(default_dtype)
            result = constructor(source)
            assert result.data.dtype == expected_dtype
            assert result.data.device == device
            # Check exact coordinates, including values not representable in half precision.
            if isinstance(result, Boxes3D) and source.shape[-1] == 6:
                actual = result.to_tensor(mode="xyzxyz_plus")
            else:
                actual = result.data.reshape(source.shape)
            self.assert_close(actual, source.to(expected_dtype), atol=0, rtol=0)
        finally:
            torch.set_default_dtype(original_default)

    @pytest.mark.parametrize("box_class, dimensions", [(Boxes, 2), (Boxes3D, 3)])
    def test_integer_transform_dtype(self, device, default_dtype, box_class, dimensions):
        expected_dtype = torch.float64 if default_dtype == torch.float64 else torch.float32
        if device.type == "mps" and expected_dtype == torch.float64:
            pytest.skip("MPS does not support float64")
        vertices = torch.zeros((1, 2**dimensions, dimensions), device=device, dtype=expected_dtype)
        transform = torch.eye(dimensions + 1, device=device, dtype=torch.int64)
        translation = torch.tensor([2049, 2051, 257][:dimensions], device=device, dtype=torch.int64)
        transform[:-1, -1] = translation
        original_default = torch.get_default_dtype()
        try:
            torch.set_default_dtype(default_dtype)
            result = box_class(vertices).transform_boxes(transform)
            assert result.data.dtype == expected_dtype
            self.assert_close(result.data, vertices + translation.to(expected_dtype), atol=0, rtol=0)
        finally:
            torch.set_default_dtype(original_default)
