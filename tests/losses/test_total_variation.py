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

import kornia
from kornia.utils import torch_meshgrid

from testing.base import BaseTester


class TestTotalVariation(BaseTester):
    # Total variation of constant vectors is 0
    @pytest.mark.parametrize(
        "pred, expected",
        [
            (torch.ones(3, 4, 5), torch.tensor([0.0, 0.0, 0.0])),
            (2 * torch.ones(2, 3, 4, 5), torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
        ],
    )
    def test_tv_on_constant(self, device, dtype, pred, expected):
        actual = kornia.losses.total_variation(pred.to(device, dtype))
        self.assert_close(actual, expected.to(device, dtype))

    # Total variation of constant vectors is 0
    @pytest.mark.parametrize(
        "pred, expected",
        [
            (torch.ones(3, 4, 5), torch.tensor([0.0, 0.0, 0.0])),
            (2 * torch.ones(2, 3, 4, 5), torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
        ],
    )
    def test_tv_on_constant_int(self, device, pred, expected):
        actual = kornia.losses.total_variation(pred.to(device, dtype=torch.int32), reduction="mean")
        self.assert_close(actual, expected.to(device))

    # Total variation for 3D tensors
    @pytest.mark.parametrize(
        "pred, expected",
        [
            (
                torch.tensor(
                    [
                        [
                            [0.11747694, 0.5717714, 0.89223915, 0.2929412, 0.63556224],
                            [0.5371079, 0.13416398, 0.7782737, 0.21392655, 0.1757018],
                            [0.62360305, 0.8563448, 0.25304103, 0.68539226, 0.6956515],
                            [0.9350611, 0.01694632, 0.78724295, 0.4760313, 0.73099905],
                        ],
                        [
                            [0.4788819, 0.45253807, 0.932798, 0.5721999, 0.7612051],
                            [0.5455887, 0.8836531, 0.79551977, 0.6677338, 0.74293613],
                            [0.4830376, 0.16420758, 0.15784949, 0.21445751, 0.34168917],
                            [0.8675162, 0.5468113, 0.6117004, 0.01305223, 0.17554593],
                        ],
                        [
                            [0.6423703, 0.5561105, 0.54304767, 0.20339686, 0.8553698],
                            [0.98024786, 0.31562763, 0.10122144, 0.17686582, 0.26260805],
                            [0.20522952, 0.14523649, 0.8601968, 0.02593213, 0.7382898],
                            [0.71935296, 0.9625162, 0.42287344, 0.07979459, 0.9149871],
                        ],
                    ]
                ),
                torch.tensor([12.6647, 7.9527, 12.3838]),
            ),
            (
                torch.tensor([[[0.09094203, 0.32630223, 0.8066123], [0.10921168, 0.09534764, 0.48588026]]]),
                torch.tensor([1.6900]),
            ),
        ],
    )
    def test_tv_on_3d(self, device, dtype, pred, expected):
        actual = kornia.losses.total_variation(pred.to(device, dtype))
        self.assert_close(actual, expected.to(device, dtype), rtol=1e-3, atol=1e-3)

    # Total variation for 4D tensors
    @pytest.mark.parametrize(
        "pred, expected",
        [
            (
                torch.tensor(
                    [
                        [
                            [[0.8756, 0.0920], [0.8034, 0.3107]],
                            [[0.3069, 0.2981], [0.9399, 0.7944]],
                            [[0.6269, 0.1494], [0.2493, 0.8490]],
                        ],
                        [
                            [[0.3256, 0.9923], [0.2856, 0.9104]],
                            [[0.4107, 0.4387], [0.2742, 0.0095]],
                            [[0.7064, 0.3674], [0.6139, 0.2487]],
                        ],
                    ]
                ),
                torch.tensor([[1.5672, 1.2836, 2.1544], [1.4134, 0.8584, 0.9154]]),
            ),
            (
                torch.tensor(
                    [
                        [[[0.1104, 0.2284, 0.4371], [0.4569, 0.1906, 0.8035]]],
                        [[[0.0552, 0.6831, 0.8310], [0.3589, 0.5044, 0.0802]]],
                        [[[0.5078, 0.5703, 0.9110], [0.4765, 0.8401, 0.2754]]],
                    ]
                ),
                torch.tensor([[1.9566], [2.5787], [2.2682]]),
            ),
        ],
    )
    def test_tv_on_4d(self, device, dtype, pred, expected):
        actual = kornia.losses.total_variation(pred.to(device, dtype))
        self.assert_close(actual, expected.to(device, dtype), rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("pred", [torch.rand(3, 5, 5), torch.rand(4, 3, 5, 5), torch.rand(4, 2, 3, 5, 5)])
    def test_tv_shapes(self, device, dtype, pred):
        pred = pred.to(device, dtype)
        actual_lesser_dims = []
        for slice in torch.unbind(pred, dim=0):
            slice_tv = kornia.losses.total_variation(slice)
            actual_lesser_dims.append(slice_tv)
        actual_lesser_dims = torch.stack(actual_lesser_dims, dim=0)
        actual_higher_dims = kornia.losses.total_variation(pred)
        self.assert_close(actual_lesser_dims, actual_higher_dims.to(device, dtype), rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("reduction, expected", [("sum", torch.tensor(20)), ("mean", torch.tensor(1))])
    def test_tv_reduction(self, device, dtype, reduction, expected):
        pred, _ = torch_meshgrid([torch.arange(5), torch.arange(5)], "ij")
        pred = pred.to(device, dtype)
        actual = kornia.losses.total_variation(pred, reduction=reduction)
        self.assert_close(actual, expected.to(device, dtype), rtol=1e-3, atol=1e-3)

    # Expect TypeError to be raised when non-torch tensors are passed
    @pytest.mark.parametrize("pred", [1, [1, 2]])
    def test_tv_on_invalid_types(self, device, dtype, pred):
        with pytest.raises(TypeError):
            kornia.losses.total_variation(pred)

    def test_dynamo(self, device, dtype, torch_optimizer):
        image = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        op = kornia.losses.total_variation
        op_optimized = torch_optimizer(op)

        self.assert_close(op(image), op_optimized(image))

    def test_module(self, device, dtype):
        image = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        op = kornia.losses.total_variation
        op_module = kornia.losses.TotalVariation()

        self.assert_close(op(image), op_module(image))

    def test_gradcheck(self, device):
        dtype = torch.float64
        image = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        self.gradcheck(kornia.losses.total_variation, (image,))
