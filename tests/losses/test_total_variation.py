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
from kornia.core.exceptions import BaseError

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

    @pytest.mark.parametrize("input_dtype", [torch.uint8, torch.int8, torch.int16, torch.int32])
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 2, 3)])
    @pytest.mark.parametrize("reduction", ["sum", "mean"])
    def test_tv_integer_extrema(self, device, input_dtype, shape, reduction):
        limits = torch.iinfo(input_dtype)
        pred = torch.tensor(
            [[limits.max, limits.min, 0], [limits.max, limits.min, 0]], device=device, dtype=input_dtype
        ).expand(shape)
        # Four horizontal edges: two span the full range, two have magnitude abs(min).
        expected_sum = 2 * (limits.max - limits.min + abs(limits.min))
        expected = torch.tensor(expected_sum, device=device, dtype=torch.int64)
        if reduction == "mean":
            expected = expected.float() / 4
        expected = expected.expand(shape[:-2])

        self.assert_close(kornia.losses.total_variation(pred, reduction), expected)
        self.assert_close(kornia.losses.total_variation(pred.flip(-1), reduction), expected)
        self.assert_close(kornia.losses.total_variation(pred.transpose(-2, -1), reduction), expected)
        if reduction == "sum":
            self.assert_close(kornia.losses.TotalVariation()(pred), expected)

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
        pred, _ = torch.meshgrid([torch.arange(5), torch.arange(5)], indexing="ij")
        pred = pred.to(device, dtype)
        actual = kornia.losses.total_variation(pred, reduction=reduction)
        self.assert_close(actual, expected.to(device, dtype), rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("layout", ["contiguous", "channels_last", "permuted"])
    @pytest.mark.parametrize("reduction", ["sum", "mean"])
    def test_tv_matches_the_two_dim_reduction(self, device, dtype, layout, reduction):
        # The reduction runs over flatten(-2) rather than dim=(-2, -1); the two must agree
        # for any memory layout the flatten has to copy, not only a contiguous input.
        pred = torch.rand(2, 3, 6, 7, device=device, dtype=dtype)
        if layout == "channels_last":
            pred = pred.contiguous(memory_format=torch.channels_last)
        elif layout == "permuted":
            pred = pred.permute(0, 1, 3, 2)

        dif1 = (pred[..., 1:, :] - pred[..., :-1, :]).abs()
        dif2 = (pred[..., :, 1:] - pred[..., :, :-1]).abs()
        if reduction == "sum":
            expected = dif1.sum(dim=(-2, -1)) + dif2.sum(dim=(-2, -1))
        else:
            expected = dif1.mean(dim=(-2, -1)) + dif2.mean(dim=(-2, -1))

        actual = kornia.losses.total_variation(pred, reduction=reduction)
        self.assert_close(actual, expected)

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

    @pytest.mark.parametrize("input_dtype", [torch.uint8, torch.int8, torch.int16, torch.int32])
    @pytest.mark.parametrize("reduction", ["sum", "mean"])
    def test_dynamo_integer(self, device, input_dtype, reduction, torch_optimizer):
        limits = torch.iinfo(input_dtype)
        image = torch.tensor([[limits.max, limits.min], [limits.max, limits.min]], device=device, dtype=input_dtype)
        op = kornia.losses.total_variation
        op_optimized = torch_optimizer(op)

        self.assert_close(op(image, reduction), op_optimized(image, reduction))

    def test_module(self, device, dtype):
        image = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        op = kornia.losses.total_variation
        op_module = kornia.losses.TotalVariation()

        self.assert_close(op(image), op_module(image))

    def test_gradcheck(self, device, dtype):
        dtype = torch.float64
        image = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        self.gradcheck(kornia.losses.total_variation, (image,))


class TestConventionsTotalVariation(BaseTester):
    @staticmethod
    def _edges(device, dtype):
        # H = 5, W = 8. Channel 0 holds one vertical edge (a step along W in each of the 5 rows), channel 1 one
        # horizontal edge (a step along H in each of the 8 columns), channel 2 is constant; sample 1 doubles sample 0.
        img = torch.zeros(2, 3, 5, 8)
        img[:, 0, :, 3:] = 1.0
        img[:, 1, 2:, :] = 1.0
        img[:, 2] = 0.5
        img[1] *= 2
        return img.to(device, dtype)

    def test_convention_total_variation_sums_over_the_last_two_axes(self, device, dtype):
        # Anisotropic L1 total variation, sum |d/dy| + sum |d/dx| over the last two axes only: (B, C, H, W) gives
        # (B, C), with the channels kept apart. The default reduction is 'sum', and TotalVariation always sums.
        img = self._edges(device, dtype)
        expected = torch.tensor([[5.0, 8.0, 0.0], [10.0, 16.0, 0.0]], device=device, dtype=dtype)
        self.assert_close(kornia.losses.total_variation(img), expected)
        self.assert_close(kornia.losses.TotalVariation()(img), expected)
        self.assert_close(kornia.losses.total_variation(img[0, 1]), expected[0, 1])

    def test_convention_total_variation_mean_divides_each_term_by_its_own_count(self, device, dtype):
        # 'mean' is mean |d/dy| over the (H - 1) W vertical differences plus mean |d/dx| over the H (W - 1) horizontal
        # ones: a vertical edge gives 1 / (W - 1) = 1/7 and a horizontal edge 1 / (H - 1) = 1/4, where dividing the sum
        # by H W would give 1/8 and 1/5. Only 'mean' and 'sum' are accepted.
        img = self._edges(device, dtype)
        expected = torch.tensor([[1 / 7, 1 / 4, 0.0], [2 / 7, 2 / 4, 0.0]], device=device, dtype=dtype)
        self.assert_close(kornia.losses.total_variation(img, reduction="mean"), expected)
        # Relabelling check: transposing the image transposes both counts with it, so the values stay.
        self.assert_close(kornia.losses.total_variation(img.transpose(-2, -1), reduction="mean"), expected)
        with pytest.raises((ValueError, BaseError)):
            kornia.losses.total_variation(img, reduction="none")
