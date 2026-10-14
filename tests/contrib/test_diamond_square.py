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

from testing.base import BaseTester, supports_reflect_padding


class TestDiamondSquare(BaseTester):
    def test_smoke(self, device, dtype):
        torch.manual_seed(0)
        output_size = (1, 1, 3, 4)
        roughness = 0.5
        random_scale = 1.0
        out = kornia.contrib.diamond_square(output_size, roughness, random_scale, device=device, dtype=dtype)
        assert out.shape == output_size
        assert out.device == device
        assert out.dtype == dtype

    def test_normalize(self, device, dtype):
        torch.manual_seed(0)
        output_size = (1, 1, 3, 4)
        roughness = 0.5
        random_scale = 1.0
        normalize_range = (0.0, 1.0)
        expected_min = torch.tensor(normalize_range[0], device=device, dtype=dtype)
        expected_max = torch.tensor(normalize_range[1], device=device, dtype=dtype)
        out = kornia.contrib.diamond_square(
            output_size, roughness, random_scale, normalize_range=normalize_range, device=device, dtype=dtype
        )
        self.assert_close(out.min(), expected_min)
        self.assert_close(out.max(), expected_max)

    @pytest.mark.parametrize("spatial_size", [(1, 1), (2, 2), (2, 3), (3, 2), (1, 5), (2, 17)])
    def test_spatial_sizes_below_three(self, spatial_size, device, dtype):
        # Sides under 3 px used to fail with `math domain error` or a float in the `rand` size
        # (kornia#4472).
        torch.manual_seed(0)
        output_size = (2, 3, *spatial_size)
        out = kornia.contrib.diamond_square(output_size, device=device, dtype=dtype)
        assert out.shape == output_size
        assert torch.isfinite(out).all()

    def test_plasma_augmentation_on_a_two_pixel_image(self, device, dtype):
        img = torch.rand(1, 1, 2, 2, device=device, dtype=dtype)
        out = kornia.augmentation.RandomPlasmaBrightness(p=1.0)(img)
        assert out.shape == img.shape

    @pytest.mark.parametrize("size", [(9, 9), (5, 17)])
    def test_mean_field_is_flat(self, size, device):
        # A-35/A-36: every seed node is U(0, 1) and each step mixes (1 - s) * neighbour mean + s * U(0, 1), so the
        # expected field is 0.5 everywhere. main gave 0.166 at the corners and the centre and 0.496 at edge
        # midpoints (9x9, 4000 draws). (5, 17) has a 3x9 seed, which exercises the interior row-1 square centres.
        torch.manual_seed(0)
        out = kornia.contrib.diamond_square((4000, 1, *size), device=device, dtype=torch.float32)
        mean = out.mean(0)[0]
        self.assert_close(mean, torch.full_like(mean, 0.5), rtol=0.0, atol=0.03)  # standard error ~0.004

    def test_constant_random_fn_gives_a_constant_field(self, device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")

        # A-35/A-36: interpolating a constant must return the constant (main: corners 1/3, border midpoints 4/3).
        def ones(shape, device=None, dtype=None):
            return torch.ones(shape, device=device, dtype=dtype)

        out = kornia.contrib.diamond_square((2, 1, 9, 14), random_fn=ones, device=device, dtype=dtype)
        self.assert_close(out, torch.ones_like(out))

    def test_signed_random_fn_fills_every_pixel(self, device):
        # A-37: centres were chosen by value > 0, so torch.randn left 56 % of the pixels at exactly 0.
        torch.manual_seed(0)
        out = kornia.contrib.diamond_square((200, 1, 17, 17), random_fn=torch.randn, device=device)
        assert (out == 0).float().mean().item() < 1e-3

    def test_tensor_roughness_follows_the_requested_dtype_and_device(self, device):
        # A-38: a float64 CPU roughness returned float64 for dtype=float32 and crashed for device='mps'.
        r = torch.tensor([0.5, 0.6], dtype=torch.float64)
        s = torch.tensor(1.0, dtype=torch.float64)
        out = kornia.contrib.diamond_square(
            (2, 1, 9, 14), roughness=r, random_scale=s, device=device, dtype=torch.float32
        )
        assert out.dtype == torch.float32
        assert out.device.type == device.type

    def test_integer_normalize_range(self, device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        # A-39: (0, 1) raised TypeError in normalize_min_max.
        out = kornia.contrib.diamond_square((1, 1, 9, 14), normalize_range=(0, 1), device=device, dtype=dtype)
        assert out.min() >= 0
        assert out.max() <= 1

    def test_seed_values(self, device, dtype):
        # A-35: the right-edge midpoint used the opposite corner, and the interior row-1 square centres of a seed
        # with K >= 5 were never interpolated. Expected values are worked by hand: a diamond centre is the mean of
        # its four corners, a border midpoint the mean of its three neighbours, an interior row-1 point the mean of
        # its four neighbours. main gave 0.333 / 92.58 / 459.25 for the K=3 corners, centre and right midpoint.
        from kornia.contrib.diamond_square import _diamond_square_seed

        def draw(values):
            def fn(shape, device=None, dtype=None):
                return torch.tensor(values, device=device, dtype=dtype).expand(shape).clone()

            return fn

        out3 = _diamond_square_seed(1, 3, 3, draw([[1.0, 10.0], [100.0, 1000.0]]), device, dtype)
        expected3 = torch.tensor(
            [[[[1.0, 96.25, 10.0], [126.25, 277.75, 429.25], [100.0, 459.25, 1000.0]]]], device=device, dtype=dtype
        )
        # a 3x3 seed takes the transposing path, so it comes back transposed
        self.assert_close(out3, expected3.transpose(-2, -1))

        out5 = _diamond_square_seed(1, 3, 5, draw([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]]), device, dtype)
        expected5 = torch.tensor(
            [
                [
                    [
                        [1.0, 3.25, 2.0, 6.5, 4.0],
                        [5.25, 6.75, 9.5625, 13.5, 16.5],
                        [8.0, 10.25, 16.0, 20.5, 32.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(out5, expected5)
