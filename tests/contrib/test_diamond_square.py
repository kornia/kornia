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

import functools

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

    @pytest.mark.parametrize("size", [(17, 5), (10, 14), (5, 17)])
    def test_mean_field_is_flat(self, size, device):
        # Every seed node is U(0, 1) and each step mixes (1 - s) * neighbour mean + s * U(0, 1), so the expected field
        # is 0.5 everywhere. A biased seed and border step gave a minimum mean of 0.165 (17x5, 10x14, 5x17, 4000 draws).
        # Sizes with H != W; (5, 17) and (17, 5) have a 3x9 seed, which exercises the interior row-1 square centres.
        torch.manual_seed(0)
        out = kornia.contrib.diamond_square((4000, 1, *size), device=device, dtype=torch.float32)
        mean = out.mean(0)[0]
        self.assert_close(mean, torch.full_like(mean, 0.5), rtol=0.0, atol=0.03)  # standard error ~0.004

    def test_constant_random_fn_gives_a_constant_field(self, device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")

        # Interpolating a constant must return the constant (a biased seed and border step gave corners 1/3 and
        # border midpoints 4/3).
        def ones(shape, device=None, dtype=None):
            return torch.ones(shape, device=device, dtype=dtype)

        out = kornia.contrib.diamond_square((2, 1, 9, 14), random_fn=ones, device=device, dtype=dtype)
        self.assert_close(out, torch.ones_like(out))

    def test_signed_random_fn_fills_every_pixel(self, device):
        # Centres chosen by value > 0 left 56 % of the pixels at exactly 0 for torch.randn.
        torch.manual_seed(0)
        out = kornia.contrib.diamond_square((200, 1, 17, 33), random_fn=torch.randn, device=device)
        assert (out == 0).float().mean().item() < 1e-3

    def test_tensor_roughness_follows_the_requested_dtype_and_device(self, device):
        # A float64 CPU roughness returned float64 for dtype=float32 and crashed for device='mps'.
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
        # An integer range (0, 1) raised TypeError in normalize_min_max.
        out = kornia.contrib.diamond_square((1, 1, 9, 14), normalize_range=(0, 1), device=device, dtype=dtype)
        assert out.min() >= 0
        assert out.max() <= 1

    def test_seed_values(self, device, dtype):
        # A seed that divided its corners and centre by 3, used the opposite corner for the right-edge midpoint and
        # never interpolated the interior row-1 square centres of K >= 5 gave 0.333 / 92.58 / 459.25 for the 3x3
        # corners, centre and right midpoint. At recursion depth 0 the output is the 3xK seed itself. Expected
        # values are worked by hand: a diamond centre is the mean of its four corners, a border midpoint the mean of
        # its three neighbours, an interior row-1 point the mean of its four neighbours.
        def draw(values):
            def fn(shape, device=None, dtype=None):
                return torch.tensor(values, device=device, dtype=dtype).expand(shape).clone()

            return fn

        expected3 = torch.tensor(
            [[1.0, 96.25, 10.0], [126.25, 277.75, 429.25], [100.0, 459.25, 1000.0]], device=device, dtype=dtype
        )
        values3 = [[1.0, 10.0], [100.0, 1000.0]]
        # a 3x3 seed takes the transposing path, so it comes back transposed
        out3 = kornia.contrib.diamond_square((1, 1, 3, 3), random_fn=draw(values3), device=device, dtype=dtype)
        self.assert_close(out3[0, 0], expected3.transpose(0, 1))

        expected5 = torch.tensor(
            [
                [1.0, 3.25, 2.0, 6.5, 4.0],
                [5.25, 6.75, 9.5625, 13.5, 16.5],
                [8.0, 10.25, 16.0, 20.5, 32.0],
            ],
            device=device,
            dtype=dtype,
        )
        values5 = [[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]]
        out5 = kornia.contrib.diamond_square((1, 1, 3, 5), random_fn=draw(values5), device=device, dtype=dtype)
        self.assert_close(out5[0, 0], expected5)
        # the transposed path: a 5x3 output is the 3x5 seed transposed
        out5t = kornia.contrib.diamond_square((1, 1, 5, 3), random_fn=draw(values5), device=device, dtype=dtype)
        self.assert_close(out5t[0, 0], expected5.transpose(0, 1))

    def test_convention_update_rule(self, device, dtype):
        # One level from a 3x5 seed: a new value is (1 - s) * mean(neighbours) + s * U, s = random_scale * roughness
        # = 0.25, U = 1, seed corner draws [[1, 2, 4], [8, 16, 32]] scaled by random_scale. Worked by hand (scaled
        # seed: corner 0.5, centre 3.375, top midpoint 1.625, left midpoint 2.625; diamond (1, 3) = 2.271484375):
        #   diamond (1, 1) = 0.75 * mean(0.5, 1.625, 2.625, 3.375) + 0.25 = 1.7734375
        #   border square (0, 1) = 0.75 * mean(0.5, 1.625, 1.7734375) + 0.25 = 1.224609375 (three neighbours)
        #   interior square (1, 2) = 0.75 * mean(1.625, 3.375, 1.7734375, 2.271484375) + 0.25 = 1.9459228515625
        def draw(shape, device=None, dtype=None):
            if shape[-2] == 2:  # the seed's corner draws
                return torch.tensor([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]], device=device, dtype=dtype).expand(shape)
            return torch.ones(shape, device=device, dtype=dtype)

        out = kornia.contrib.diamond_square(
            (1, 1, 5, 9), roughness=0.5, random_scale=0.5, random_fn=draw, device=device, dtype=dtype
        )[0, 0]
        assert out.shape == (5, 9)
        # seed corner, seed centre, level-1 diamond centre, border square centre, interior square centre
        nodes = torch.stack([out[0, 0], out[2, 2], out[1, 1], out[0, 1], out[1, 2]])
        expected = torch.tensor([0.5, 3.375, 1.7734375, 1.224609375, 1.9459228515625], device=device, dtype=dtype)
        self.assert_close(nodes, expected)

    def test_convention_normalize_range_is_min_max_per_plane(self, device, dtype):
        torch.manual_seed(0)
        out = kornia.contrib.diamond_square((3, 2, 9, 14), normalize_range=(-1.0, 1.0), device=device, dtype=dtype)
        self.assert_close(out.amin((-2, -1)), torch.full((3, 2), -1.0, device=device, dtype=dtype))
        self.assert_close(out.amax((-2, -1)), torch.full((3, 2), 1.0, device=device, dtype=dtype))

    def test_convention_random_fn_is_the_only_source_of_randomness(self, device):
        g = torch.Generator(device=device)
        draw = functools.partial(torch.rand, generator=g)
        state = torch.get_rng_state()
        g.manual_seed(3)
        first = kornia.contrib.diamond_square((2, 1, 9, 14), random_fn=draw, device=device)
        g.manual_seed(3)
        second = kornia.contrib.diamond_square((2, 1, 9, 14), random_fn=draw, device=device)
        assert torch.equal(first, second)
        assert torch.equal(state, torch.get_rng_state())

    def test_wart_mean_field_drifts_for_random_scale_not_one_5963(self, device, dtype):
        # The map is linear in the draws, so a constant 0.5 draw gives the expected field of torch.rand exactly.
        # With random_scale=0.5 the seed is 0.5 * 0.5 = 0.25, and each level mixes towards 0.5: a level-1 diamond is
        # 0.75 * 0.25 + 0.25 * 0.5 = 0.3125, a level-1 square 0.328125 at the border and 0.3359375 inside.
        def half(shape, device=None, dtype=None):
            return torch.full(shape, 0.5, device=device, dtype=dtype)

        out = kornia.contrib.diamond_square(
            (1, 1, 9, 17), random_scale=0.5, random_fn=half, device=device, dtype=dtype
        )[0, 0]
        nodes = torch.stack([out[0, 0], out[4, 8], out[2, 2], out[0, 2], out[2, 4]])
        expected = torch.tensor([0.25, 0.25, 0.3125, 0.328125, 0.3359375], device=device, dtype=dtype)
        self.assert_close(nodes, expected)
