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

    @pytest.mark.parametrize("tensor_arg", ["random_scale", "roughness"])
    def test_dtype_none_follows_a_tensor_argument_at_every_size(self, tensor_arg, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64")
        # With dtype=None each level promotes the map to the dtype of random_scale * roughness, and a map with a side
        # of 3 px or less, which is the seed alone, is cast to it too, without being scaled: it has the same dtype as
        # a larger one, and its values are the float32 draws.
        g = torch.Generator(device=device)
        draw = functools.partial(torch.rand, generator=g)
        value = torch.tensor(0.5, dtype=torch.float64)
        for size in ((1, 1, 2, 2), (1, 1, 3, 9), (2, 1, 9, 14)):
            g.manual_seed(0)
            out = kornia.contrib.diamond_square(size, random_fn=draw, device=device, **{tensor_arg: value})
            assert out.dtype == torch.float64
            if min(size[2:]) <= 3:
                g.manual_seed(0)
                ref = kornia.contrib.diamond_square(size, random_fn=draw, device=device, **{tensor_arg: 0.5})
                assert ref.dtype == torch.float32
                assert torch.equal(out, ref.double())

    @pytest.mark.parametrize(
        ("scale_f64", "roughness_f64", "expected"),
        [
            (False, False, [None, torch.float32, torch.float32]),
            (False, True, [None, torch.float32, torch.float64]),
            (True, False, [None, torch.float64, torch.float64]),
            (True, True, [None, torch.float64, torch.float64]),
        ],
        ids=["floats", "roughness-f64", "scale-f64", "both-f64"],
    )
    def test_dtype_none_draws_level_by_level_as_before(self, scale_f64, roughness_f64, expected, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64")
        # random_fn is asked for the seed (dtype=None), then for each level in the dtype of the map entering it. The
        # unscaled seed enters level 1 promoted with random_scale's dtype only, as the seed scaled by random_scale did,
        # so each level draws in the same dtype, and random_scale=1 gives the same map. Expected sequences generated
        # on dd3bddb7e, where the seed was scaled by random_scale, with this recording random_fn and
        #   kornia.contrib.diamond_square((2, 1, 9, 14), random_scale=s, roughness=r, random_fn=record)
        # for s = 1.0 or torch.tensor(1.0, dtype=torch.float64), r = 0.5 or torch.tensor(0.5, dtype=torch.float64).
        g = torch.Generator(device=device).manual_seed(0)
        asked = []

        def record(shape, device=None, dtype=None):
            asked.append(dtype)
            return torch.rand(shape, generator=g, device=device, dtype=dtype)

        scale = torch.tensor(1.0, dtype=torch.float64) if scale_f64 else 1.0
        roughness = torch.tensor(0.5, dtype=torch.float64) if roughness_f64 else 0.5
        out = kornia.contrib.diamond_square(
            (2, 1, 9, 14), roughness=roughness, random_scale=scale, random_fn=record, device=device
        )
        assert asked == expected
        assert out.dtype == expected[-1]

    @pytest.mark.parametrize(
        ("dtype", "drawn", "expected_dtype", "expected", "seed_only_dtype"),
        [
            (
                None,
                torch.float64,
                torch.float64,
                [
                    [0.9700530171394348, 0.6190204124439906, 0.5825837657368917, 0.7164888574434708],
                    [0.6135796297792718, 0.7338367893977338, 0.6909577292507927, 0.6741751155470027],
                ],
                torch.float32,
            ),
            (
                torch.float16,
                torch.float32,
                torch.float32,
                [
                    [0.496337890625, 0.6477437019348145, 0.7212197780609131, 0.6671783924102783],
                    [0.6196125149726868, 0.6749759912490845, 0.501835823059082, 0.5487739443778992],
                ],
                torch.float16,
            ),
        ],
        ids=["none-draws-f64", "f16-draws-f32"],
    )
    def test_random_fn_ignoring_dtype_gives_the_previous_map(
        self, dtype, drawn, expected_dtype, expected, seed_only_dtype, device
    ):
        if device.type != "cpu":
            pytest.skip("the literals are CPU values, and MPS has no float64")
        if dtype is not None and not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        # A random_fn that returns another dtype than it is asked for promotes the levels' arithmetic to its dtype,
        # so at random_scale=1 a map past its seed runs the operations of dd3bddb7e, where the seed was scaled by
        # random_scale, dtype included. Literals generated there on arm64 macOS, torch 2.14, with
        #   g = torch.Generator().manual_seed(0)
        #   draw = lambda shape, device=None, dtype=None: torch.rand(shape, generator=g, dtype=drawn)
        #   out = kornia.contrib.diamond_square((1, 1, 9, 17), random_fn=draw, dtype=dtype)
        #   out.dtype, out[0, 0, :2, :4]
        # They are not a cross-CPU bit pattern: torch 2.14 on the AMD EPYC CI runners rounds the float32 levels'
        # convolution at (1, 2) one ulp away, as a convolution accumulated in float64 does. A few ulps of the output
        # dtype still reject levels computed in the requested float16 (~1e-4 off) or a float64 map computed in float32
        # (~1e-8 off).
        # A map that is the seed alone stores the draws in the requested dtype (dd3bddb7e: same dtype at (1, 1, 3, 9)).
        g = torch.Generator().manual_seed(0)

        def draw(shape, device=None, dtype=None):
            return torch.rand(shape, generator=g, device=device, dtype=drawn)

        out = kornia.contrib.diamond_square((1, 1, 9, 17), random_fn=draw, device=device, dtype=dtype)
        assert out.dtype == expected_dtype
        ulps = 4 * torch.finfo(expected_dtype).eps
        self.assert_close(
            out[0, 0, :2, :4], torch.tensor(expected, device=device, dtype=expected_dtype), rtol=0, atol=ulps
        )
        seed_only = kornia.contrib.diamond_square((1, 1, 3, 9), random_fn=draw, device=device, dtype=dtype)
        assert seed_only.dtype == seed_only_dtype

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
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")

        # One level from a 3x5 seed: a new value is (1 - s) * mean(neighbours) + s * U, s = random_scale * roughness
        # = 0.25, U = 1, seed corner draws [[1, 2, 4], [8, 16, 32]], not scaled by random_scale. Worked by hand (seed:
        # corner 1, centre (1 + 2 + 8 + 16) / 4 = 6.75, top midpoint 3.25, left midpoint 5.25, row-1 point between
        # the centres 9.5625 as in test_seed_values; diamond (1, 3) = 0.75 * mean(3.25, 2, 6.75, 9.5625) + 0.25 =
        # 4.29296875):
        #   diamond (1, 1) = 0.75 * mean(1, 3.25, 5.25, 6.75) + 0.25 = 3.296875
        #   border square (0, 1) = 0.75 * mean(1, 3.25, 3.296875) + 0.25 = 2.13671875 (three neighbours)
        #   interior square (1, 2) = 0.75 * mean(3.25, 6.75, 3.296875, 4.29296875) + 0.25 = 3.548095703125
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
        expected = torch.tensor([1.0, 6.75, 3.296875, 2.13671875, 3.548095703125], device=device, dtype=dtype)
        self.assert_close(nodes, expected)

    def test_convention_normalize_range_is_min_max_per_plane(self, device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        torch.manual_seed(0)
        out = kornia.contrib.diamond_square((3, 2, 9, 14), normalize_range=(-1.0, 1.0), device=device, dtype=dtype)
        self.assert_close(out.amin((-2, -1)), torch.full((3, 2), -1.0, device=device, dtype=dtype))
        self.assert_close(out.amax((-2, -1)), torch.full((3, 2), 1.0, device=device, dtype=dtype))

    def test_convention_random_fn_is_the_only_source_of_randomness(self, device):
        def device_state():
            if device.type == "cuda":
                return torch.cuda.get_rng_state(device)
            if device.type == "mps":
                return torch.mps.get_rng_state()
            return None

        g = torch.Generator(device=device)
        draw = functools.partial(torch.rand, generator=g)
        cpu_state, accelerator_state = torch.get_rng_state(), device_state()
        g.manual_seed(3)
        first = kornia.contrib.diamond_square((2, 1, 9, 14), random_fn=draw, device=device)
        g.manual_seed(3)
        second = kornia.contrib.diamond_square((2, 1, 9, 14), random_fn=draw, device=device)
        assert torch.equal(first, second)
        assert torch.equal(cpu_state, torch.get_rng_state())
        if accelerator_state is not None:
            assert torch.equal(accelerator_state, device_state())

    @pytest.mark.parametrize("random_scale", [0.25, 0.5, 1.0, 2.0])
    def test_convention_mean_field_is_flat_for_every_random_scale(self, random_scale, device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")

        # The map is linear in the draws, so a constant 0.5 draw gives the expected field of torch.rand exactly. The
        # seed is the draws themselves (0.5) and each level computes (1 - s) * 0.5 + s * 0.5 = 0.5 for any s, so every
        # pixel is 0.5. A seed scaled by random_scale gave, at random_scale=0.5, 0.25 at the seed nodes, 0.3125 at
        # the level-1 diamonds and 0.328125 / 0.3359375 at the border / interior level-1 squares.
        def half(shape, device=None, dtype=None):
            return torch.full(shape, 0.5, device=device, dtype=dtype)

        out = kornia.contrib.diamond_square(
            (1, 1, 9, 17), random_scale=random_scale, random_fn=half, device=device, dtype=dtype
        )
        self.assert_close(out, torch.full_like(out, 0.5))

    def test_random_scale_one_is_unchanged(self, device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")

        # random_scale=1, the default and the value the plasma augmentations use, gives the same map whether or not
        # the seed is scaled by it. Draws come from a seeded CPU generator in float64 and are cast afterwards, so every
        # device and dtype sees the same numbers. Literal generated before the seed stopped being scaled:
        #   g = torch.Generator().manual_seed(0)
        #   draw = lambda shape, device=None, dtype=None: torch.rand(shape, generator=g, dtype=torch.float64).to(
        #       device=device, dtype=dtype)
        #   out = kornia.contrib.diamond_square((1, 1, 5, 9), random_scale=1.0, random_fn=draw, dtype=torch.float64)
        #   out[0, 0, :3, :5]
        g = torch.Generator().manual_seed(0)

        def draw(shape, device=None, dtype=None):
            return torch.rand(shape, generator=g, dtype=torch.float64).to(device=device, dtype=dtype)

        out = kornia.contrib.diamond_square((1, 1, 5, 9), random_scale=1.0, random_fn=draw, device=device, dtype=dtype)
        expected = torch.tensor(
            [
                [0.9700530018065531, 0.5825837538768377, 0.8295946779397027, 0.5193563724418668, 0.707819864399788],
                [0.6392595145199516, 0.6425319703329853, 0.47735102570438287, 0.7140160159032599, 0.7129631915412449],
                [0.9005706178470932, 0.6569745561601958, 0.8109111676127666, 0.7014195097526202, 0.7036497142732241],
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(out[0, 0, :3, :5], expected)
