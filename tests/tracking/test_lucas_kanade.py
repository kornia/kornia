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

from kornia.tracking import track_points_lk

from testing.base import BaseTester


def _texture(device, dtype, shift=(0.0, 0.0), height=41, width=45):
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    x, y = x - shift[0], y - shift[1]
    return (
        0.5
        + 0.18 * (0.42 * x + 0.12 * y).sin()
        + 0.15 * (0.16 * x - 0.48 * y).cos()
        + 0.08 * (0.29 * x + 0.36 * y).sin()
    )[None, None]


@pytest.fixture
def lk_dtype(dtype):
    if dtype not in (torch.float32, torch.float64):
        pytest.skip("Sparse LK deliberately supports float32 and float64; rejection is tested separately.")
    return dtype


class TestTrackPointsLK(BaseTester):
    def test_identity(self, device, lk_dtype):
        image = _texture(device, lk_dtype)
        points = image.new_tensor([[[15.2, 16.4], [25.3, 24.2]]])
        result, valid, error = track_points_lk(image, image, points, window_size=9)
        self.assert_close(result, points)
        assert valid.all()
        self.assert_close(error, torch.zeros_like(error))
        assert valid.dtype == torch.bool

    @pytest.mark.parametrize("shift", [(0.25, -0.4), (1.2, 0.7)])
    def test_translation(self, device, lk_dtype, shift):
        prev = _texture(device, lk_dtype)
        nxt = _texture(device, lk_dtype, shift)
        points = prev.new_tensor([[[15.2, 16.4], [25.3, 24.2]]])
        result, valid, error = track_points_lk(prev, nxt, points, window_size=9)
        assert valid.all()
        self.assert_close(result, points + points.new_tensor(shift), atol=0.03, rtol=0)
        # Independent NumPy bilinear sampling + central differences + np.linalg.solve,
        # iterating H delta = mean(gradient * (reference - target)) to norm(delta) <= 1e-3.
        expected = (
            [[15.437749174134025, 15.994120521302845], [25.546526617384533, 23.789613827350657]]
            if shift[0] == 0.25
            else [[16.391343894154595, 17.09794379998892], [26.496160632673345, 24.888968462275130]]
        )
        self.assert_close(result[0], points.new_tensor(expected), atol=1e-5, rtol=0)
        expected_error = (
            [6.2579779010709655e-6, 1.1924735706667254e-7]
            if shift[0] == 0.25
            else [2.7920649218770828e-6, 7.679103210196524e-7]
        )
        self.assert_close(error[0], points.new_tensor(expected_error), atol=1e-9, rtol=0.001)

    def test_batch_and_initial_estimate(self, device, lk_dtype):
        image = _texture(device, lk_dtype).expand(2, -1, -1, -1)
        nxt = torch.cat([_texture(device, lk_dtype, (6.0, -4.0)), _texture(device, lk_dtype, (-1.0, 2.0))])
        points = image.new_tensor([[[15.2, 16.4], [25.3, 24.2]]]).expand(2, -1, -1)
        shift = image.new_tensor([[6.0, -4.0], [-1.0, 2.0]])[:, None]
        initial = points + shift + 0.1
        result, valid, error = track_points_lk(image, nxt, points, initial, window_size=9)
        assert valid.all()
        self.assert_close(result, points + shift, atol=0.01, rtol=0)
        for batch in range(2):
            single = track_points_lk(
                image[batch : batch + 1],
                nxt[batch : batch + 1],
                points[batch : batch + 1],
                initial[batch : batch + 1],
                window_size=9,
            )
            for actual, expected in zip(
                (result[batch : batch + 1], valid[batch : batch + 1], error[batch : batch + 1]), single
            ):
                self.assert_close(actual, expected)

    def test_singular_and_bad_points(self, device, lk_dtype):
        image = torch.cat(
            [
                _texture(device, lk_dtype),
                torch.ones(1, 1, 41, 45, device=device, dtype=lk_dtype),
                torch.arange(45, device=device, dtype=lk_dtype)[None, None, None].expand(1, 1, 41, -1),
            ]
        )
        points = image.new_tensor([[[15.2, 16.4], [0.0, 0.0], [float("nan"), 20.0]]]).expand(3, -1, -1)
        result, valid, error = track_points_lk(image, image, points, window_size=9, min_eigenvalue=0)
        assert valid.tolist() == [[True, False, False], [False, False, False], [False, False, False]]
        assert torch.isfinite(result).all()
        assert torch.isinf(error[~valid]).all()
        self.assert_close(result[:, 2], torch.zeros_like(result[:, 2]))

    def test_nonfinite_image_isolation_and_gradients(self, device, lk_dtype):
        image = torch.cat(
            [
                _texture(device, lk_dtype),
                _texture(device, lk_dtype),
                torch.zeros(1, 1, 41, 45, device=device, dtype=lk_dtype),
            ]
        )
        image[1, 0, 0, 0] = float("nan")
        image.requires_grad_()
        points = image.new_tensor([[[15.2, 16.4]]]).expand(3, -1, -1).clone().requires_grad_()
        result, valid, error = track_points_lk(image, image, points, window_size=9)
        assert valid.flatten().tolist() == [True, False, False]
        (result.sum() + error[valid].sum()).backward()
        assert torch.isfinite(image.grad).all()
        assert torch.isfinite(points.grad).all()

    def test_not_converged_and_escaped(self, device, lk_dtype):
        image = _texture(device, lk_dtype)
        nxt = _texture(device, lk_dtype, (1.2, 0.7))
        points = image.new_tensor([[[15.2, 16.4], [25.3, 24.2]]])
        for initial in (None, points + 100):
            result, valid, error = track_points_lk(
                image, nxt, points, initial, window_size=9, max_iterations=1, epsilon=1e-7
            )
            assert not valid.any()
            self.assert_close(result, points)
            assert torch.isinf(error).all()

    @pytest.mark.parametrize("batch,count", [(0, 2), (2, 0)])
    def test_empty(self, device, lk_dtype, batch, count):
        image = torch.empty(batch, 1, 11, 11, device=device, dtype=lk_dtype, requires_grad=True)
        points = torch.empty(batch, count, 2, device=device, dtype=lk_dtype, requires_grad=True)
        result, valid, error = track_points_lk(image, image, points, window_size=3)
        assert result.shape == (batch, count, 2)
        assert valid.shape == error.shape == (batch, count)
        (result.sum() + error.sum()).backward()
        assert image.grad is not None and points.grad is not None

    def test_noncontiguous(self, device, lk_dtype):
        image = _texture(device, lk_dtype).transpose(-1, -2)
        points = image.new_tensor([[[15.2, 16.4], [25.3, 24.2]]]).transpose(0, 1).expand(-1, 2, -1)
        image = image.expand(2, -1, -1, -1)
        result, valid, _ = track_points_lk(image, image, points, window_size=9)
        assert valid.all()
        self.assert_close(result, points)

    @pytest.mark.parametrize(
        "name,value",
        [
            ("window_size", 2),
            ("window_size", True),
            ("window_size", 51),
            ("max_iterations", 0),
            ("epsilon", 0),
            ("epsilon", float("nan")),
            ("min_eigenvalue", -1),
        ],
    )
    def test_invalid_parameters(self, device, name, value):
        image = _texture(device, torch.float32)
        with pytest.raises((TypeError, ValueError)):
            track_points_lk(image, image, image.new_zeros(1, 1, 2), **{name: value})

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64, torch.complex64])
    def test_unsupported_dtype(self, device, dtype):
        image = torch.zeros(1, 1, 17, 17, device=device, dtype=dtype)
        with pytest.raises(TypeError, match="float32 or float64"):
            track_points_lk(image, image, image.new_zeros(1, 1, 2))

    def test_invalid_shapes(self, device):
        image = _texture(device, torch.float32)
        points = image.new_zeros(1, 1, 2)
        for prev, nxt, pts, guess in [
            (image[:, 0], image, points, None),
            (image, image[..., :-1], points, None),
            (image, image, points[0], None),
            (image, image, points, points.expand(2, -1, -1)),
        ]:
            with pytest.raises((TypeError, ValueError)):
                track_points_lk(prev, nxt, pts, guess)

    def test_gradcheck(self, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64 gradcheck")
        image = _texture(device, torch.float64, height=13, width=15).requires_grad_()
        nxt = _texture(device, torch.float64, (0.2, -0.1), height=13, width=15).requires_grad_()
        points = image.new_tensor([[[6.3, 6.4]]]).requires_grad_()
        assert track_points_lk(image, nxt, points, window_size=5)[1].all()

        def fn(a, b, p):
            q, _, e = track_points_lk(a, b, p, window_size=5, epsilon=1e-4)
            return q, e

        self.gradcheck(fn, (image, nxt, points), fast_mode=True)

    def test_dynamo(self, device, lk_dtype, torch_optimizer):
        image = _texture(device, lk_dtype)
        points = image.new_tensor([[[15.2, 16.4], [0.0, 0.0]]])
        optimized = torch_optimizer(track_points_lk, fullgraph=True)
        eager = track_points_lk(image, image, points, window_size=9)
        compiled = optimized(image, image, points, window_size=9)
        for actual, expected in zip(compiled, eager):
            self.assert_close(actual, expected)
