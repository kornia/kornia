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

import numpy as np
import pytest
import torch

from kornia.feature import match_template_zncc

from testing.base import BaseTester


def _reference(image, template, min_variance=1e-8):
    """Independent scalar NumPy definition, centering each channel in extended precision."""
    image = image.detach().cpu().numpy().astype(np.longdouble)
    template = template.detach().cpu().numpy().astype(np.longdouble)
    b, _, h, w = image.shape
    th, tw = template.shape[-2:]
    scores = np.zeros((b, 1, h - th + 1, w - tw + 1), dtype=np.float64)
    valid = np.zeros(scores.shape, dtype=bool)
    for bi in range(b):
        t = template[0 if len(template) == 1 else bi]
        t = t - t.mean(axis=(-2, -1), keepdims=True)
        et = (t * t).mean()
        for y in range(h - th + 1):
            for x in range(w - tw + 1):
                p = image[bi, :, y : y + th, x : x + tw]
                p = p - p.mean(axis=(-2, -1), keepdims=True)
                ep = (p * p).mean()
                ok = np.isfinite(ep) and np.isfinite(et) and ep > min_variance and et > min_variance
                if ok:
                    valid[bi, 0, y, x] = True
                    scores[bi, 0, y, x] = (p * t).mean() / np.sqrt(ep) / np.sqrt(et)
    return torch.from_numpy(scores), torch.from_numpy(valid)


@pytest.fixture
def zncc_dtype(dtype):
    if dtype not in (torch.float32, torch.float64):
        pytest.skip("ZNCC supports float32 and float64; unsupported dtype rejection is tested separately.")
    return dtype


class TestMatchTemplateZNCC(BaseTester):
    @pytest.mark.parametrize("shared", [False, True])
    def test_full_score_map(self, device, zncc_dtype, shared):
        image = torch.rand(2, 3, 7, 9, device=device, dtype=zncc_dtype)
        template = torch.rand(1 if shared else 2, 3, 3, 4, device=device, dtype=zncc_dtype)
        scores, valid = match_template_zncc(image, template)
        expected, mask = _reference(image, template)
        self.assert_close(scores, expected.to(scores), atol=2e-6, rtol=2e-5)
        self.assert_close(valid, mask.to(device))
        assert scores.shape == (2, 1, 5, 6)

    def test_location_and_channel_offsets(self, device, zncc_dtype):
        image = torch.rand(2, 3, 9, 11, device=device, dtype=zncc_dtype)
        template = torch.rand(2, 3, 3, 4, device=device, dtype=zncc_dtype)
        image[0, :, 2:5, 4:8] = template[0]
        image[1, :, 5:8, 1:5] = template[1]
        scores, valid = match_template_zncc(image, template)
        assert scores.flatten(1).argmax(1).tolist() == [2 * 8 + 4, 5 * 8 + 1]
        assert valid.all()
        shifted, mask = match_template_zncc(
            image * 2 + image.new_tensor([1, -2, 3])[None, :, None, None], template * 0.5 + 5
        )
        self.assert_close(scores, shifted, atol=3e-6, rtol=1e-5)
        self.assert_close(valid, mask)

    def test_two_backgrounds_weak_texture(self, device, zncc_dtype):
        pattern = torch.tensor([[0, 2, 1], [3, -1, 4]], device=device, dtype=zncc_dtype) * 0.002
        image = torch.cat([pattern + 1000, pattern + 2000], -1)[None, None]
        template = pattern[None, None] + 500
        actual, valid = match_template_zncc(image, template)
        expected, mask = _reference(image, template)
        self.assert_close(actual, expected.to(actual), atol=3e-6, rtol=1e-5)
        self.assert_close(valid, mask.to(device))
        assert valid.all()

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
    def test_nonfinite_locality(self, device, zncc_dtype, bad):
        image = torch.rand(2, 2, 6, 7, device=device, dtype=zncc_dtype)
        template = torch.rand(2, 2, 2, 3, device=device, dtype=zncc_dtype)
        original = match_template_zncc(image, template)
        image[0, 0, 0, 0] = bad
        image.requires_grad_()
        scores, valid = match_template_zncc(image, template)
        assert not valid[0, 0, 0, 0]
        assert scores[0, 0, 0, 0] == 0
        self.assert_close(scores[0, :, 1:], original[0][0, :, 1:])
        self.assert_close(scores[1], original[0][1])
        scores.sum().backward()
        assert torch.isfinite(image.grad).all()
        template[0, 0, 0, 0] = bad
        scores, valid = match_template_zncc(image.detach(), template)
        assert not valid[0].any()
        assert valid[1].all()
        assert (scores[0] == 0).all()

    def test_degenerate_gradients(self, device, zncc_dtype):
        image = torch.rand(2, 2, 6, 7, device=device, dtype=zncc_dtype)
        image[0] = 5
        template = torch.ones(2, 2, 2, 3, device=device, dtype=zncc_dtype)
        template[0] = torch.rand_like(template[0])
        image.requires_grad_()
        template.requires_grad_()
        scores, valid = match_template_zncc(image, template, min_variance=0)
        assert not valid.any()
        assert (scores == 0).all()
        scores.sum().backward()
        assert torch.isfinite(image.grad).all() and torch.isfinite(template.grad).all()

    def test_variance_threshold_and_one_pixel(self, device, zncc_dtype):
        image = torch.tensor([[[[0.0, 2.0], [2.0, 0.0]]]], device=device, dtype=zncc_dtype)
        assert match_template_zncc(image, image, min_variance=0.99)[1].all()
        assert not match_template_zncc(image, image, min_variance=1.01)[1].any()
        assert not match_template_zncc(image, image[..., :1, :1])[1].any()

    def test_extreme_finite_scale(self, device, zncc_dtype):
        image = torch.tensor(
            [[[[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]]]], device=device, dtype=zncc_dtype
        )
        large = torch.finfo(zncc_dtype).max ** 0.75
        image = (image * large).requires_grad_()
        scores, valid = match_template_zncc(image, image[..., :2, :2])
        assert valid.all() and torch.isfinite(scores).all()
        self.assert_close(scores, scores.new_tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]))
        scores.sum().backward()
        assert torch.isfinite(image.grad).all()

    def test_noncontiguous(self, device, zncc_dtype):
        image = torch.rand(2, 3, 8, 7, device=device, dtype=zncc_dtype).transpose(-1, -2)
        template = image[:1, :, :2, :3]
        actual = match_template_zncc(image, template)
        expected = match_template_zncc(image.contiguous(), template.contiguous())
        for a, e in zip(actual, expected):
            self.assert_close(a, e)

    def test_tile_boundaries_and_backward(self, device, zncc_dtype):
        image = torch.rand(2, 3, 30, 40, device=device, dtype=zncc_dtype, requires_grad=True)
        template = torch.rand(1, 3, 9, 9, device=device, dtype=zncc_dtype, requires_grad=True)
        scores, valid = match_template_zncc(image, template)
        assert valid.all()
        actual_grad = torch.autograd.grad(scores.sum(), (image, template))
        # Direct full-window definition is small enough for a reference in this test.
        patches = image.unfold(2, 9, 1).unfold(3, 9, 1)
        patches = patches - patches.mean((-2, -1), keepdim=True)
        centered = template - template.mean((-2, -1), keepdim=True)
        centered = centered[:, :, None, None]
        expected = (patches * centered).sum((1, -2, -1))
        expected = expected / patches.square().sum((1, -2, -1)).sqrt()
        expected = expected / centered.square().sum((1, -2, -1)).sqrt()
        expected_grad = torch.autograd.grad(expected.sum(), (image, template))
        self.assert_close(scores[:, 0], expected, atol=1e-6, rtol=1e-5)
        for a, e in zip(actual_grad, expected_grad):
            self.assert_close(a, e, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64, torch.complex64])
    def test_unsupported_dtype(self, device, dtype):
        image = torch.zeros(1, 1, 4, 4, device=device, dtype=dtype)
        with pytest.raises(TypeError, match="float32 or float64"):
            match_template_zncc(image, image)

    def test_exceptions(self, device):
        image = torch.rand(2, 3, 6, 7, device=device)
        for template in [
            image[0],
            image[:1, :2],
            image.new_zeros(3, 3, 2, 2),
            image.new_zeros(1, 3, 8, 2),
            image[:, :, :0],
        ]:
            with pytest.raises((ValueError, TypeError)):
                match_template_zncc(image, template)
        for threshold in [-1, float("nan"), float("inf")]:
            with pytest.raises(ValueError):
                match_template_zncc(image, image, min_variance=threshold)

    def test_gradcheck(self, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64 gradcheck")
        image = torch.rand(2, 2, 4, 5, device=device, dtype=torch.float64, requires_grad=True)
        template = torch.rand(1, 2, 2, 3, device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(lambda i, t: match_template_zncc(i, t)[0], (image, template), fast_mode=True)

    def test_dynamo(self, device, zncc_dtype, torch_optimizer):
        image = torch.rand(2, 2, 6, 7, device=device, dtype=zncc_dtype)
        template = torch.rand(1, 2, 2, 3, device=device, dtype=zncc_dtype)
        optimized = torch_optimizer(match_template_zncc, fullgraph=True)
        for a, e in zip(optimized(image, template), match_template_zncc(image, template)):
            self.assert_close(a, e)

    def test_dynamo_with_gradients(self, device, zncc_dtype, torch_optimizer):
        image = torch.rand(1, 2, 4, 5, device=device, dtype=zncc_dtype, requires_grad=True)
        template = torch.rand(1, 2, 2, 3, device=device, dtype=zncc_dtype, requires_grad=True)
        expected = match_template_zncc(image, template)[0]
        expected_grad = torch.autograd.grad(expected.sum(), (image, template))
        optimized = torch_optimizer(match_template_zncc, fullgraph=True)
        actual = optimized(image, template)[0]
        actual_grad = torch.autograd.grad(actual.sum(), (image, template))
        self.assert_close(actual, expected)
        for a, e in zip(actual_grad, expected_grad):
            self.assert_close(a, e, atol=1e-5, rtol=1e-4)
