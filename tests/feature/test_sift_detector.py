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
# limitations under the License.

import torch
from torch import nn

from kornia.feature import get_laf_center, laf_is_filled
from kornia.feature.sift_detector import _SIFTScaleSpaceDetector

from testing.base import BaseTester


class _FixedPyramid(nn.Module):
    def __init__(self, dog: torch.Tensor) -> None:
        super().__init__()
        self.dog = dog

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        gaussian = torch.cat([torch.zeros_like(self.dog[:, :1]), self.dog.cumsum(1)], 1).unsqueeze(1)
        return [gaussian.to(image)]


def _quadratic_dog(
    device: torch.device,
    dtype: torch.dtype,
    amplitude: float = 1.0,
    curvature: tuple[float, float, float] = (1.0, 1.0, 1.0),
    center: tuple[float, float, float] = (2.2, 10.3, 11.2),
) -> torch.Tensor:
    # The detector constructs its pyramid in float32 for reduced-precision
    # inputs; preserve the analytic curvature in this substitute pyramid too.
    dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    s = torch.arange(5, device=device, dtype=dtype).view(1, 5, 1, 1)
    y = torch.arange(32, device=device, dtype=dtype).view(1, 1, 32, 1)
    x = torch.arange(32, device=device, dtype=dtype).view(1, 1, 1, 32)
    cs, cy, cx = curvature
    return amplitude * (
        20.0 - cs * (s - center[0]).square() - cy * (y - center[1]).square() - cx * (x - center[2]).square()
    )


class TestSIFTScaleSpaceDetector(BaseTester):
    def test_refines_analytic_extremum_and_keeps_tiny_amplitude(self, device, dtype):
        image = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        dog = _quadratic_dog(device, dtype)
        detector = _SIFTScaleSpaceDetector(4, _FixedPyramid(dog)).to(device, dtype)
        lafs, responses = detector(image)
        filled = laf_is_filled(lafs)
        assert filled[0, 0]
        expected = torch.tensor([11.2 * 0.5, 10.3 * 0.5], device=device, dtype=dtype)
        self.assert_close(get_laf_center(lafs)[0, 0], expected, rtol=2e-3, atol=2e-3)
        amplitude = 1e-4 if dtype == torch.float16 else 1e-8
        tiny_lafs, tiny_responses = _SIFTScaleSpaceDetector(4, _FixedPyramid(dog * amplitude)).to(device, dtype)(image)
        assert laf_is_filled(tiny_lafs)[0, 0]
        self.assert_close(get_laf_center(tiny_lafs)[0, 0], get_laf_center(lafs)[0, 0], rtol=2e-3, atol=2e-3)
        self.assert_close(tiny_responses[0, 0], responses[0, 0] * amplitude, rtol=5e-3, atol=1e-12)

    def test_keeps_edge_like_anisotropic_extremum(self, device, dtype):
        image = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        dog = _quadratic_dog(device, dtype, curvature=(1.0, 1e-3, 1.0))
        lafs, _ = _SIFTScaleSpaceDetector(4, _FixedPyramid(dog)).to(device, dtype)(image)
        assert laf_is_filled(lafs)[0, 0]

    def test_minimum_is_ranked_by_absolute_response_and_output_is_padded(self, device, dtype):
        image = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        maximum = _quadratic_dog(device, dtype)
        minimum = -_quadratic_dog(device, dtype) + 1.0
        # Put a stronger negative extremum in the second batch item; both signs
        # are valid and responses are absolute refined DoG values.
        dog = torch.cat([maximum, minimum * 2.0], 0)
        lafs, responses = _SIFTScaleSpaceDetector(3, _FixedPyramid(dog)).to(device, dtype)(image.expand(2, -1, -1, -1))
        assert laf_is_filled(lafs)[:, 0].all()
        assert (responses[:, 0] > 0).all()
        assert (~laf_is_filled(lafs)[:, 1:]).all()
        assert not responses[:, 1:].any()

    def test_mask_broadcast_zero_and_fractional_weight(self, device, dtype):
        image = torch.zeros(2, 1, 32, 32, device=device, dtype=dtype)
        dog = _quadratic_dog(device, dtype).expand(2, -1, -1, -1).clone()
        detector = _SIFTScaleSpaceDetector(2, _FixedPyramid(dog)).to(device, dtype)
        _, unmasked = detector(image)
        mask = torch.full((1, 1, 32, 32), 0.25, device=device, dtype=dtype)
        lafs, weighted = detector(image, mask)
        assert laf_is_filled(lafs)[:, 0].all()
        self.assert_close(weighted[:, 0], unmasked[:, 0] * 0.25, rtol=2e-3, atol=2e-3)
        zero_lafs, zero_responses = detector(image, torch.zeros_like(mask))
        assert not zero_lafs.any() and not zero_responses.any()

    def test_flat_and_singular_refinement_reject_without_oob_access(self, device, dtype):
        image = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        flat = torch.zeros(1, 5, 32, 32, device=device, dtype=dtype)
        lafs, responses = _SIFTScaleSpaceDetector(2, _FixedPyramid(flat)).to(device, dtype)(image)
        assert not lafs.any() and not responses.any()
        detector = _SIFTScaleSpaceDetector(1, _FixedPyramid(flat)).to(device, dtype)
        b = torch.zeros(1, device=device, dtype=torch.long)
        s = torch.full_like(b, 2)
        y = torch.full_like(b, 10)
        x = torch.full_like(b, 10)
        *_, converged = detector._refine(flat, b, s, y, x)
        assert not converged.any()

    def test_fractional_offsets_at_right_bottom_border(self, device, dtype):
        image = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        dog = _quadratic_dog(device, dtype, center=(2.2, 26.3, 26.2))
        lafs, _ = _SIFTScaleSpaceDetector(1, _FixedPyramid(dog))(image)
        assert laf_is_filled(lafs).all()
        expected = torch.tensor([26.2, 26.3], device=device, dtype=dtype) * 0.5
        self.assert_close(get_laf_center(lafs)[0, 0], expected, atol=0.01, rtol=0.001)

    def test_topk_sorts_by_refined_response(self, device, dtype):
        image = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        weaker = _quadratic_dog(device, dtype)
        stronger = _quadratic_dog(device, dtype, amplitude=2.0, center=(2.2, 10.3, 21.2))
        dog = torch.maximum(weaker, stronger)
        lafs, scores = _SIFTScaleSpaceDetector(2, _FixedPyramid(dog))(image)
        assert laf_is_filled(lafs).all()
        assert scores[0, 0] > scores[0, 1]
        expected = torch.tensor([[21.2, 10.3], [11.2, 10.3]], device=device, dtype=dtype) * 0.5
        self.assert_close(get_laf_center(lafs)[0], expected, atol=0.01, rtol=0.001)

    def test_refinement_backward_is_finite(self, device):
        dog = _quadratic_dog(device, torch.float32).requires_grad_()
        detector = _SIFTScaleSpaceDetector(1, _FixedPyramid(dog))
        lafs, responses = detector(torch.zeros(1, 1, 32, 32, device=device))
        (lafs.sum() + responses.sum()).backward()
        assert torch.isfinite(dog.grad).all()
