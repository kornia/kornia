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

import math

import pytest
import torch

import kornia

from testing.base import BaseTester, supports_reflect_padding, supports_replicate_padding


def _skip_without_half_padding(device: torch.device, dtype: torch.dtype) -> None:
    # torch 2.5.1 has no float16 CPU reflection_pad2d or replication_pad2d; these pins blur (reflect) and take
    # gradients (replicate).
    if not (supports_reflect_padding(device, dtype) and supports_replicate_padding(device, dtype)):
        pytest.skip(f"no reflection or replication padding kernel for {dtype} on {device.type}")


def _response_atol(expected: torch.Tensor, dtype: torch.dtype) -> float:
    r"""Absolute bound for a response map compared against hard-coded expected values.

    A response is a difference of quantities of order 1 -- the blurred image and its derivatives --
    so its absolute error is set by one ULP at 1.0 (9.8e-4 float16, 3.9e-3 bfloat16) and *not* by the
    response's own magnitude, which is 0.007-0.28 across these tests. The historical ``atol=1e-4``
    sits below that floor, so in half precision no implementation can meet it; four of these eight
    half-precision nodes were manifest entries for exactly that reason.

    Two ULP is that floor with a 2x margin -- the same bound, and the same reasoning, as
    ``TestALIKED::test_descriptor_normalized``. It is then capped at a twentieth of the largest
    expected value so that the assertion still discriminates: ``harris_response``'s expected map peaks
    at 6.9e-3, *below* bfloat16's own two-ULP floor, and an uncapped bound there would be satisfied by
    an all-zero response. The cap never binds in float32 or float64, and 1e-4 is far above their two
    ULP, so both keep the original bound exactly.
    """
    return min(max(1e-4, 2 * torch.finfo(dtype).eps), expected.abs().max().item() / 20.0)


class TestCornerHarris(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 3, 4, 4, device=device)
        harris = kornia.feature.CornerHarris(k=0.04).to(device)
        assert harris(inp).shape == (1, 3, 4, 4)

    def test_shape_batch(self, device):
        inp = torch.zeros(2, 6, 4, 4, device=device)
        harris = kornia.feature.CornerHarris(k=0.04).to(device)
        assert harris(inp).shape == (2, 6, 4, 4)

    def test_corners(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        ).float()

        expected = torch.tensor(
            [
                [
                    [
                        [0.0042, 0.0054, 0.0035, 0.0006, 0.0035, 0.0054, 0.0042],
                        [0.0054, 0.0068, 0.0046, 0.0014, 0.0046, 0.0068, 0.0054],
                        [0.0035, 0.0046, 0.0034, 0.0014, 0.0034, 0.0046, 0.0035],
                        [0.0006, 0.0014, 0.0014, 0.0006, 0.0014, 0.0014, 0.0006],
                        [0.0035, 0.0046, 0.0034, 0.0014, 0.0034, 0.0046, 0.0035],
                        [0.0054, 0.0068, 0.0046, 0.0014, 0.0046, 0.0068, 0.0054],
                        [0.0042, 0.0054, 0.0035, 0.0006, 0.0035, 0.0054, 0.0042],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        ).float()
        harris = kornia.feature.CornerHarris(k=0.04).to(device)
        scores = harris(inp)
        self.assert_close(scores, expected, atol=1e-4, rtol=1e-3)

    def test_corners_batch(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)
        expected = (
            torch.tensor(
                [
                    [
                        [0.0415, 0.0541, 0.0346, 0.0058, 0.0346, 0.0541, 0.0415],
                        [0.0541, 0.0678, 0.0457, 0.0145, 0.0457, 0.0678, 0.0541],
                        [0.0346, 0.0457, 0.0335, 0.0139, 0.0335, 0.0457, 0.0346],
                        [0.0058, 0.0145, 0.0139, 0.0064, 0.0139, 0.0145, 0.0058],
                        [0.0346, 0.0457, 0.0335, 0.0139, 0.0335, 0.0457, 0.0346],
                        [0.0541, 0.0678, 0.0457, 0.0145, 0.0457, 0.0678, 0.0541],
                        [0.0415, 0.0541, 0.0346, 0.0058, 0.0346, 0.0541, 0.0415],
                    ],
                    [
                        [0.0415, 0.0547, 0.0447, 0.0440, 0.0490, 0.0182, 0.0053],
                        [0.0547, 0.0688, 0.0557, 0.0549, 0.0610, 0.0229, 0.0066],
                        [0.0447, 0.0557, 0.0444, 0.0437, 0.0489, 0.0168, 0.0035],
                        [0.0440, 0.0549, 0.0437, 0.0431, 0.0481, 0.0166, 0.0034],
                        [0.0490, 0.0610, 0.0489, 0.0481, 0.0541, 0.0205, 0.0060],
                        [0.0182, 0.0229, 0.0168, 0.0166, 0.0205, 0.0081, 0.0025],
                        [0.0053, 0.0066, 0.0035, 0.0034, 0.0060, 0.0025, 0.0008],
                    ],
                ],
                device=device,
                dtype=dtype,
            ).repeat(2, 1, 1, 1)
            / 10.0
        )
        scores = kornia.feature.harris_response(inp, k=0.04)
        self.assert_close(scores, expected, atol=_response_atol(expected, dtype), rtol=1e-4)

    def test_gradcheck(self, device):
        k = 0.04
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.harris_response, (img, k), nondet_tol=1e-4)


class TestCornerGFTT(BaseTester):
    def test_sigmas_wrong_length_is_rejected_5724(self, device):
        # B = 1 with three sigmas used to broadcast silently to a (3, C, H, W) response
        inp = torch.rand(1, 1, 8, 8, device=device)
        with pytest.raises(ValueError, match="Invalid sigmas shape"):
            kornia.feature.gftt_response(inp, sigmas=torch.ones(3, device=device))

    def test_sigmas_wrong_ndim_is_rejected_5724(self, device):
        inp = torch.rand(2, 1, 8, 8, device=device)
        with pytest.raises(ValueError, match="Invalid sigmas shape"):
            kornia.feature.gftt_response(inp, sigmas=torch.ones(2, 1, device=device))

    def test_sigmas_wrong_type_is_rejected_5724(self, device):
        inp = torch.rand(2, 1, 8, 8, device=device)
        with pytest.raises(TypeError, match=r"sigmas type is not a torch\.Tensor"):
            kornia.feature.gftt_response(inp, sigmas=[1.0, 2.0])  # type: ignore[arg-type]

    def test_sigmas_scale_the_response_5724(self, device):
        inp = torch.rand(2, 1, 8, 8, device=device)
        sigmas = torch.tensor([1.0, 2.0], device=device)
        plain = kornia.feature.gftt_response(inp)
        scaled = kornia.feature.gftt_response(inp, sigmas=sigmas)
        assert scaled.shape == plain.shape
        self.assert_close(scaled, plain * sigmas.pow(4).view(-1, 1, 1, 1))

    def test_shape(self, device):
        inp = torch.ones(1, 3, 4, 4, device=device)
        shi_tomasi = kornia.feature.CornerGFTT().to(device)
        assert shi_tomasi(inp).shape == (1, 3, 4, 4)

    def test_shape_batch(self, device):
        inp = torch.zeros(2, 6, 4, 4, device=device)
        shi_tomasi = kornia.feature.CornerGFTT().to(device)
        assert shi_tomasi(inp).shape == (2, 6, 4, 4)

    def test_corners(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        ).float()

        expected = torch.tensor(
            [
                [
                    [
                        [0.0379, 0.0456, 0.0283, 0.0121, 0.0283, 0.0456, 0.0379],
                        [0.0456, 0.0598, 0.0402, 0.0168, 0.0402, 0.0598, 0.0456],
                        [0.0283, 0.0402, 0.0545, 0.0245, 0.0545, 0.0402, 0.0283],
                        [0.0121, 0.0168, 0.0245, 0.0276, 0.0245, 0.0168, 0.0121],
                        [0.0283, 0.0402, 0.0545, 0.0245, 0.0545, 0.0402, 0.0283],
                        [0.0456, 0.0598, 0.0402, 0.0168, 0.0402, 0.0598, 0.0456],
                        [0.0379, 0.0456, 0.0283, 0.0121, 0.0283, 0.0456, 0.0379],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        ).float()
        shi_tomasi = kornia.feature.CornerGFTT().to(device)
        scores = shi_tomasi(inp)
        self.assert_close(scores, expected, atol=1e-4, rtol=1e-3)

    def test_corners_batch(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)
        expected = torch.tensor(
            [
                [
                    [0.0379, 0.0456, 0.0283, 0.0121, 0.0283, 0.0456, 0.0379],
                    [0.0456, 0.0598, 0.0402, 0.0168, 0.0402, 0.0598, 0.0456],
                    [0.0283, 0.0402, 0.0545, 0.0245, 0.0545, 0.0402, 0.0283],
                    [0.0121, 0.0168, 0.0245, 0.0276, 0.0245, 0.0168, 0.0121],
                    [0.0283, 0.0402, 0.0545, 0.0245, 0.0545, 0.0402, 0.0283],
                    [0.0456, 0.0598, 0.0402, 0.0168, 0.0402, 0.0598, 0.0456],
                    [0.0379, 0.0456, 0.0283, 0.0121, 0.0283, 0.0456, 0.0379],
                ],
                [
                    [0.0379, 0.0462, 0.0349, 0.0345, 0.0443, 0.0248, 0.0112],
                    [0.0462, 0.0608, 0.0488, 0.0483, 0.0581, 0.0274, 0.0119],
                    [0.0349, 0.0488, 0.0669, 0.0664, 0.0460, 0.0191, 0.0084],
                    [0.0345, 0.0483, 0.0664, 0.0660, 0.0455, 0.0189, 0.0083],
                    [0.0443, 0.0581, 0.0460, 0.0455, 0.0555, 0.0262, 0.0114],
                    [0.0248, 0.0274, 0.0191, 0.0189, 0.0262, 0.0172, 0.0084],
                    [0.0112, 0.0119, 0.0084, 0.0083, 0.0114, 0.0084, 0.0046],
                ],
            ],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)
        shi_tomasi = kornia.feature.CornerGFTT().to(device)
        scores = shi_tomasi(inp)
        self.assert_close(scores, expected, atol=_response_atol(expected, dtype), rtol=1e-4)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.gftt_response, (img), nondet_tol=1e-4)

    def test_flat_patch_gradient(self, device, dtype):
        # the structure tensor is exactly zero inside a flat 12 x 12 patch, where the old sqrt(abs(trace^2 - 4 det))
        # backward was 0 * inf = nan for the whole image
        img = torch.rand(1, 1, 32, 32, device=device, dtype=dtype)
        img[..., 8:20, 8:20] = 0.5
        img.requires_grad_(True)
        # anomaly mode also sees a nan that a ``torch.where`` masks out of the result, i.e. a sqrt evaluated at 0
        with torch.autograd.detect_anomaly():
            scores = kornia.feature.gftt_response(img)
            (grad,) = torch.autograd.grad(scores.sum(), img)
        assert torch.isfinite(scores).all()
        assert torch.isfinite(grad).all()
        self.assert_close(scores[..., 14, 14], torch.zeros_like(scores[..., 14, 14]))
        self.assert_close(grad[..., 14, 14], torch.zeros_like(grad[..., 14, 14]))

    def test_smaller_eigenvalue(self, device, dtype):
        img = torch.rand(2, 3, 24, 20, device=device, dtype=dtype)
        gradients = kornia.filters.spatial_gradient(img)
        dx, dy = gradients[:, :, 0], gradients[:, :, 1]
        dx2 = kornia.filters.gaussian_blur2d(dx**2, (7, 7), (1.0, 1.0))
        dy2 = kornia.filters.gaussian_blur2d(dy**2, (7, 7), (1.0, 1.0))
        dxy = kornia.filters.gaussian_blur2d(dx * dy, (7, 7), (1.0, 1.0))
        structure = torch.stack((torch.stack((dx2, dxy), -1), torch.stack((dxy, dy2), -1)), -2)
        expected = torch.linalg.eigvalsh(structure.cpu().double())[..., 0].to(device, dtype)
        self.assert_close(kornia.feature.gftt_response(img), expected)


class TestBlobHessian(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 3, 4, 4, device=device)
        shi_tomasi = kornia.feature.BlobHessian().to(device)
        assert shi_tomasi(inp).shape == (1, 3, 4, 4)

    def test_shape_batch(self, device):
        inp = torch.zeros(2, 6, 4, 4, device=device)
        shi_tomasi = kornia.feature.BlobHessian().to(device)
        assert shi_tomasi(inp).shape == (2, 6, 4, 4)

    def test_blobs_batch(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)
        expected = torch.tensor(
            [
                [
                    [-0.0137, -0.0332, -0.0342, -0.0332, -0.0137, -0.0010, 0.0000],
                    [-0.0332, 0.0098, 0.0752, 0.0098, -0.0332, -0.0049, 0.0000],
                    [-0.0342, 0.0752, 0.1914, 0.0752, -0.0342, -0.0068, 0.0000],
                    [-0.0332, 0.0098, 0.0752, 0.0098, -0.0332, -0.0049, 0.0000],
                    [-0.0137, -0.0332, -0.0342, -0.0332, -0.0137, -0.0010, 0.0000],
                    [-0.0010, -0.0049, -0.0068, -0.0049, -0.0010, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                ],
                [
                    [-0.0137, -0.0332, -0.0332, -0.0137, -0.0010, 0.0000, 0.0000],
                    [-0.0332, 0.0820, 0.0693, -0.0039, 0.0117, -0.0010, 0.0000],
                    [-0.0332, 0.0693, 0.0137, -0.0352, -0.0039, -0.0137, -0.0010],
                    [-0.0137, -0.0039, -0.0352, 0.0137, 0.0693, -0.0332, -0.0059],
                    [-0.0010, 0.0117, -0.0039, 0.0693, 0.0820, -0.0332, -0.0059],
                    [0.0000, -0.0010, -0.0137, -0.0332, -0.0332, -0.0137, -0.0010],
                    [0.0000, 0.0000, -0.0010, -0.0059, -0.0059, -0.0010, 0.0000],
                ],
            ],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)
        shi_tomasi = kornia.feature.BlobHessian().to(device)
        scores = shi_tomasi(inp)
        self.assert_close(scores, expected, atol=_response_atol(expected, dtype), rtol=1e-4)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.hessian_response, (img), nondet_tol=1e-4)


class TestBlobDoGSingle(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 3, 9, 9, device=device)
        shi_tomasi = kornia.feature.BlobDoGSingle().to(device)
        assert shi_tomasi(inp).shape == (1, 3, 9, 9)

    def test_shape_batch(self, device):
        inp = torch.zeros(2, 6, 9, 9, device=device)
        shi_tomasi = kornia.feature.BlobHessian().to(device)
        assert shi_tomasi(inp).shape == (2, 6, 9, 9)

    def test_blobs_batch(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ],
            device=device,
            dtype=dtype,
        ).expand(2, 2, 7, 7)
        expected = torch.tensor(
            [
                [
                    [
                        [0.1640, 0.0811, -0.0148, 0.0040, 0.0850, 0.0962, 0.0754],
                        [0.0811, -0.0249, -0.1416, -0.0950, 0.0437, 0.0894, 0.0754],
                        [-0.0148, -0.1416, -0.2758, -0.2005, -0.0044, 0.0777, 0.0718],
                        [0.0040, -0.0950, -0.2005, -0.1445, 0.0045, 0.0648, 0.0586],
                        [0.0850, 0.0437, -0.0044, 0.0045, 0.0441, 0.0488, 0.0382],
                        [0.0962, 0.0894, 0.0777, 0.0648, 0.0488, 0.0295, 0.0197],
                        [0.0754, 0.0754, 0.0718, 0.0586, 0.0382, 0.0197, 0.0124],
                    ],
                    [
                        [0.0689, -0.0056, -0.0254, 0.0802, 0.1118, 0.0708, 0.0491],
                        [-0.0056, -0.0895, -0.1031, 0.0362, 0.0986, 0.0771, 0.0621],
                        [-0.0254, -0.1031, -0.1423, -0.0651, 0.0039, 0.0567, 0.0794],
                        [0.0802, 0.0362, -0.0651, -0.1795, -0.1617, -0.0048, 0.0771],
                        [0.1118, 0.0986, 0.0039, -0.1617, -0.1706, -0.0095, 0.0761],
                        [0.0708, 0.0771, 0.0567, -0.0048, -0.0095, 0.0521, 0.0836],
                        [0.0491, 0.0621, 0.0794, 0.0771, 0.0761, 0.0836, 0.0858],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        ).expand(2, 2, 7, 7)
        det = kornia.feature.BlobDoGSingle(1.0, 1.6).to(device)
        scores = det(inp)
        self.assert_close(scores, expected, atol=_response_atol(expected, dtype), rtol=1e-4)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 9, 11
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.dog_response_single, (img), nondet_tol=1e-4)


def _conv_grid(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the float64 CPU ``(y, x)`` pixel-centre grids of a ``height x width`` image."""
    return torch.meshgrid(
        torch.arange(height, dtype=torch.float64), torch.arange(width, dtype=torch.float64), indexing="ij"
    )


def _conv_rectangle_and_blob(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """48 x 64 image: a bright rectangle (rows 10-25, columns 14-41) and a 0.5-high blob of sigma 3 at (52, 36)."""
    yy, xx = _conv_grid(48, 64)
    img = torch.zeros(48, 64, dtype=torch.float64)
    img[10:26, 14:42] = 1.0
    img = img + 0.5 * torch.exp(-((xx - 52.0) ** 2 + (yy - 36.0) ** 2) / 18.0)
    return img.view(1, 1, 48, 64).to(device, dtype)


def _conv_blob(x0: float, y0: float, sigma: float) -> torch.Tensor:
    """Float64 CPU Gaussian blob of ``sigma`` centred at pixel ``(x0, y0)`` of a 48 x 64 image, ``(1, 1, 48, 64)``."""
    yy, xx = _conv_grid(48, 64)
    return torch.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma**2)).view(1, 1, 48, 64)


class TestConventionsResponses(BaseTester):
    @pytest.mark.parametrize(
        "response, factor",
        [
            (kornia.feature.harris_response, 16.0),
            (kornia.feature.gftt_response, 4.0),
            (kornia.feature.hessian_response, 4.0),
        ],
        ids=["harris", "gftt", "hessian"],
    )
    def test_convention_response_scales_with_image_intensity(self, device, dtype, response, factor):
        # Harris is quartic in the gradients, the smaller structure-tensor eigenvalue and the Hessian determinant are
        # quadratic, and none of them sees an offset: f(2 x + 1) = factor * f(x), so a threshold tuned on images in
        # [0, 1] keeps nothing or everything on images in [0, 255]. Factors: response(2 * x) / response(x), CPU float64.
        _skip_without_half_padding(device, dtype)
        img = _conv_rectangle_and_blob(device, dtype)
        expected = factor * response(img)
        # half precision rounds the offset image; the bound is relative to the peak response
        scale = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-6
        self.assert_close(response(2 * img + 1), expected, rtol=0.0, atol=scale * expected.abs().max().item())

    def test_convention_harris_response_is_negative_on_an_edge_and_positive_at_a_corner(self, device, dtype):
        # R = det(M) - k trace(M) ** 2 is not clamped: positive at a corner, negative along an edge.
        # harris_response(img)[0, 0, 10, [14, 28]] on CPU float64: the rectangle's top-left corner and its top edge.
        _skip_without_half_padding(device, dtype)
        img = _conv_rectangle_and_blob(device, dtype)
        response = kornia.feature.harris_response(img)
        expected = torch.tensor([5.2460662e-03, -1.0274798e-03], device=device, dtype=dtype)
        rtol = {torch.float16: 2e-3, torch.bfloat16: 2e-2}.get(dtype, 1e-5)
        self.assert_close(response[0, 0, 10, [14, 28]], expected, rtol=rtol, atol=0.0)
        # relabelling: the response of the transposed image is the transposed response, to a rounding of the peak
        scale = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        transposed = kornia.feature.harris_response(img.transpose(-1, -2))
        self.assert_close(transposed, response.transpose(-1, -2), rtol=0.0, atol=scale * response.abs().max().item())

    @pytest.mark.parametrize("grads_mode", ["sobel", "diff"])
    def test_convention_harris_response_on_a_ramp_is_minus_k_times_the_gradient_norm_to_the_fourth(
        self, device, dtype, grads_mode
    ):
        # Both gradient modes return intensity per pixel and the 7 x 7 window is a weighted mean, so on the ramp
        # I = x / 4 + y / 8 the structure tensor is the outer product of (1/4, 1/8): det = 0 and
        # R = -k (1/16 + 1/64) ** 2 = -2.44140625e-4 for k = 0.04. The ramp is exact in every dtype; the outer four
        # rows and columns see the replicate-padded Sobel border.
        _skip_without_half_padding(device, dtype)
        yy, xx = _conv_grid(48, 64)
        ramp = (0.25 * xx + 0.125 * yy).view(1, 1, 48, 64).to(device, dtype)
        interior = kornia.feature.harris_response(ramp, 0.04, grads_mode)[..., 4:-4, 4:-4]
        self.assert_close(interior, torch.full_like(interior, -2.44140625e-4), rtol=1e-3, atol=0.0)

    def test_convention_hessian_response_is_positive_at_bright_and_dark_blobs_and_negative_at_a_saddle(
        self, device, dtype
    ):
        # The signed determinant Ixx Iyy - Ixy ** 2: a blob of either polarity is a positive maximum, a saddle negative.
        # hessian_response(...)[:, 0, 20, 30] on CPU float64: blob of sigma 3 at (30, 20), bright and dark,
        # 8.9222449e-03 each; saddle (x - 30)(y - 20) / 100, -(1 / 100) ** 2.
        _skip_without_half_padding(device, dtype)
        blob = _conv_blob(30.0, 20.0, 3.0)
        yy, xx = _conv_grid(48, 64)
        saddle = ((xx - 30.0) * (yy - 20.0) / 100.0).view(1, 1, 48, 64)
        imgs = torch.cat([blob, 1.0 - blob, saddle]).to(device, dtype)
        response = kornia.feature.hessian_response(imgs)[:, 0, 20, 30]
        expected = torch.tensor([8.9222449e-03, 8.9222449e-03, -1.0e-04], device=device, dtype=dtype)
        rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        self.assert_close(response, expected, rtol=rtol, atol=1e-7)

    @pytest.mark.parametrize(
        "response",
        [kornia.feature.harris_response, kornia.feature.gftt_response, kornia.feature.hessian_response],
        ids=["harris", "gftt", "hessian"],
    )
    def test_convention_sigmas_multiply_the_response_by_sigma_to_the_fourth(self, device, dtype, response):
        # sigmas[b] multiplies the response of image b by sigmas[b] ** 4 in all three (VLFeat covdet.c's scale
        # normalisation of Harris and of the Hessian determinant; GFTT takes the same power): 2 ** 4 = 16 and
        # 0.5 ** 4 = 1 / 16. The bound is 2 * eps relative per element: the power itself may round (MPS gives 0.5 ** 4
        # as 1 / 16 minus 2 ** -27 in float32), the product rounds once more.
        _skip_without_half_padding(device, dtype)
        img = _conv_rectangle_and_blob(device, dtype).repeat(2, 1, 1, 1)
        sigmas = torch.tensor([2.0, 0.5], device=device, dtype=dtype)
        expected = response(img) * torch.tensor([16.0, 0.0625], device=device, dtype=dtype).view(-1, 1, 1, 1)
        self.assert_close(response(img, sigmas=sigmas), expected, rtol=2 * torch.finfo(dtype).eps, atol=0.0)

    @pytest.mark.parametrize("angle, exponent_bounds", [(30.0, (2.0, 3.0)), (90.0, (3.0, 4.0))])
    def test_convention_gftt_blur_scaling_depends_on_corner_geometry(self, device, dtype, angle, exponent_bounds):
        # These binary wedges give effective peak-decay exponents 2.53 and 3.31 over sigma=1.6..4 on CPU float64.
        # They lie on opposite sides of 3: sigma ** 4 is not a universally closer correction than sigma ** 2.
        # The bounds describe these fixtures only; the separate sigmas pin checks the actual fourth-power multiplier.
        _skip_without_half_padding(device, dtype)
        n = 97
        yy, xx = _conv_grid(n, n)
        wedge = torch.atan2(yy - n // 2, xx - n // 2).abs() <= math.radians(angle / 2)
        img = wedge[None, None].to(device, dtype)
        peaks = []
        for sigma, ksize in ((1.6, 15), (4.0, 33)):
            blurred = kornia.filters.gaussian_blur2d(img, (ksize, ksize), (sigma, sigma))
            response = kornia.feature.gftt_response(blurred)[..., n // 2 - 20 : n // 2 + 21, n // 2 - 20 : n // 2 + 21]
            peaks.append(response.max().cpu().double())
        exponent = (torch.log(peaks[0] / peaks[1]) / math.log(4.0 / 1.6)).item()
        assert exponent_bounds[0] < exponent < exponent_bounds[1]

    def test_convention_dog_response_is_each_level_minus_the_previous_one(self, device, dtype):
        # The level axis of (B, C, D, H, W) is dim 2, and output level i is L[i + 1] - L[i]: D - 1 levels.
        # Level d holds d ** 2 (plus 10 in channel 1), so the differences are 1, 3, 5 in both channels.
        levels = torch.arange(4, dtype=torch.float64) ** 2
        stack = levels.view(1, 1, 4, 1, 1) + torch.tensor([0.0, 10.0], dtype=torch.float64).view(1, 2, 1, 1, 1)
        out = kornia.feature.dog_response(stack.expand(1, 2, 4, 5, 7).to(device, dtype))
        assert out.shape == (1, 2, 3, 5, 7)
        expected = torch.tensor([1.0, 3.0, 5.0], device=device, dtype=dtype).view(1, 1, 3, 1, 1).expand(1, 2, 3, 5, 7)
        self.assert_close(out, expected)

    def test_convention_dog_is_negative_at_a_bright_blob(self, device, dtype):
        # Both DoG functions subtract the finer blur from the coarser (Lowe's D = L(k sigma) - L(sigma)), so a bright
        # blob is a minimum and a dark one a maximum. dog_response_single(img, 1.0, 1.6)[:, 0, 20, 30] at a blob of
        # sigma 3 centred at (30, 20), CPU float64: -0.1214043 (bright), +0.1214043 (dark).
        _skip_without_half_padding(device, dtype)
        blob = _conv_blob(30.0, 20.0, 3.0)
        imgs = torch.cat([blob, 1.0 - blob]).to(device, dtype)
        single = kornia.feature.dog_response_single(imgs, 1.0, 1.6)
        expected = torch.tensor([-0.1214043, 0.1214043], device=device, dtype=dtype)
        rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        self.assert_close(single[:, 0, 20, 30], expected, rtol=rtol, atol=0.0)
        # The stack form agrees: the same two blurs (kernel int(8 sigma + 1), made odd: 9 and 13) as levels 0 and 1.
        stack = torch.stack(
            [
                kornia.filters.gaussian_blur2d(imgs, (9, 9), (1.0, 1.0)),
                kornia.filters.gaussian_blur2d(imgs, (13, 13), (1.6, 1.6)),
            ],
            dim=2,
        )
        self.assert_close(kornia.feature.dog_response(stack)[:, :, 0], single)

    @pytest.mark.parametrize("name", ["harris", "gftt", "hessian", "dog_single", "dog"])
    def test_convention_response_modules_return_their_functions(self, device, dtype, name):
        # Each module is its function with the constructor arguments bound; BlobDoG and BlobDoGSingle ignore `sigmas`.
        _skip_without_half_padding(device, dtype)
        img = torch.rand(2, 1, 24, 30, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
        img = img.to(device, dtype)
        sigmas = torch.tensor([2.0, 0.5], device=device, dtype=dtype)
        if name == "harris":  # k = 1 / 16 is exact in every dtype, so the buffer cast does not round it
            module = kornia.feature.CornerHarris(0.0625, grads_mode="diff")
            expected = kornia.feature.harris_response(img, 0.0625, "diff", sigmas)
        elif name == "gftt":
            module = kornia.feature.CornerGFTT(grads_mode="diff")
            expected = kornia.feature.gftt_response(img, "diff", sigmas)
        elif name == "hessian":
            module = kornia.feature.BlobHessian(grads_mode="diff")
            expected = kornia.feature.hessian_response(img, "diff", sigmas)
        elif name == "dog_single":
            module = kornia.feature.BlobDoGSingle(1.2, 2.0)
            expected = kornia.feature.dog_response_single(img, 1.2, 2.0)
        else:  # BlobDoG takes the (B, C, D, H, W) scale stack
            img = img.view(1, 1, 2, 24, 30)
            module = kornia.feature.BlobDoG()
            expected = kornia.feature.dog_response(img)
        out = module.to(device, dtype)(img, sigmas)
        self.assert_close(out, expected, rtol=0.0, atol=0.0)

    def test_convention_dog_kernel_size_is_eight_sigma_plus_one_rounded_up_to_odd(self, device, dtype):
        # Each blur of dog_response_single has int(8 * sigma + 1) taps, rounded up to an odd number: sigma 0.875 gives
        # int(8.0) = 8, so 9 taps reaching 4 pixels (rounding down would give 7, reaching 3), and sigma 0.5 gives 5.
        # On an impulse at (13, 9) the response is therefore positive 4 pixels away and exactly zero 5 pixels away,
        # along x and along y.
        _skip_without_half_padding(device, dtype)
        img = torch.zeros(1, 1, 23, 31, device=device, dtype=dtype)
        img[..., 9, 13] = 1.0
        out = kornia.feature.dog_response_single(img, 0.5, 0.875)[0, 0]
        reach_4 = out[[9, 9, 13, 5], [17, 9, 13, 13]]
        reach_5 = out[[9, 9, 14, 4], [18, 8, 13, 13]]
        assert bool((reach_4 > 0).all())
        assert bool((reach_5 == 0).all())
