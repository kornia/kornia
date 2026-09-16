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

import math

import pytest
import torch

import kornia.augmentation as K
from kornia.constants import Resample, SamplePadding

from testing.base import BaseTester, supports_bilinear_2d_grid_sample


class TestDenseGeometricConventions(BaseTester):
    @pytest.fixture(autouse=True)
    def require_sampler(self, device, dtype):
        if not supports_bilinear_2d_grid_sample(device, dtype):
            pytest.skip("2D bilinear grid sampling is unavailable for this device/dtype")

    @pytest.mark.parametrize("align_corners", [False, True])
    def test_convention_elastic_zero_displacement(self, device, dtype, align_corners):
        image = torch.arange(35, device=device, dtype=dtype).reshape(1, 1, 5, 7) / 35
        aug = K.RandomElasticTransform(kernel_size=(3, 5), alpha=(0.0, 0.0), align_corners=align_corners, p=1)
        self.assert_close(aug(image), image)
        assert aug.flags["resample"] == Resample.BILINEAR
        assert aug.flags["padding_mode"] == "zeros"
        assert K.RandomElasticTransform().flags["align_corners"] is False

    @pytest.mark.parametrize("axis", [0, 1])
    def test_convention_elastic_normalized_xy_displacement_and_clamp(self, device, dtype, axis):
        image = torch.arange(35, device=device, dtype=dtype).reshape(1, 1, 5, 7)
        # With corner alignment, one input pixel is 2/(size - 1) normalized units.
        alpha = (2 / 6, 0.0) if axis == 0 else (0.0, 2 / 4)
        aug = K.RandomElasticTransform(kernel_size=(1, 1), alpha=alpha, align_corners=True, p=1)
        params = aug.forward_parameters(image.shape)
        assert params["noise"].shape == (1, 2, 5, 7)
        params["noise"] = torch.ones(1, 2, 5, 7, device=device, dtype=dtype)
        expected = image[..., [1, 2, 3, 4, 5, 6, 6]] if axis == 0 else image[..., [1, 2, 3, 4, 4], :]
        self.assert_close(aug(image, params=params), expected)

    @pytest.mark.parametrize("kernel_size", [(3, 1), (1, 3)])
    def test_convention_elastic_kernel_and_sigma_yx(self, device, dtype, kernel_size):
        image = torch.arange(5, device=device, dtype=dtype).repeat(5, 1).reshape(1, 1, 5, 5)
        # sigma_y gives weights (1/4, 1/2, 1/4); sigma_x gives (1/6, 2/3, 1/6).
        sigma = (math.sqrt(1 / (2 * math.log(2))), math.sqrt(1 / (2 * math.log(4))))
        aug = K.RandomElasticTransform(kernel_size=kernel_size, sigma=sigma, alpha=(0.5, 0), align_corners=True, p=1)
        params = aug.forward_parameters(image.shape)
        params["noise"] = torch.zeros(1, 2, 5, 5, device=device, dtype=dtype)
        params["noise"][0, 0, 2, 2] = 1
        expected = image.clone()
        if kernel_size == (3, 1):
            expected[0, 0, 1:4, 2] += image.new_tensor([0.25, 0.5, 0.25])
        else:
            expected[0, 0, 2, 1:4] += image.new_tensor([1 / 6, 2 / 3, 1 / 6])
        self.assert_close(aug(image, params=params), expected)

    def test_convention_fisheye_normalized_radial_sampling(self, device, dtype):
        image = torch.arange(25, device=device, dtype=dtype).reshape(1, 1, 5, 5)
        aug = K.RandomFisheye(torch.tensor([0.5, 0.5]), torch.tensor([0.0, 0.0]), torch.tensor([2.0, 2.0]), p=1)
        # q=(x,y)*(1+(x-.5)^2+y^2), mapped to pixel coordinates by 2*(q+1).
        # Bilinear interpolation with zero outside [0,4] gives this literal;
        # e.g. output (row=3,col=2) reads row=3.5,col=2, averaging 17 and 22.
        expected = image.new_tensor(
            [[0, 0, 0, 0, 0], [0, 0, 4.5, 7, 0], [0, 10, 12, 13, 7], [0, 11.25, 19.5, 19.5, 0], [0, 0, 0, 0, 0]]
        ).reshape_as(image)
        self.assert_close(aug(image), expected)

    @pytest.mark.device_agnostic
    def test_convention_tps_control_points_and_noise_sample_on_cpu(self):
        aug = K.RandomThinPlateSpline(scale=0.2, same_on_batch=True, p=1)
        params = aug.forward_parameters((2, 1, 5, 7))
        expected = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1], [0, 0]], dtype=torch.float32)
        assert params["src"].device.type == "cpu"
        assert params["src"].dtype == torch.float32
        assert params["dst"].device.type == "cpu"
        assert params["dst"].dtype == torch.float32
        self.assert_close(params["src"], expected.expand(2, 5, 2))
        assert ((params["dst"] - params["src"]).abs() <= 0.201).all()
        self.assert_close(params["dst"][0], params["dst"][1], atol=0, rtol=0)
        zero = K.RandomThinPlateSpline(scale=0, p=1).forward_parameters((2, 1, 5, 7))
        assert zero["src"].device.type == "cpu"
        assert zero["src"].dtype == torch.float32
        self.assert_close(zero["src"], zero["dst"], atol=0, rtol=0)
        assert aug.flags["padding_mode"] == SamplePadding.ZEROS
        assert aug.flags["align_corners"] is False

    @pytest.mark.parametrize("align_corners", [False, True])
    def test_convention_thin_plate_spline_identity_grid_3928(self, device, dtype, align_corners):
        image = torch.ones(1, 1, 3, 3, device=device, dtype=dtype)
        aug = K.RandomThinPlateSpline(scale=0, align_corners=align_corners, p=1)
        self.assert_close(aug(image), image)

    @pytest.mark.parametrize("kind", ["elastic", "fisheye", "tps"])
    def test_convention_dense_shape_and_matrix_interface(self, device, dtype, kind):
        aug = {
            "elastic": lambda: K.RandomElasticTransform(kernel_size=(3, 3), p=1),
            "fisheye": lambda: K.RandomFisheye(
                torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]), p=1
            ),
            "tps": lambda: K.RandomThinPlateSpline(scale=0, p=1),
        }[kind]()
        image = torch.ones(2, 1, 5, 7, device=device, dtype=dtype)
        output = aug(image)
        assert output.shape == image.shape
        if kind == "tps":
            assert torch.isfinite(output).all()
        assert not hasattr(aug, "transform_matrix")
        assert not hasattr(aug, "inverse")
