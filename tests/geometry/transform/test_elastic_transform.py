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

from kornia.geometry.transform import elastic_transform2d

from testing.base import BaseTester


class TestElasticTransform(BaseTester):
    def test_smoke(self, device, dtype):
        image = torch.rand(1, 4, 5, 5, device=device, dtype=dtype)
        noise = torch.rand(1, 2, 5, 5, device=device, dtype=dtype)
        assert elastic_transform2d(image, noise) is not None

    @pytest.mark.parametrize("batch, channels, height, width", [(1, 3, 3, 4), (2, 2, 2, 4), (1, 5, 4, 1)])
    def test_cardinality(self, device, dtype, batch, channels, height, width):
        shape = batch, channels, height, width
        img = torch.ones(shape, device=device, dtype=dtype)
        noise = torch.ones((batch, 2, height, width), device=device, dtype=dtype)
        assert elastic_transform2d(img, noise).shape == shape

    def test_exception(self, device, dtype):
        from kornia.core.exceptions import ShapeError, TypeCheckError

        ex = torch.ones(1, device=device, dtype=dtype)
        with pytest.raises(TypeCheckError) as errinfo:
            elastic_transform2d([0.0], ex)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        with pytest.raises(TypeCheckError) as errinfo:
            elastic_transform2d(ex, 1)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        with pytest.raises(ShapeError) as errinfo:
            img = torch.ones(1, 1, 1, device=device, dtype=dtype)
            noise = torch.ones(1, 2, 1, 1, device=device, dtype=dtype)
            elastic_transform2d(img, noise)
        assert "Shape dimension mismatch" in str(errinfo.value)

        with pytest.raises(ShapeError) as errinfo:
            img = torch.ones(1, 1, 1, 1, device=device, dtype=dtype)
            noise = torch.ones(2, 1, 1, device=device, dtype=dtype)
            elastic_transform2d(img, noise)
        assert "Shape dimension mismatch" in str(errinfo.value)

        with pytest.raises(RuntimeError) as errinfo:
            img = torch.ones(1, 1, 1, 1, device=device, dtype=dtype)
            noise = torch.ones(1, 3, 1, 1, device=device, dtype=dtype)
            elastic_transform2d(img, noise)
        assert "The size of tensor a (2) must match the size of tensor b (3)" in str(errinfo.value)

    @pytest.mark.parametrize(
        "kernel_size, sigma, alpha",
        [
            [(3, 3), (4.0, 4.0), (32.0, 32.0)],
            [(5, 3), (4.0, 8.0), (16.0, 32.0)],
            [(5, 5), torch.tensor([2.0, 8.0]), torch.tensor([16.0, 64.0])],
        ],
    )
    def test_valid_paramters(self, device, dtype, kernel_size, sigma, alpha):
        image = torch.rand(1, 4, 5, 5, device=device, dtype=dtype)
        noise = torch.rand(1, 2, 5, 5, device=device, dtype=dtype)
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.to(device, dtype)
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.to(device, dtype)
        assert elastic_transform2d(image, noise, kernel_size, sigma, alpha) is not None

    def test_values(self, device, dtype):
        image = torch.tensor(
            [[[[0.0018, 0.7521, 0.7550], [0.2053, 0.4249, 0.1369], [0.1027, 0.3992, 0.8773]]]],
            device=device,
            dtype=dtype,
        )

        noise = torch.ones(1, 2, 3, 3, device=device, dtype=dtype)

        expected = torch.tensor(
            [[[[0.0005, 0.3795, 0.1905], [0.1034, 0.4235, 0.0702], [0.0259, 0.2007, 0.2193]]]],
            device=device,
            dtype=dtype,
        )

        actual = elastic_transform2d(image, noise)
        self.assert_close(actual, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("requires_grad", [True, False])
    def test_gradcheck(self, device, dtype, requires_grad):
        image = torch.rand(1, 1, 3, 3, device=device, dtype=torch.float64, requires_grad=requires_grad)
        noise = torch.rand(1, 2, 3, 3, device=device, dtype=torch.float64, requires_grad=not requires_grad)
        assert self.gradcheck(
            elastic_transform2d, (image, noise), raise_exception=True, fast_mode=True, nondet_tol=1e-4
        )

    def test_convention_noise_channel0_is_x_positive_shifts_left(self, device, dtype):
        # noise channel 0 is the x-sampling-offset channel; a positive constant value there
        # shifts image content LEFT (not right, and not channel 1/y). test_values uses a
        # symmetric (both-channels-equal) noise tensor and so cannot discriminate this.
        image = torch.zeros(1, 1, 5, 5, device=device, dtype=dtype)
        image[0, 0, 2, 2] = 1.0
        noise = torch.zeros(1, 2, 5, 5, device=device, dtype=dtype)
        noise[0, 0] = 0.5

        # Snippet used to generate expected:
        # img = torch.zeros(1, 1, 5, 5); img[0, 0, 2, 2] = 1.0
        # noise = torch.zeros(1, 2, 5, 5); noise[0, 0] = 0.5
        # out = elastic_transform2d(img, noise, kernel_size=(3, 3), sigma=(1., 1.), alpha=(1., 1.))
        # (out[0, 0] > 0.1).nonzero() -> [[2, 1]]
        out = elastic_transform2d(image, noise, kernel_size=(3, 3), sigma=(1.0, 1.0), alpha=(1.0, 1.0))
        assert (out[0, 0] > 0.1).nonzero().tolist() == [[2, 1]]

    def test_convention_kernel_size_sigma_yx_alpha_xy_order(self, device, dtype):
        # kernel_size and sigma are (y, x) order (matching the docstring), but alpha is
        # genuinely (x, y) order in the executed code: alpha[0] always scales the
        # x-displacement and alpha[1] the y-displacement -- contradicting the "in the y and x
        # directions, respectively" docstring text, which only holds for kernel_size/sigma.
        image = torch.zeros(1, 1, 9, 9, device=device, dtype=dtype)
        image[0, 0, 4, 4] = 1.0
        noise = torch.zeros(1, 2, 9, 9, device=device, dtype=dtype)
        noise[0, 0] = 0.2

        out_a = elastic_transform2d(image, noise, kernel_size=(3, 3), sigma=(1.0, 1.0), alpha=(2.0, 0.5))
        out_b = elastic_transform2d(image, noise, kernel_size=(3, 3), sigma=(1.0, 1.0), alpha=(0.5, 2.0))

        # Snippet used to generate expected (requires only this module):
        # image = torch.zeros(1, 1, 9, 9); image[0, 0, 4, 4] = 1.0
        # noise = torch.zeros(1, 2, 9, 9); noise[0, 0] = 0.2
        # out_a = elastic_transform2d(image, noise, kernel_size=(3, 3), sigma=(1.0, 1.0), alpha=(2.0, 0.5))
        # out_b = elastic_transform2d(image, noise, kernel_size=(3, 3), sigma=(1.0, 1.0), alpha=(0.5, 2.0))
        # (out_a[0, 0] > 0.05).nonzero().tolist() / (out_b[0, 0] > 0.05).nonzero().tolist()
        # alpha=(2.0, 0.5): larger alpha[0] -> larger x-displacement -> marker moves further left.
        assert (out_a[0, 0] > 0.05).nonzero().tolist() == [[4, 2], [4, 3]]
        # alpha=(0.5, 2.0): smaller alpha[0] -> smaller x-displacement -> marker moves less.
        assert (out_b[0, 0] > 0.05).nonzero().tolist() == [[4, 3], [4, 4]]

        # sigma is genuinely (sigma_y, sigma_x): an impulse in the x-displacement noise channel,
        # blurred with a strongly anisotropic sigma, spreads along ROWS when sigma[0] is large and
        # is confined to a single row when sigma[1] is large instead -- pinning sigma[0] to the y
        # (row) direction, independently of the alpha check above.
        #
        # Pinned as an ORDERING PROPERTY (row-wise spread of the affected region), not as exact
        # threshold-crossing pixel lists: compute the per-row total displaced energy, then the
        # weighted standard deviation of that energy across rows (how far the effect spreads
        # vertically). sigma=(3.0, 0.3) must spread markedly wider across rows than sigma=(0.3, 3.0).
        #
        # Snippet used to generate expected (requires only this module):
        # N, c = 21, 10
        # image = torch.zeros(1, 1, N, N); image[0, 0, :, c] = 1.0
        # noise = torch.zeros(1, 2, N, N); noise[0, 0, c, c] = 1.0
        # kwargs = dict(kernel_size=(9, 9), alpha=(3.0, 0.0))
        # out_y = elastic_transform2d(image, noise, sigma=(3.0, 0.3), **kwargs)
        # out_x = elastic_transform2d(image, noise, sigma=(0.3, 3.0), **kwargs)
        # def row_std(out):
        #     energy = (out[0, 0, 1:-1] - image[0, 0, 1:-1]).abs().sum(dim=1)
        #     rows = torch.arange(energy.shape[0], dtype=energy.dtype)
        #     w = energy / energy.sum()
        #     mean = (w * rows).sum()
        #     return (w * (rows - mean) ** 2).sum().sqrt().item()
        # row_std(out_y), row_std(out_x)  # -> 2.582, 0.144 (ratio ~18x)
        n_sigma, center = 21, 10
        sigma_image = torch.zeros(1, 1, n_sigma, n_sigma, device=device, dtype=dtype)
        sigma_image[0, 0, :, center] = 1.0
        sigma_noise = torch.zeros(1, 2, n_sigma, n_sigma, device=device, dtype=dtype)
        sigma_noise[0, 0, center, center] = 1.0
        sigma_kwargs = {"kernel_size": (9, 9), "alpha": (3.0, 0.0)}

        out_sigma_y_wide = elastic_transform2d(sigma_image, sigma_noise, sigma=(3.0, 0.3), **sigma_kwargs)
        out_sigma_x_wide = elastic_transform2d(sigma_image, sigma_noise, sigma=(0.3, 3.0), **sigma_kwargs)

        def _row_spread_std(out: torch.Tensor) -> torch.Tensor:
            # weighted standard deviation (over interior rows) of the displaced energy,
            # i.e. how far the affected region spreads vertically.
            energy = (out[0, 0, 1:-1] - sigma_image[0, 0, 1:-1]).abs().sum(dim=1)
            rows = torch.arange(energy.shape[0], dtype=energy.dtype, device=energy.device)
            weights = energy / energy.sum()
            mean = (weights * rows).sum()
            return (weights * (rows - mean) ** 2).sum().sqrt()

        spread_y_wide = _row_spread_std(out_sigma_y_wide)
        spread_x_wide = _row_spread_std(out_sigma_x_wide)
        # sigma=(3.0, 0.3) (large sigma_y) spreads markedly wider across rows than
        # sigma=(0.3, 3.0) (large sigma_x) -- a robust ordering property, not an exact pixel list.
        assert spread_y_wide > 5 * spread_x_wide

    def test_convention_padding_mode_affects_border_sampling(self, device, dtype):
        # padding_mode's default ('zeros') genuinely affects boundary sampling, but only once the
        # displacement pushes the (internally clamped-to-[-1,1]) sampling grid all the way to the
        # edge -- no other existing test drives the displacement far enough to reach it.
        image = torch.zeros(1, 1, 5, 5, device=device, dtype=dtype)
        image[0, 0, 2, 4] = 5.0
        noise = torch.zeros(1, 2, 5, 5, device=device, dtype=dtype)
        noise[0, 0] = 1.0
        kwargs = {"kernel_size": (3, 3), "sigma": (1.0, 1.0), "alpha": (4.0, 0.0)}

        out_default = elastic_transform2d(image, noise, **kwargs)
        out_zeros = elastic_transform2d(image, noise, padding_mode="zeros", **kwargs)
        # Snippet used to generate expected (requires only this module):
        # image = torch.zeros(1, 1, 5, 5); image[0, 0, 2, 4] = 5.0
        # noise = torch.zeros(1, 2, 5, 5); noise[0, 0] = 1.0
        # kwargs = dict(kernel_size=(3, 3), sigma=(1.0, 1.0), alpha=(4.0, 0.0))
        # elastic_transform2d(image, noise, **kwargs)[0, 0, 2]                       # 'zeros' row
        # elastic_transform2d(image, noise, padding_mode="border", **kwargs)[0, 0, 2]  # 'border' row
        expected_zeros_row = torch.tensor([2.5, 2.5, 2.5, 2.5, 2.5], device=device, dtype=dtype)
        self.assert_close(out_default[0, 0, 2], expected_zeros_row, rtol=1e-2, atol=1e-2)
        self.assert_close(out_zeros[0, 0, 2], expected_zeros_row, rtol=1e-2, atol=1e-2)

        # MPS 2D grid_sample raises "Unsupported Border padding mode" (torch 2.9.1); 3D supports it.
        if device.type != "mps":
            out_border = elastic_transform2d(image, noise, padding_mode="border", **kwargs)
            expected_border_row = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0], device=device, dtype=dtype)
            self.assert_close(out_border[0, 0, 2], expected_border_row, rtol=1e-2, atol=1e-2)
