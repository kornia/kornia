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

import torch

from kornia.filters import get_gaussian_erf_kernel1d

from testing.base import BaseTester


class TestGaussianErfPrecision(BaseTester):
    def test_convention_integrated_tails(self, device, dtype):
        # Nine unit-wide bins centered on integers -4,...,4 of N(0,1), divided by their sum.
        # Offline scalar reference: (erf((x+.5)/sqrt(2))-erf((x-.5)/sqrt(2)))/2.
        expected = torch.tensor(
            [
                [
                    0.00022923297362889435,
                    0.005977076863047462,
                    0.06059794863488053,
                    0.2417319808759537,
                    0.3829275213049788,
                    0.2417319808759537,
                    0.06059794863488053,
                    0.005977076863047462,
                    0.00022923297362889435,
                ]
            ],
            device=device,
            dtype=dtype,
        )
        actual = get_gaussian_erf_kernel1d(9, 1.0, device=device, dtype=dtype)
        self.assert_close(actual, expected, atol=3e-8, rtol=2 * torch.finfo(dtype).eps + 1e-7)

    def test_convention_nonzero_sigma_gradient(self, device, dtype):
        # Differentiate the normalized, integrated mass of nine bins at x=-4,...,4,
        # weighted by x**2. Each unnormalized bin derivative is
        # ((x-.5)*exp(-(x-.5)**2/(2*sigma**2)) - (x+.5)*exp(-(x+.5)**2/(2*sigma**2)))
        # / (sqrt(2*pi)*sigma**2); the quotient rule gives these offline scalar values.
        sigma = torch.tensor([[0.5], [1.0], [2.0]], device=device, dtype=dtype, requires_grad=True)
        actual = get_gaussian_erf_kernel1d(9, sigma)
        coefficients = torch.tensor([[16.0, 9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 9.0, 16.0]], device=device, dtype=dtype)
        expected = torch.tensor(
            [[1.1275781146482198], [1.996560378787884], [2.1301847507189673]], device=device, dtype=dtype
        )
        self.assert_close(torch.autograd.grad((actual * coefficients).sum(), sigma)[0], expected)

    def test_gradcheck(self, device):
        self.gradcheck(get_gaussian_erf_kernel1d, (5, torch.tensor([[0.5]], device=device, dtype=torch.float64)))

    def test_dynamo(self, device, dtype, torch_optimizer):
        expected = get_gaussian_erf_kernel1d(9, 1.0, device=device, dtype=dtype)
        actual = torch_optimizer(get_gaussian_erf_kernel1d)(9, 1.0, device=device, dtype=dtype)
        self.assert_close(actual, expected)
