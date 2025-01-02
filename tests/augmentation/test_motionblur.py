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

from kornia.augmentation import RandomMotionBlur, RandomMotionBlur3D
from kornia.filters import motion_blur, motion_blur3d

from testing.base import BaseTester


class TestRandomMotionBlur(BaseTester):
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self, device):
        f = RandomMotionBlur(kernel_size=(3, 5), angle=(10, 30), direction=0.5)
        repr = (
            "RandomMotionBlur(kernel_size=(3, 5), angle=tensor([10., 30.]), direction=tensor([-0.5000, 0.5000]), "
            "border_type='constant', p=0.5, p_batch=1.0, same_on_batch=False, return_transform=None)"
        )
        assert str(f) == repr

    @pytest.mark.parametrize("kernel_size", [(3, 5), (7, 21)])
    @pytest.mark.parametrize("same_on_batch", [True, False])
    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_random_motion_blur(self, kernel_size, same_on_batch, p, device, dtype):
        f = RandomMotionBlur(kernel_size=kernel_size, angle=(10, 30), direction=0.5, same_on_batch=same_on_batch, p=p)
        torch.manual_seed(0)
        batch_size = 2
        input = torch.randn(1, 3, 5, 6, device=device, dtype=dtype).repeat(batch_size, 1, 1, 1)

        output = f(input)

        if same_on_batch:
            self.assert_close(output[0], output[1], rtol=1e-4, atol=1e-4)
        elif p == 0:
            self.assert_close(output, input, rtol=1e-4, atol=1e-4)
        else:
            assert not torch.allclose(output[0], output[1], rtol=1e-4, atol=1e-4)

        assert output.shape == torch.Size([batch_size, 3, 5, 6])

    @pytest.mark.parametrize("input_shape", [(1, 1, 5, 5), (2, 1, 5, 5)])
    def test_against_functional(self, input_shape):
        input = torch.randn(*input_shape)

        f = RandomMotionBlur(kernel_size=(3, 5), angle=(10, 30), direction=0.5, p=1.0)
        output = f(input)

        expected = motion_blur(
            input,
            f._params["ksize_factor"].unique().item(),
            f._params["angle_factor"],
            f._params["direction_factor"],
            f.flags["border_type"].name.lower(),
        )

        self.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.slow
    def test_gradcheck(self, device):
        torch.manual_seed(0)  # for random reproductibility
        inp = torch.rand((1, 3, 11, 7), device=device, dtype=torch.float64)
        # TODO: Gradcheck for param random gen failed. Suspect get_motion_kernel2d issue.
        params = {
            "batch_prob": torch.tensor([True]),
            "ksize_factor": torch.tensor([31]),
            "angle_factor": torch.tensor([30.0]),
            "direction_factor": torch.tensor([-0.5]),
            "border_type": torch.tensor([0]),
            "idx": torch.tensor([0]),
        }
        self.gradcheck(
            RandomMotionBlur(kernel_size=3, angle=(10, 30), direction=(-0.5, 0.5), p=1.0),
            (inp, params),
            fast_mode=False,
        )


class TestRandomMotionBlur3D(BaseTester):
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self, device, dtype):
        f = RandomMotionBlur3D(kernel_size=(3, 5), angle=(10, 30), direction=0.5)
        repr = (
            "RandomMotionBlur3D(kernel_size=(3, 5), angle=tensor([[10., 30.],"
            "\n        [10., 30.],\n        [10., 30.]]), direction=tensor([-0.5000, 0.5000]), "
            "border_type='constant', p=0.5, p_batch=1.0, same_on_batch=False, return_transform=None)"
        )
        assert str(f) == repr

    @pytest.mark.parametrize("same_on_batch", [True, False])
    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_random_motion_blur(self, same_on_batch, p, device, dtype):
        f = RandomMotionBlur3D(kernel_size=(3, 5), angle=(10, 30), direction=0.5, same_on_batch=same_on_batch, p=p)
        batch_size = 2
        input = torch.randn(1, 3, 5, 6, 7, device=device, dtype=dtype).repeat(batch_size, 1, 1, 1, 1)

        output = f(input)

        if same_on_batch:
            self.assert_close(output[0], output[1], rtol=1e-4, atol=1e-4)
        elif p == 0:
            self.assert_close(output, input, rtol=1e-4, atol=1e-4)
        else:
            assert not torch.allclose(output[0], output[1], rtol=1e-4, atol=1e-4)

        assert output.shape == torch.Size([batch_size, 3, 5, 6, 7])

    @pytest.mark.parametrize("input_shape", [(1, 1, 5, 6, 7), (2, 1, 5, 6, 7)])
    def test_against_functional(self, input_shape):
        input = torch.randn(*input_shape)

        f = RandomMotionBlur3D(kernel_size=(3, 5), angle=(10, 30), direction=0.5, p=1.0)
        output = f(input)

        expected = motion_blur3d(
            input,
            f._params["ksize_factor"].unique().item(),
            f._params["angle_factor"],
            f._params["direction_factor"],
            f.flags["border_type"].name.lower(),
        )

        self.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.slow
    def test_gradcheck(self, device):
        torch.manual_seed(0)  # for random reproductibility
        inp = torch.rand((1, 3, 6, 7), device=device, dtype=torch.float64)
        params = {
            "batch_prob": torch.tensor([True]),
            "ksize_factor": torch.tensor([31]),
            "angle_factor": torch.tensor([[30.0, 30.0, 30.0]]),
            "direction_factor": torch.tensor([-0.5]),
            "border_type": torch.tensor([0]),
            "idx": torch.tensor([0]),
        }
        self.gradcheck(
            RandomMotionBlur3D(kernel_size=3, angle=(10, 30), direction=(-0.5, 0.5), p=1.0),
            (inp, params),
            fast_mode=False,
        )
