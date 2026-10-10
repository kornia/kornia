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

from testing.base import BaseTester


class TestDepthSmoothnessLoss(BaseTester):
    @pytest.mark.parametrize("data_shape", [(1, 1, 10, 16), (2, 4, 8, 15)])
    def test_smoke(self, device, dtype, data_shape):
        image = torch.rand(data_shape, device=device, dtype=dtype)
        depth = torch.rand(data_shape, device=device, dtype=dtype)

        criterion = kornia.losses.InverseDepthSmoothnessLoss()
        assert criterion(depth, image) is not None

    def test_exception(self):
        with pytest.raises(TypeError) as errinf:
            kornia.losses.InverseDepthSmoothnessLoss()(1, 1)
        assert "Input idepth type is not a torch.Tensor. Got" in str(errinf)

        with pytest.raises(TypeError) as errinf:
            kornia.losses.InverseDepthSmoothnessLoss()(torch.rand(1), 1)
        assert "Input image type is not a torch.Tensor. Got" in str(errinf)

        with pytest.raises(ValueError) as errinf:
            kornia.losses.InverseDepthSmoothnessLoss()(torch.rand(1, 1), torch.rand(1, 1, 1, 1))
        assert "Invalid idepth shape, we expect BxCxHxW. Got" in str(errinf)

        with pytest.raises(ValueError) as errinf:
            kornia.losses.InverseDepthSmoothnessLoss()(torch.rand(1, 1, 1, 1), torch.rand(1, 1, 1))
        assert "Invalid image shape, we expect BxCxHxW. Got:" in str(errinf)

        with pytest.raises(ValueError) as errinf:
            kornia.losses.InverseDepthSmoothnessLoss()(torch.rand(1, 1, 1, 1), torch.rand(1, 1, 1, 2))
        assert "idepth and image shapes must be the same. Got" in str(errinf)

        with pytest.raises(ValueError) as errinf:
            kornia.losses.InverseDepthSmoothnessLoss()(torch.rand(1, 1, 1, 1), torch.rand(1, 1, 1, 1, device="meta"))
        assert "idepth and image must be in the same device. Got:" in str(errinf)

        with pytest.raises(ValueError) as errinf:
            kornia.losses.InverseDepthSmoothnessLoss()(
                torch.rand(1, 1, 1, 1, dtype=torch.float32), torch.rand(1, 1, 1, 1, dtype=torch.float64)
            )
        assert "idepth and image must be in the same dtype. Got:" in str(errinf)

    def test_dynamo(self, device, dtype, torch_optimizer):
        image = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        depth = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        op = kornia.losses.inverse_depth_smoothness_loss
        op_optimized = torch_optimizer(op)

        self.assert_close(op(image, depth), op_optimized(image, depth))

    def test_module(self, device, dtype):
        image = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        depth = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        op = kornia.losses.inverse_depth_smoothness_loss
        op_module = kornia.losses.InverseDepthSmoothnessLoss()

        self.assert_close(op(image, depth), op_module(image, depth))

    def test_gradcheck(self, device, dtype):
        image = torch.rand(1, 2, 3, 4, device=device, dtype=torch.float64)
        depth = torch.rand(1, 2, 3, 4, device=device, dtype=torch.float64)
        self.gradcheck(kornia.losses.inverse_depth_smoothness_loss, (depth, image))


class TestConventionsInverseDepthSmoothness(BaseTester):
    @staticmethod
    def _steps(device, dtype):
        # H = 4, W = 6. The image steps once, between columns 1 and 2, by 0.5, 1.0 and 1.5 in its three channels
        # (channel mean 1.0). The inverse depth steps by 1 on that image edge and by 0.5 between columns 3 and 4, where
        # the image is flat, and rises by 0.25 per row; sample 1 doubles sample 0.
        rows = torch.arange(4.0).view(4, 1)
        cols = torch.arange(6.0).view(1, 6)
        idepth = (1.0 * (cols >= 2) + 0.5 * (cols >= 4) + 0.25 * rows).expand(1, 1, 4, 6)
        idepth = torch.cat([idepth, 2 * idepth])
        image = (torch.tensor([0.5, 1.0, 1.5]).view(3, 1, 1) * (cols >= 2)).expand(2, 3, 4, 6)
        return idepth.to(device, dtype), image.to(device, dtype)

    def test_convention_inverse_depth_smoothness_weights_by_the_channel_mean_image_gradient(self, device, dtype):
        # loss = mean(|dx d| exp(-mean_c |dx I|)) + mean(|dy d| exp(-mean_c |dy I|)) with forward differences, the
        # weight of each difference taken from the image difference at the same place, each mean over its own count and
        # over the batch. Per row of sample 0 the five x-differences cost exp(-1) on the image edge and 0.5 on the flat
        # step, and every y-difference 0.25 under a weight of 1: (exp(-1) + 0.5) / 5 + 0.25. Sample 1 costs twice that:
        # the inverse depth is not normalised.
        idepth, image = self._steps(device, dtype)
        expected = torch.tensor(1.5 * ((math.exp(-1.0) + 0.5) / 5 + 0.25), device=device, dtype=dtype)
        self.assert_close(kornia.losses.inverse_depth_smoothness_loss(idepth, image), expected)
        self.assert_close(kornia.losses.InverseDepthSmoothnessLoss()(idepth, image), expected)

    def test_convention_inverse_depth_smoothness_penalises_the_first_argument(self, device, dtype):
        # The first argument (the inverse depth) is penalised and the second (the image) only sets the weights: a
        # constant inverse depth costs 0 whatever the image, and a constant image leaves the plain mean |grad d|. The
        # channel counts (N, 1, H, W) and (N, 3, H, W) are not validated, so swapped arguments also return a value, and
        # neither is the batch size: a batch of one broadcasts against the other argument.
        idepth, image = self._steps(device, dtype)
        loss = kornia.losses.inverse_depth_smoothness_loss
        self.assert_close(loss(idepth[:1], image), loss(idepth[:1].expand_as(idepth), image))
        zero = torch.zeros((), device=device, dtype=dtype)
        self.assert_close(kornia.losses.inverse_depth_smoothness_loss(torch.ones_like(idepth), image), zero)
        flat = torch.ones_like(image)
        expected = torch.tensor(1.5 * ((1.0 + 0.5) / 5 + 0.25), device=device, dtype=dtype)
        self.assert_close(kornia.losses.inverse_depth_smoothness_loss(idepth, flat), expected)
        swapped = kornia.losses.inverse_depth_smoothness_loss(image, idepth)
        assert torch.isfinite(swapped)
        assert (swapped - kornia.losses.inverse_depth_smoothness_loss(idepth, image)).abs() > 0.1
