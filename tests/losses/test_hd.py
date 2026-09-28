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

import kornia

from testing.base import BaseTester


class TestHausdorffLoss(BaseTester):
    @pytest.mark.parametrize("reduction", ["mean", "none", "sum"])
    @pytest.mark.parametrize(
        "hd,shape", [[kornia.losses.HausdorffERLoss, (10, 10)], [kornia.losses.HausdorffERLoss3D, (10, 10, 10)]]
    )
    def test_smoke_none(self, hd, shape, reduction, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, *shape, dtype=dtype, device=device)
        labels = (torch.rand(2, 1, *shape, dtype=dtype, device=device) * (num_classes - 1)).long()
        loss = hd(reduction=reduction)

        loss(logits, labels)

    def test_exception_2d(self):
        with pytest.raises(ValueError) as errinf:
            kornia.losses.HausdorffERLoss()((torch.rand(1, 2, 1) > 0.5) * 1, (torch.rand(1, 1, 1, 2) > 0.5) * 1)
        assert "Only 2D images supported. Got " in str(errinf)

        with pytest.raises(ValueError) as errinf:
            kornia.losses.HausdorffERLoss()(
                (torch.rand(1, 2, 1, 1) > 0.5) * 1, torch.tensor([[[[1]]]], dtype=torch.float32)
            )
        assert "Expect long type target value in range" in str(errinf)

        with pytest.raises(ValueError) as errinf:
            kornia.losses.HausdorffERLoss()((torch.rand(1, 2, 1, 1) > 0.5) * 1, (torch.rand(1, 1, 1, 2) > 0.5) * 1)
        assert "Prediction and target need to be of same size, and target should not be one-hot." in str(errinf)

    def test_exception_3d(self):
        with pytest.raises(ValueError) as errinf:
            kornia.losses.HausdorffERLoss3D()((torch.rand(1, 2, 1) > 0.5) * 1, (torch.rand(1, 1, 1, 2) > 0.5) * 1)
        assert "Only 3D images supported. Got " in str(errinf)

        with pytest.raises(ValueError) as errinf:
            kornia.losses.HausdorffERLoss3D()(
                (torch.rand(1, 2, 1, 1, 1) > 0.5) * 1, torch.tensor([[[[[5]]]]], dtype=torch.float32)
            )
        assert "Invalid target value" in str(errinf)

    def test_numeric(self, device, dtype):
        if dtype == torch.float64:
            pytest.xfail("Sometimes failing on float64")
        num_classes = 3
        shape = (50, 50)
        hd = kornia.losses.HausdorffERLoss
        logits = torch.rand(2, num_classes, *shape, dtype=dtype, device=device)
        labels = (torch.rand(2, 1, *shape, dtype=dtype, device=device) * (num_classes - 1)).long()
        loss = hd(k=10)

        expected = torch.tensor(0.025, device=device, dtype=dtype)

        actual = loss(logits, labels)
        self.assert_close(actual, expected, rtol=0.005, atol=0.005)

    def test_numeric_3d(self, device, dtype):
        num_classes = 3
        shape = (50, 50, 50)
        hd = kornia.losses.HausdorffERLoss3D
        logits = torch.rand(2, num_classes, *shape, dtype=dtype, device=device)
        labels = (torch.rand(2, 1, *shape, dtype=dtype, device=device) * (num_classes - 1)).long()
        loss = hd(k=10)

        expected = torch.tensor(0.011, device=device, dtype=dtype)
        actual = loss(logits, labels)
        self.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "hd,shape", [[kornia.losses.HausdorffERLoss, (8, 8)], [kornia.losses.HausdorffERLoss3D, (6, 6, 6)]]
    )
    def test_gradient_at_the_threshold_tie(self, hd, shape, device, dtype):
        # Quantized predictions land the dilation exactly on the 0.5 threshold, where the derivative of
        # the soft threshold is version-dependent for clamp (#4229). Pin it against the reference: zero
        # the negatives through a boolean index, whose gradient at the tie is 1.
        if dtype != torch.float32:
            pytest.skip("the 0.2 cross kernel sums to exactly 0.5 in float32 only, so no other dtype reaches the tie")
        generator = torch.Generator().manual_seed(0)
        pred = (torch.randint(0, 3, (1, 2, *shape), generator=generator) * 0.5).to(device, dtype)
        target = torch.randint(0, 2, (1, 1, *shape), generator=generator).to(device)
        loss = hd(k=3)

        def reference_erosion(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            bound = (pred - target) ** 2
            kernel = torch.as_tensor(loss.kernel, device=pred.device, dtype=pred.dtype)
            padding = (kernel.size(-1) - 1) // 2
            eroded = torch.zeros_like(bound)
            ties = 0
            for k in range(loss.k):
                dilation = loss.conv(bound, weight=kernel, padding=padding, groups=1)
                ties += int((dilation == 0.5).sum())
                erosion = dilation - 0.5
                erosion[erosion < 0] = 0
                erosion_max = loss.max_pool(erosion)
                erosion_min = -loss.max_pool(-erosion)
                _range = erosion_max - erosion_min
                _to_norm = _range != 0
                erosion = torch.where(_to_norm, (erosion - erosion_min) / torch.where(_to_norm, _range, 1.0), erosion)
                eroded = eroded + erosion * (k + 1) ** loss.alpha
                bound = erosion
            assert ties > 0, "the fixture no longer reaches the threshold tie"
            return eroded

        pred_actual = pred.clone().requires_grad_()
        loss(pred_actual, target).backward()

        pred_expected = pred.clone().requires_grad_()
        per_class = [
            reference_erosion(pred_expected[:, i : i + 1], (target == i).to(dtype)) for i in range(pred.size(1))
        ]
        torch.stack(per_class).mean().backward()

        self.assert_close(pred_actual.grad, pred_expected.grad, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize(
        "hd,shape", [[kornia.losses.HausdorffERLoss, (5, 5)], [kornia.losses.HausdorffERLoss3D, (5, 5, 5)]]
    )
    def test_gradcheck(self, hd, shape, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, *shape, device=device)
        labels = (torch.rand(2, 1, *shape, device=device) * (num_classes - 1)).long()
        loss = hd(k=2)

        self.gradcheck(loss, (logits, labels), dtypes=[torch.float64, torch.int64])
