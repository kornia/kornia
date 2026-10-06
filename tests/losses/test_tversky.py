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


class TestTverskyLoss(BaseTester):
    @pytest.mark.parametrize(
        "size,ignore_index,ignored_rows",
        [(256, None, 0), (256, -100, 0), (512, -100, 8)],
    )
    def test_large_image_half_precision(self, device, dtype, size, ignore_index, ignored_rows):
        logits = torch.full((1, 2, size, size), -2.0, device=device, dtype=dtype)
        logits[:, 0] = 2.0
        logits.requires_grad_()
        labels = torch.zeros((1, size, size), device=device, dtype=torch.int64)
        if ignored_rows:
            labels[:, :ignored_rows] = ignore_index

        # Use a reference dtype that can represent the spatial sums.
        reference_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        expected = kornia.losses.tversky_loss(
            logits.detach().to(reference_dtype),
            labels,
            alpha=0.5,
            beta=0.5,
            ignore_index=ignore_index,
        )
        actual = kornia.losses.tversky_loss(
            logits,
            labels,
            alpha=0.5,
            beta=0.5,
            ignore_index=ignore_index,
        )

        assert actual.dtype == dtype
        assert actual.device == logits.device
        assert torch.isfinite(actual)
        self.assert_close(actual, expected.to(dtype))

        actual.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        if ignored_rows:
            assert torch.count_nonzero(logits.grad[:, :, :ignored_rows]) == 0

    def test_small_loss_ratio_in_float32(self, device, dtype):
        # Half of the pixels have a logit gap of 20 and half a gap of 6, so the loss is about 1e-3 to 2e-3.
        # A ratio formed in bfloat16 has a step of 2**-8 near 1 and returns 0 or 2**-8 instead.
        logits = torch.zeros(1, 2, 128, 128, device=device, dtype=dtype)
        logits[:, 0, :, ::2] = 20.0
        logits[:, 0, :, 1::2] = 6.0
        labels = torch.zeros(1, 128, 128, device=device, dtype=torch.int64)

        p_true = logits.softmax(dim=1)[:, 0].cpu().double()
        expected = 1.0 - p_true.sum() / (p_true.numel() + 1e-8)
        actual = kornia.losses.tversky_loss(logits, labels, alpha=0.5, beta=0.5)

        assert actual.dtype == dtype
        self.assert_close(actual.cpu().double(), expected, rtol=1e-2, atol=1e-6)

    def test_smoke(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        criterion = kornia.losses.TverskyLoss(alpha=0.5, beta=0.5)
        assert criterion(logits, labels) is not None

    def test_exception(self):
        criterion = kornia.losses.TverskyLoss(alpha=0.5, beta=0.5)

        with pytest.raises(TypeError) as errinfo:
            criterion("not a tensor", torch.rand(1))
        assert "pred type is not a torch.Tensor. Got" in str(errinfo)

        with pytest.raises(ValueError) as errinfo:
            criterion(torch.rand(1), torch.rand(1))
        assert "Invalid pred shape, we expect BxNxHxW. Got:" in str(errinfo)

        with pytest.raises(ValueError) as errinfo:
            criterion(torch.rand(1, 1, 1, 1), torch.rand(1, 1, 1, 2))
        assert "pred and target shapes must be the same. Got:" in str(errinfo)

        with pytest.raises(ValueError) as errinfo:
            criterion(torch.rand(1, 1, 1, 1), torch.rand(1, 1, 1, 1, device="meta"))
        assert "pred and target must be in the same device. Got:" in str(errinfo)

        # The target batch has to match the prediction batch, as for focal_loss (#5544).
        with pytest.raises(ValueError) as errinfo:
            criterion(torch.rand(2, 3, 4, 6), torch.randint(0, 3, (1, 4, 6)))
        assert "pred and target shapes must be the same. Got:" in str(errinfo)

        with pytest.raises(ValueError) as errinfo:
            criterion(torch.rand(1, 3, 4, 6), torch.randint(0, 3, (2, 4, 6)))
        assert "pred and target shapes must be the same. Got:" in str(errinfo)

    @pytest.mark.parametrize("ignore_index", [-100, None])
    def test_all_zeros(self, device, dtype, ignore_index):
        num_classes = 3
        logits = torch.zeros(2, num_classes, 1, 2, device=device, dtype=dtype)
        logits[:, 0] = 10.0
        logits[:, 1] = 1.0
        logits[:, 2] = 1.0
        labels = torch.zeros(2, 1, 2, device=device, dtype=torch.int64)

        criterion = kornia.losses.TverskyLoss(alpha=0.5, beta=0.5, ignore_index=ignore_index)
        loss = criterion(logits, labels)
        self.assert_close(loss, torch.zeros_like(loss), atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("ignore_index", [-100, 255])
    def test_ignore_index(self, device, dtype, ignore_index):
        num_classes = 2

        logits = torch.zeros(2, num_classes, 1, 4, device=device, dtype=dtype)
        logits[:, 0, :, 0] = 100.0
        logits[:, 1, :, 1:] = 100.0
        labels = torch.zeros(2, 1, 4, device=device, dtype=torch.int64)

        labels[..., 2:] = ignore_index
        expected_loss = torch.tensor([1.0 / 2.0], device=device, dtype=dtype).squeeze()
        criterion = kornia.losses.TverskyLoss(alpha=0.5, beta=0.5, ignore_index=ignore_index)
        loss = criterion(logits, labels)
        self.assert_close(loss, expected_loss, rtol=1e-3, atol=1e-3)

    def test_gradcheck(self, device, dtype):
        num_classes = 3
        alpha, beta = 0.5, 0.5  # for tversky loss
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=torch.float64)
        labels = torch.randint(0, num_classes, (2, 3, 2), device=device)
        ignore = torch.rand(2, 3, 2, device=device) > 0.8
        labels[ignore] = -100

        self.gradcheck(
            kornia.losses.tversky_loss, (logits, labels, alpha, beta), dtypes=[torch.float64, torch.int64, None, None]
        )

    def test_dynamo(self, device, dtype, torch_optimizer):
        num_classes = 3
        params = (0.5, 0.05)
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.tversky_loss
        op_optimized = torch_optimizer(op)

        actual = op_optimized(logits, labels, *params)
        expected = op(logits, labels, *params)
        self.assert_close(actual, expected)

    def test_module(self, device, dtype):
        num_classes = 3
        params = (0.5, 0.5)
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.tversky_loss
        op_module = kornia.losses.TverskyLoss(*params)

        actual = op_module(logits, labels)
        expected = op(logits, labels, *params)
        self.assert_close(actual, expected)
