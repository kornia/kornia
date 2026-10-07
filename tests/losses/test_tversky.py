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
    def test_per_class_issue_5543(self, device):
        # The issue's CPU-seeded fixture; all classes are present in each sample.
        generator = torch.Generator().manual_seed(1206)
        logits = torch.randn(2, 3, 4, 6, dtype=torch.float64, generator=generator) * 2
        labels = torch.randint(0, 3, (2, 4, 6), generator=generator)
        labels[1, :, :2] = 2
        logits, labels = logits.to(device=device, dtype=torch.float32), labels.to(device)
        # Exact one-hot, spatial-only TP / (TP + alpha FP + beta FN + 1e-8).
        expected = (0.6015094950, 0.5990951811, 0.6037325821)
        for (alpha, beta), value in zip(((0.3, 0.7), (0.7, 0.3), (0.5, 0.5)), expected):
            loss = kornia.losses.tversky_loss(logits, labels, alpha, beta)
            self.assert_close(loss, logits.new_tensor(value), rtol=0, atol=1e-7)

    @pytest.mark.parametrize("alpha,beta", [(0.3, 0.7), (0.7, 0.3), (0.5, 0.5), (1.0, 1.0)])
    @pytest.mark.parametrize("eps", [0.0, 0.25])
    def test_target_presence_per_sample(self, device, dtype, alpha, beta, eps):
        probabilities = torch.tensor(
            [
                [[[0.5, 0.25, 0.125, 0.5]], [[0.25, 0.5, 0.125, 0.25]], [[0.25, 0.25, 0.75, 0.25]]],
                [[[0.5, 0.25, 0.5, 0.25]], [[0.25, 0.5, 0.25, 0.5]], [[0.25, 0.25, 0.25, 0.25]]],
            ],
            device=device,
            dtype=dtype,
        )
        labels = torch.tensor([[[0, 1, 1, 0]], [[2, 2, 2, 2]]], device=device)
        # Sample 0: (TP, FP, FN) = (1, 3/8, 1), (5/8, 1/2, 11/8).
        # Sample 1: (1, 0, 3). Absent classes can win argmax but are not averaged in.
        score_0 = 1 / (1 + alpha * 0.375 + beta + eps)
        score_1 = 0.625 / (0.625 + alpha * 0.5 + beta * 1.375 + eps)
        score_2 = 1 / (1 + beta * 3 + eps)
        expected = 1 - ((score_0 + score_1) / 2 + score_2) / 2
        loss = kornia.losses.tversky_loss(probabilities.log(), labels, alpha, beta, eps=eps)
        assert loss.dtype == dtype
        assert loss.device == device
        self.assert_close(loss, probabilities.new_tensor(expected))

    def test_absent_channels(self, device, dtype):
        labels = torch.tensor([[[0, 1, 1, 0]]], device=device)
        losses = []
        for num_classes in (2, 3, 4):
            logits = torch.full((1, num_classes, 1, 4), -40.0, device=device, dtype=dtype)
            # Imperfect predictions, unchanged when negligible unused channels are added.
            logits[:, :2] = torch.tensor([[[[2.0, 0.0, 1.0, 2.0]], [[0.0, 2.0, 2.0, 0.0]]]], device=device)
            losses.append(kornia.losses.tversky_loss(logits, labels, 0.3, 0.7))
        for loss in losses[1:]:
            self.assert_close(loss, losses[0], rtol=0, atol=1e-7)

    @pytest.mark.parametrize("alpha,beta", [(0.3, 0.7), (1.0, 0.0), (0.0, 1.0)])
    def test_present_class_without_correct_predictions(self, device, dtype, alpha, beta):
        logits = torch.tensor([[[[100.0, 100.0]], [[-100.0, -100.0]], [[-100.0, -100.0]]]], device=device)
        logits = logits.to(dtype).requires_grad_()
        labels = torch.tensor([[[0, 1]]], device=device)
        loss = kornia.losses.tversky_loss(logits, labels, alpha, beta)
        # Class 0: TP=1, FP=1, FN=0. Class 1: TP=0, FP=0, FN=1; its loss stays 1.
        self.assert_close(loss, logits.new_tensor(1 - (1 / (1 + alpha)) / 2))
        loss.backward()
        assert torch.isfinite(logits.grad).all()

    def test_exact_target_rare_class(self, device, dtype):
        labels = torch.zeros((1, 128, 192), device=device, dtype=torch.int64)
        labels[0, 0, 0] = 1
        logits = torch.full((1, 3, 128, 192), -40.0, device=device, dtype=dtype)
        logits.scatter_(1, labels[:, None], 40.0).requires_grad_()
        loss = kornia.losses.tversky_loss(logits, labels, 0.3, 0.7)
        # An epsilon floor in off-class target entries gives the rare class spurious FN mass.
        self.assert_close(loss, torch.zeros_like(loss), rtol=0, atol=1e-7)
        loss.backward()
        assert torch.isfinite(logits.grad).all()

    @pytest.mark.parametrize("eps", [0.0, 0.25])
    def test_macro_dice_correspondence(self, device, dtype, eps):
        logits = (torch.arange(18, device=device, dtype=dtype).reshape(2, 3, 1, 3) / 4).requires_grad_()
        labels = torch.tensor([[[0, 1, 1]], [[2, 2, -100]]], device=device)
        probabilities = logits.to(torch.float64 if dtype == torch.float64 else torch.float32).softmax(1)
        # Independent macro Dice over each sample's actual labels, as required by #5539.
        # Tversky's eps becomes 2*eps when the Dice numerator/denominator are doubled.
        sample_losses = []
        for sample, classes in enumerate(((0, 1), (2,))):
            class_losses = []
            valid = labels[sample] != -100
            for label in classes:
                pred = probabilities[sample, label] * valid
                target = (labels[sample] == label).to(pred.dtype)
                class_losses.append(1 - 2 * (pred * target).sum() / ((pred + target).sum() + 2 * eps))
            sample_losses.append(torch.stack(class_losses).mean())
        expected = torch.stack(sample_losses).mean().to(dtype)
        actual = kornia.losses.tversky_loss(logits, labels, 0.5, 0.5, eps=eps)
        self.assert_close(actual, expected)
        actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, logits)[0]
        assert torch.isfinite(actual_grad).all()
        self.assert_close(actual_grad, expected_grad)

    @pytest.mark.parametrize("ignore_index", [-100, 0, 255])
    @pytest.mark.parametrize("eps", [0.0, 1e-8])
    def test_ignored_samples(self, device, dtype, ignore_index, eps):
        logits = torch.zeros((2, 3, 1, 2), device=device, dtype=dtype, requires_grad=True)
        labels = torch.tensor([[[1, ignore_index]], [[ignore_index, ignore_index]]], device=device)
        kwargs = {"alpha": 0.3, "beta": 0.7, "eps": eps, "ignore_index": ignore_index}
        loss = kornia.losses.tversky_loss(logits, labels, **kwargs)
        # Valid sample: TP=1/3, FP=0, FN=2/3. Fully ignored sample: score=0, loss=1.
        self.assert_close(loss, logits.new_tensor(1 - (1 / 2.4) / 2))
        empty_loss = kornia.losses.tversky_loss(logits[1:], labels[1:], **kwargs)
        self.assert_close(empty_loss, logits.new_tensor(1.0), rtol=0, atol=0)
        loss.backward()
        assert torch.isfinite(logits.grad).all()
        self.assert_close(logits.grad[1], torch.zeros_like(logits.grad[1]), rtol=0, atol=0)
        self.assert_close(logits.grad[0, :, :, 1], torch.zeros_like(logits.grad[0, :, :, 1]), rtol=0, atol=0)
        assert torch.count_nonzero(logits.grad[0, :, :, 0]) > 0

    def test_absent_and_ignored_gradcheck(self, device):
        logits = torch.arange(12, device=device, dtype=torch.float64).reshape(2, 3, 1, 2) / 4
        labels = torch.tensor([[[1, -100]], [[-100, -100]]], device=device)
        self.gradcheck(
            kornia.losses.tversky_loss, (logits, labels, 0.3, 0.7), dtypes=[torch.float64, torch.int64, None, None]
        )

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
        # Half of the pixels have a logit gap of 20 and half a gap of 6, so the loss is about 6e-4 to 1e-3.
        # A ratio formed in bfloat16 has a step of 2**-8 near 1 and returns 0 or 2**-8 instead.
        logits = torch.zeros(1, 2, 128, 128, device=device, dtype=dtype)
        logits[:, 0, :, ::2] = 20.0
        logits[:, 0, :, 1::2] = 6.0
        labels = torch.zeros(1, 128, 128, device=device, dtype=torch.int64)

        reduction_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        p_true = logits.to(reduction_dtype).softmax(dim=1)[:, 0].cpu().double()
        # Only class 0 is present: TP=sum(p_true), FP=0, FN=N-TP.
        expected = 1.0 - p_true.sum() / (p_true.sum() + 0.5 * (p_true.numel() - p_true.sum()) + 1e-8)
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
        # Only class 0 is present, with TP=1, FP=0, FN=1 after masking.
        expected_loss = torch.tensor([1.0 / 3.0], device=device, dtype=dtype).squeeze()
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

    @pytest.mark.parametrize("ignored_sample", [False, True])
    def test_dynamo(self, device, dtype, torch_optimizer, ignored_sample):
        num_classes = 3
        params = (0.5, 0.05)
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        if ignored_sample:
            labels[0, :, 0] = -100
            labels[1] = -100

        op = kornia.losses.tversky_loss
        op_optimized = torch_optimizer(op)

        actual = op_optimized(logits, labels, *params)
        expected = op(logits, labels, *params)
        self.assert_close(actual, expected)

    def test_module(self, device, dtype):
        num_classes = 3
        params = (0.3, 0.7, 0.25, -100)
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()
        labels[0, :, 0] = -100
        labels[1] = -100

        op = kornia.losses.tversky_loss
        op_module = kornia.losses.TverskyLoss(*params)

        actual = op_module(logits, labels)
        expected = op(logits, labels, *params)
        self.assert_close(actual, expected)
