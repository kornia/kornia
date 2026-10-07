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


class TestDiceLoss(BaseTester):
    def test_macro_absent_channels_5539(self, device, dtype):
        labels = torch.zeros((1, 4, 6), device=device, dtype=torch.int64)
        labels[0, :, 3:] = 1
        labels[0, 0, 0] = 1
        losses = []
        for num_classes in (2, 3, 4):
            logits = torch.full((1, num_classes, 4, 6), -20.0, device=device, dtype=dtype)
            logits.scatter_(1, labels[:, None], 20.0)
            loss = kornia.losses.dice_loss(logits, labels, average="macro")
            assert loss.dtype == dtype
            assert loss.device == device
            losses.append(loss)
        for loss in losses[1:]:
            self.assert_close(loss, losses[0], rtol=0, atol=1e-6)

    @pytest.mark.parametrize("weighted", [False, True])
    def test_macro_target_presence_per_sample(self, device, dtype, weighted):
        labels = torch.tensor([[[0, 1, 1, 0]], [[2, 2, 2, 2]]], device=device)
        logits = torch.tensor(
            [
                [[[2.0, 0.0, 1.0, -1.0]], [[0.0, 2.0, 0.0, 1.0]], [[1.0, 1.0, 3.0, 2.0]]],
                [[[1.0, 3.0, 2.0, 0.0]], [[2.0, 0.0, 1.0, 3.0]], [[0.0, 2.0, 0.0, 1.0]]],
            ],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        weight = torch.tensor([0.25, 0.5, 8.0], device=device, dtype=dtype) if weighted else None
        probabilities = logits.softmax(1)
        # Evaluate only actual target labels, including when an absent class wins argmax.
        sample_losses = []
        for sample in range(2):
            class_losses, class_weights = [], []
            for label in labels[sample].unique():
                # dice_loss builds an exact one-hot target, in float32 for half inputs.
                reduction_dtype = torch.float64 if dtype == torch.float64 else torch.float32
                target = (labels[sample] == label).to(reduction_dtype)
                pred = probabilities[sample, label]
                intersection = (pred * target).sum(dtype=reduction_dtype)
                cardinality = (pred + target).sum(dtype=reduction_dtype)
                class_losses.append(1 - 2 * intersection / (cardinality + 1e-8))
                class_weights.append(weight[label] if weighted else logits.new_tensor(1.0))
            weights = torch.stack(class_weights)
            sample_losses.append((torch.stack(class_losses) * weights).sum() / weights.sum())
        expected = torch.stack(sample_losses).mean().to(dtype)
        actual = kornia.losses.dice_loss(logits, labels, average="macro", weight=weight)
        self.assert_close(actual, expected)
        actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, logits)[0]
        assert actual_grad.dtype == dtype
        assert actual_grad.device == device
        assert torch.isfinite(actual_grad).all()
        self.assert_close(actual_grad, expected_grad)

    @pytest.mark.parametrize(
        "average,all_classes,weighted",
        [("micro", False, False), ("micro", True, False), ("macro", True, False), ("macro", True, True)],
    )
    def test_unchanged_aggregation(self, device, dtype, average, all_classes, weighted):
        labels = torch.tensor([[[0, 1, 2 if all_classes else 1]]], device=device)
        logits = torch.arange(9, device=device, dtype=dtype).reshape(1, 3, 1, 3) / 4
        # dice_loss builds the one-hot target in float32 for half inputs.
        reduction_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        target = kornia.losses.one_hot(labels, 3, device=device, dtype=reduction_dtype)
        pred = logits.softmax(1)
        dims = (1, 2, 3) if average == "micro" else (2, 3)
        intersection = (pred * target).sum(dims, dtype=reduction_dtype)
        cardinality = (pred + target).sum(dims, dtype=reduction_dtype)
        expected = 1 - 2 * intersection / (cardinality + 1e-8)
        weight = torch.tensor([0.25, 0.5, 8.0], device=device, dtype=dtype) if weighted else None
        if weighted:
            expected = (expected * weight).sum(-1) / weight.sum()
        expected = expected.mean().to(dtype)
        actual = kornia.losses.dice_loss(logits, labels, average=average, weight=weight)
        self.assert_close(actual, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("ignore_index", [-100, 0, 255])
    @pytest.mark.parametrize("weighted", [False, True])
    def test_macro_ignored_samples(self, device, dtype, ignore_index, weighted):
        labels = torch.tensor([[[1, ignore_index]], [[ignore_index, ignore_index]]], device=device)
        logits = torch.zeros((2, 3, 1, 2), device=device, dtype=dtype, requires_grad=True)
        weight = torch.tensor([8.0, 0.25, 4.0], device=device, dtype=dtype) if weighted else None
        kwargs = {"average": "macro", "ignore_index": ignore_index, "weight": weight}
        loss = kornia.losses.dice_loss(logits, labels, **kwargs)
        # One valid class: Dice = 2 * (1/3) / (1 + 1/3) = 1/2; empty sample: loss 1.
        self.assert_close(loss, logits.new_tensor(0.75))
        empty_loss = kornia.losses.dice_loss(logits[1:], labels[1:], **kwargs)
        self.assert_close(empty_loss, logits.new_tensor(1.0), rtol=0, atol=0)
        loss.backward()
        assert torch.isfinite(logits.grad).all()
        self.assert_close(logits.grad[1], torch.zeros_like(logits.grad[1]), rtol=0, atol=0)
        self.assert_close(logits.grad[0, :, :, 1], torch.zeros_like(logits.grad[0, :, :, 1]), rtol=0, atol=0)

    def test_macro_sample_without_weighted_class(self, device, dtype):
        # Sample 0 holds only class 0, whose weight is 0: no weighted class is left, as in a fully ignored sample, so
        # it keeps loss 1 with a zero gradient instead of 0 / 0 spreading NaN over the batch.
        labels = torch.tensor([[[0, 0, 0, 0]], [[0, 1, 1, 2]]], device=device)
        logits = torch.tensor(
            [
                [[[2.0, 0.0, 1.0, -1.0]], [[0.0, 2.0, 0.0, 1.0]], [[1.0, 1.0, 3.0, 2.0]]],
                [[[1.0, 3.0, 2.0, 0.0]], [[2.0, 0.0, 1.0, 3.0]], [[0.0, 2.0, 0.0, 1.0]]],
            ],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        weight = torch.tensor([0.0, 1.0, 2.0], device=device, dtype=dtype)
        kwargs = {"average": "macro", "weight": weight}
        alone = kornia.losses.dice_loss(logits[:1], labels[:1], **kwargs)
        self.assert_close(alone, logits.new_tensor(1.0), rtol=0, atol=0)
        loss = kornia.losses.dice_loss(logits, labels, **kwargs)
        self.assert_close(loss, (1 + kornia.losses.dice_loss(logits[1:], labels[1:], **kwargs)) / 2)
        loss.backward()
        assert torch.isfinite(logits.grad).all()
        self.assert_close(logits.grad[0], torch.zeros_like(logits.grad[0]), rtol=0, atol=0)

    def test_macro_absent_gradcheck(self, device):
        logits = torch.arange(12, device=device, dtype=torch.float64).reshape(2, 3, 1, 2) / 4
        labels = torch.tensor([[[1, -100]], [[-100, -100]]], device=device)
        weight = torch.tensor([8.0, 0.25, 4.0], device=device, dtype=torch.float64)
        self.gradcheck(
            lambda pred, target: kornia.losses.dice_loss(pred, target, average="macro", weight=weight),
            (logits, labels),
            dtypes=[torch.float64, torch.int64],
        )

    @pytest.mark.parametrize("height", [256, 512])
    @pytest.mark.parametrize("average", ["micro", "macro"])
    @pytest.mark.parametrize("weighted", [False, True])
    def test_large_image_reduction(self, device, dtype, height, average, weighted):
        logits = torch.full((1, 2, height, 256), -2.0, device=device, dtype=dtype, requires_grad=True)
        with torch.no_grad():
            logits[:, 0] = 2.0
        labels = torch.zeros((1, height, 256), device=device, dtype=torch.int64)
        weight = torch.tensor([1.0, 2.0], device=device, dtype=dtype) if weighted else None
        criterion = kornia.losses.DiceLoss(average=average, weight=weight)

        loss = criterion(logits, labels)
        reference_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        reference = kornia.losses.dice_loss(
            logits.to(reference_dtype),
            labels,
            average=average,
            weight=weight.to(reference_dtype) if weight is not None else None,
        ).to(dtype)

        assert loss.dtype == dtype
        assert loss.device == device
        assert torch.isfinite(loss)
        self.assert_close(loss, reference)
        loss.backward()
        assert torch.isfinite(logits.grad).all()

    @pytest.mark.parametrize("average", ["micro", "macro"])
    def test_weight_dtype_promotion(self, device, dtype, average):
        logits = torch.tensor([2.0, -2.0], device=device, dtype=dtype).reshape(1, 2, 1, 1)
        labels = torch.zeros((1, 1, 1), device=device, dtype=torch.int64)
        weight = torch.tensor([1.0, 2.0], device=device, dtype=torch.float32)
        loss = kornia.losses.dice_loss(logits, labels, average=average, weight=weight)
        assert loss.dtype == torch.promote_types(dtype, weight.dtype)

    def test_smoke(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        criterion = kornia.losses.DiceLoss()
        assert criterion(logits, labels) is not None

    @pytest.mark.parametrize("ignore_index", [-100, None])
    def test_all_zeros(self, device, dtype, ignore_index):
        num_classes = 3
        logits = torch.zeros(2, num_classes, 1, 2, device=device, dtype=dtype)
        logits[:, 0] = 10.0
        logits[:, 1] = 1.0
        logits[:, 2] = 1.0
        labels = torch.zeros(2, 1, 2, device=device, dtype=torch.int64)

        criterion = kornia.losses.DiceLoss(ignore_index=ignore_index)
        loss = criterion(logits, labels)
        self.assert_close(loss, torch.zeros_like(loss), rtol=1e-3, atol=1e-3)

    def test_perfect_prediction_of_a_rare_class(self, device, dtype):
        # The target is an exact one-hot, so a perfect prediction scores 0 whatever the image size. The former
        # eps floor of the target gave the one-pixel class a cardinality of 1 + eps * (pixels - 1), 0.006 here.
        labels = torch.zeros(1, 128, 192, device=device, dtype=torch.int64)
        labels[0, 10, 10] = 1
        logits = torch.full((1, 2, 128, 192), -30.0, device=device, dtype=dtype).scatter(1, labels[:, None], 30.0)

        loss = kornia.losses.dice_loss(logits, labels, average="macro")

        self.assert_close(loss, torch.zeros_like(loss))

    def test_gradient_of_an_absent_class_is_finite(self, device, dtype):
        # Class 2 is absent from the target and its probabilities underflow, so its cardinality is eps alone and its
        # intersection gradient is far past the float16 range. Times the exact zero target, that must not give NaN.
        labels = torch.zeros(1, 16, 16, device=device, dtype=torch.int64)
        labels[0, :, 8:] = 1
        logits = torch.full((1, 3, 16, 16), -10.0, device=device, dtype=dtype).scatter(1, labels[:, None], 10.0)
        logits.requires_grad_(True)

        loss = kornia.losses.dice_loss(logits, labels, average="macro")
        (grad,) = torch.autograd.grad(loss, logits)

        assert grad.isfinite().all()

    def test_exception(self):
        with pytest.raises(ValueError) as errinf:
            kornia.losses.DiceLoss()(torch.rand(1, 1, 1), torch.rand(1, 1, 1))
        assert "Invalid pred shape, we expect BxNxHxW. Got:" in str(errinf)

        with pytest.raises(ValueError) as errinf:
            kornia.losses.DiceLoss()(torch.rand(1, 1, 1, 1), torch.rand(1, 1, 1, 2))
        assert "pred and target shapes must be the same. Got: " in str(errinf)

        with pytest.raises(ValueError) as errinf:
            kornia.losses.DiceLoss()(torch.rand(1, 1, 1, 1), torch.rand(1, 1, 1, 1, device="meta"))
        assert "pred and target must be in the same device. Got:" in str(errinf)

        # The target batch has to match the prediction batch, as for focal_loss (#5544).
        with pytest.raises(ValueError, match=r"Expected target size torch.Size\(\[2, 4, 6\]\)"):
            kornia.losses.DiceLoss()(torch.rand(2, 3, 4, 6), torch.randint(0, 3, (1, 4, 6)))

        with pytest.raises(ValueError, match=r"Expected target size torch.Size\(\[1, 4, 6\]\)"):
            kornia.losses.DiceLoss()(torch.rand(1, 3, 4, 6), torch.randint(0, 3, (2, 4, 6)))

        # A target with a channel axis, (B, 1, H, W), is not (B, H, W) either.
        with pytest.raises(ValueError, match=r"Expected target size torch.Size\(\[1, 4, 6\]\)"):
            kornia.losses.DiceLoss()(torch.rand(1, 3, 4, 6), torch.randint(0, 3, (1, 1, 4, 6)))

    def test_averaging_micro(self, device, dtype):
        num_classes = 2
        eps = 1e-8

        logits = torch.zeros(1, num_classes, 4, 1, device=device, dtype=dtype)
        logits[:, 0, 0:3] = 10.0
        logits[:, 0, 3:4] = 1.0
        logits[:, 1, 0:3] = 1.0
        logits[:, 1, 3:4] = 10.0

        labels = torch.zeros(1, 4, 1, device=device, dtype=torch.int64)

        exp_1_0 = torch.exp(torch.tensor([1.0], device=device, dtype=dtype))
        exp_10_0 = torch.exp(torch.tensor([10.0], device=device, dtype=dtype))

        expected_intersection = (3.0 * exp_10_0 + 1.0 * exp_1_0) / (exp_1_0 + exp_10_0)
        expected_cardinality = 8.0  # for micro averaging cardinality is equal 2 * H * W
        expected_loss = 1.0 - 2.0 * expected_intersection / (expected_cardinality + eps)
        expected_loss = expected_loss.squeeze()

        criterion = kornia.losses.DiceLoss(average="micro", eps=eps)
        loss = criterion(logits, labels)
        self.assert_close(loss, expected_loss, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("avg", ["micro", "macro"])
    def test_weight(self, device, dtype, avg):
        num_classes = 3
        eps = 1e-8
        logits = torch.zeros(4, num_classes, 1, 4, device=device, dtype=dtype)
        logits[:, 0, :, 0] = 100.0
        logits[:, 2, :, 1:] = 100.0
        labels = torch.tensor([0, 1, 2, 2], device=device, dtype=torch.int64).expand((4, 1, -1))

        # class 0 is all correct
        expected_loss = torch.tensor([0.0], device=device, dtype=dtype).squeeze()
        weight = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        criterion = kornia.losses.DiceLoss(average=avg, eps=eps, weight=weight)
        loss = criterion(logits, labels)
        self.assert_close(loss, expected_loss, rtol=1e-3, atol=1e-3)

        # class 1 is all incorrect
        expected_loss = torch.tensor([1.0], device=device, dtype=dtype).squeeze()
        weight = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        criterion = kornia.losses.DiceLoss(average=avg, eps=eps, weight=weight)
        loss = criterion(logits, labels)
        self.assert_close(loss, expected_loss, rtol=1e-3, atol=1e-3)

        # class 2 is partially correct
        expected_loss = torch.tensor([1.0 / 5.0], device=device, dtype=dtype).squeeze()
        weight = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        criterion = kornia.losses.DiceLoss(average=avg, eps=eps, weight=weight)
        loss = criterion(logits, labels)
        self.assert_close(loss, expected_loss, rtol=1e-3, atol=1e-3)

        # ignore class 3
        expected_loss = kornia.losses.dice_loss(logits, labels, average=avg, eps=eps)
        weight = torch.tensor([1.0, 1.0, 1.0, 0.0], device=device, dtype=dtype)
        criterion = kornia.losses.DiceLoss(average=avg, eps=eps, weight=weight)
        loss = criterion(torch.cat([logits, logits.new_zeros((4, 1, 1, 4))], dim=1), labels)
        self.assert_close(loss, expected_loss, rtol=1e-3, atol=1e-3)

        # test non binary weights
        w_cl_0, w_cl_1 = 0.3, 0.7
        if avg == "macro":
            expected_loss = torch.tensor([0.7], device=device, dtype=dtype).squeeze()
        else:
            dims = (1, 2)
            preds = logits.argmax(1)
            tp_cl_0 = ((preds == 0) & (labels == 0)).sum(dims)
            tp_cl_1 = ((preds == 1) & (labels == 1)).sum(dims)

            fnfp_cl_0 = ((preds == 0) ^ (labels == 0)).sum(dims)
            fnfp_cl_1 = ((preds == 1) ^ (labels == 1)).sum(dims)

            expected_loss = (
                (
                    1
                    - 2
                    * (w_cl_0 * tp_cl_0 + w_cl_1 * tp_cl_1)
                    / (w_cl_0 * (2 * tp_cl_0 + fnfp_cl_0) + w_cl_1 * (2 * tp_cl_1 + fnfp_cl_1) + eps)
                )
                .mean()
                .to(dtype)
            )

        weight = torch.tensor([w_cl_0, w_cl_1, 0.0], device=device, dtype=dtype)
        criterion = kornia.losses.DiceLoss(average=avg, eps=eps, weight=weight)
        loss = criterion(logits, labels)
        self.assert_close(loss, expected_loss, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("avg", ["micro", "macro"])
    @pytest.mark.parametrize("scale", [0.5, 2.0])
    def test_uniform_weight(self, device, dtype, avg, scale):
        num_classes = 3
        logits = torch.randn(2, num_classes, 4, 6, device=device, dtype=dtype)
        labels = torch.randint(num_classes, (2, 4, 6), device=device)
        weight = torch.full((num_classes,), scale, device=device, dtype=dtype)
        expected = kornia.losses.dice_loss(logits, labels, average=avg)
        self.assert_close(kornia.losses.dice_loss(logits, labels, average=avg, weight=weight), expected)

        # a perfect prediction has zero loss whatever the scale of the weights
        perfect = torch.full_like(logits, -30.0).scatter(1, labels[:, None], 30.0)
        loss = kornia.losses.dice_loss(perfect, labels, average=avg, weight=weight)
        self.assert_close(loss, torch.zeros_like(loss), rtol=1e-3, atol=1e-3)

    def test_weight_micro_formula(self, device, dtype):
        num_classes = 3
        eps = 1e-8
        logits = torch.randn(2, num_classes, 4, 6, device=device, dtype=dtype)
        labels = torch.randint(num_classes, (2, 4, 6), device=device)
        weight = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)

        # weighted Dice per sample: 2 sum(w p t) / sum(w (p + t)), summed over classes and pixels
        probs = logits.softmax(1)
        targets = torch.nn.functional.one_hot(labels, num_classes).permute(0, 3, 1, 2).to(dtype)
        w = weight.view(1, -1, 1, 1)
        intersection = (w * probs * targets).sum((1, 2, 3))
        cardinality = (w * (probs + targets)).sum((1, 2, 3))
        expected = (1.0 - 2.0 * intersection / (cardinality + eps)).mean()

        loss = kornia.losses.dice_loss(logits, labels, average="micro", eps=eps, weight=weight)
        self.assert_close(loss, expected)
        assert 0.0 <= loss.item() <= 1.0

    def test_averaging_macro(self, device, dtype):
        num_classes = 2
        eps = 1e-8

        logits = torch.zeros(1, num_classes, 1, 4, device=device, dtype=dtype)
        logits[:, 0, :, 0:3] = 10.0
        logits[:, 0, :, 3:4] = 1.0
        logits[:, 1, :, 0:3] = 1.0
        logits[:, 1, :, 3:4] = 10.0

        labels = torch.zeros(1, 1, 4, device=device, dtype=torch.int64)

        exp_1_0 = torch.exp(torch.tensor([1.0], device=device, dtype=dtype))
        exp_10_0 = torch.exp(torch.tensor([10.0], device=device, dtype=dtype))

        expected_intersection_1 = (3.0 * exp_10_0 + exp_1_0) / (exp_1_0 + exp_10_0)
        expected_cardinality_1 = 4.0 + (3.0 * exp_10_0 + 1.0 * exp_1_0) / (exp_1_0 + exp_10_0)

        reduction_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        expected_loss_1 = 1.0 - 2.0 * expected_intersection_1.to(reduction_dtype) / (
            expected_cardinality_1.to(reduction_dtype) + eps
        )
        # Class 1 is absent from the target, so only class 0 contributes to the macro mean.
        expected_loss = expected_loss_1.squeeze().to(dtype)

        criterion = kornia.losses.DiceLoss(average="macro", eps=eps)
        loss = criterion(logits, labels)
        self.assert_close(loss, expected_loss, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("ignore_index", [-100, 255])
    def test_ignore_index(self, device, dtype, ignore_index):
        num_classes = 2
        eps = 1e-8

        logits = torch.zeros(2, num_classes, 1, 4, device=device, dtype=dtype)
        logits[:, 0, :, 0] = 100.0
        logits[:, 1, :, 1:] = 100.0
        labels = torch.zeros(2, 1, 4, device=device, dtype=torch.int64)

        labels[..., 2:] = ignore_index
        expected_loss = torch.tensor([1.0 / 2.0], device=device, dtype=dtype).squeeze()
        criterion = kornia.losses.DiceLoss(average="micro", eps=eps, ignore_index=ignore_index)
        loss = criterion(logits, labels)
        self.assert_close(loss, expected_loss, rtol=1e-3, atol=1e-3)

    def test_gradcheck(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=torch.float64)
        labels = torch.randint(0, num_classes, (2, 3, 2), device=device)
        ignore = torch.rand(2, 3, 2, device=device) > 0.8
        labels[ignore] = -100
        self.gradcheck(kornia.losses.dice_loss, (logits, labels), dtypes=[torch.float64, torch.int64])

    @pytest.mark.parametrize("average", ["micro", "macro"])
    def test_dynamo(self, device, dtype, torch_optimizer, average):
        num_classes = 3
        logits = torch.rand(2, num_classes, 1, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 1, 2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.dice_loss
        op_optimized = torch_optimizer(op)

        self.assert_close(op(logits, labels, average=average), op_optimized(logits, labels, average=average))

    def test_module(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 1, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 1, 2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.dice_loss
        op_module = kornia.losses.DiceLoss()

        self.assert_close(op(logits, labels), op_module(logits, labels))
