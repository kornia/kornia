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


def _lovasz_grad_reference(foreground_sorted):
    """Berman's lovasz_grad in float64, with the counts kept as Python integers (foreground_sorted holds 0. and 1.)."""
    total = int(foreground_sorted.sum().item())
    seen_foreground = seen_background = 0
    jaccard = []
    for value in foreground_sorted.tolist():
        if value:
            seen_foreground += 1
        else:
            seen_background += 1
        jaccard.append(1.0 - (total - seen_foreground) / (total + seen_background))
    out = torch.tensor(jaccard, dtype=torch.float64)
    out[1:] = out[1:] - out[:-1]
    return out


def _lovasz_softmax_reference(logits, labels, weight=None):
    """kornia's reduction of Berman's per-image Lovasz-Softmax, evaluated in float64 one sample and class at a time."""
    probabilities = logits.double().softmax(1)
    B, C = probabilities.shape[:2]
    per_class = torch.zeros(C, dtype=torch.float64)
    for b in range(B):
        for c in range(C):
            foreground = (labels[b] == c).double().flatten()
            errors = (foreground - probabilities[b, c].flatten()).abs()
            errors_sorted, permutation = errors.sort(descending=True)
            per_class[c] = per_class[c] + errors_sorted.dot(_lovasz_grad_reference(foreground[permutation])) / B
    if weight is not None:
        per_class = per_class * weight.double()
    return per_class.mean()


class TestLovaszSoftmaxLoss(BaseTester):
    def test_smoke(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 1, 1, device=device, dtype=dtype)
        labels = torch.rand(2, 1, 1) * num_classes
        labels = labels.to(device).long()

        criterion = kornia.losses.LovaszSoftmaxLoss()
        assert criterion(logits, labels) is not None

    def test_exception(self):
        from kornia.core.exceptions import ShapeError

        criterion = kornia.losses.LovaszSoftmaxLoss()

        with pytest.raises(ShapeError) as errinfo:
            criterion(torch.rand(1), torch.rand(1))
        assert "Shape dimension mismatch" in str(errinfo.value) or "Expected shape" in str(errinfo.value)

        with pytest.raises(ShapeError) as errinfo:
            criterion(torch.rand(1, 1, 1, 1), torch.rand(1))
        assert "Shape dimension mismatch" in str(errinfo.value) or "Expected shape" in str(errinfo.value)

        with pytest.raises(ValueError) as errinfo:
            criterion(torch.rand(1, 1, 1, 1), torch.rand(1, 1, 1))
        assert "Invalid pred shape, we expect BxNxHxW, with N > 1." in str(errinfo)

        with pytest.raises(ValueError) as errinfo:
            criterion(torch.rand(1, 2, 1, 1), torch.rand(1, 1, 2))
        assert "pred and target shapes must be the same. Got:" in str(errinfo)

        with pytest.raises(ValueError) as errinfo:
            criterion(torch.rand(1, 2, 1, 1), torch.rand(1, 1, 1, device="meta"))
        assert "pred and target must be in the same device. Got:" in str(errinfo)

    def test_binary(self, device, dtype):
        num_classes = 1
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        criterion = kornia.losses.LovaszSoftmaxLoss()
        with pytest.raises(Exception):
            criterion(logits, labels)

    def test_all_ones(self, device, dtype):
        num_classes = 2
        # make perfect prediction
        # note that softmax(prediction[:, 1]) == 1. softmax(prediction[:, 0]) == 0.
        prediction = torch.zeros(2, num_classes, 1, 2, device=device, dtype=dtype)
        prediction[:, 1] = 100.0
        labels = torch.ones(2, 1, 2, device=device, dtype=torch.int64)

        criterion = kornia.losses.LovaszSoftmaxLoss()
        loss = criterion(prediction, labels)

        self.assert_close(loss, torch.zeros_like(loss), rtol=1e-3, atol=1e-3)

    def test_weight(self, device, dtype):
        num_classes = 2
        # make perfect prediction
        # note that softmax(prediction[:, 1]) == 1. softmax(prediction[:, 0]) == 0.
        prediction = torch.zeros(2, num_classes, 1, 2, device=device, dtype=dtype)
        prediction[:, 0] = 100.0
        labels = torch.ones(2, 1, 2, device=device, dtype=torch.int64)

        criterion = kornia.losses.LovaszSoftmaxLoss(weight=torch.tensor([1.0, 0.0], device=device, dtype=dtype))
        loss = criterion(prediction, labels)

        self.assert_close(loss, 0.5 * torch.ones_like(loss), rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("case", ["multiclass", "absent_classes", "single_pixel"])
    @pytest.mark.parametrize("weighted", [False, True])
    def test_foreground_reference(self, device, dtype, case, weighted):
        # Berman's reference with classes="all", per_image=True:
        # https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py
        # lovasz_softmax(probabilities, labels, classes="all", per_image=True).
        probabilities = torch.tensor(
            [[[[0.6, 0.21, 0.12]], [[0.29, 0.68, 0.19]], [[0.11, 0.11, 0.69]]]], device=device, dtype=dtype
        )
        labels = torch.tensor([[[0, 1, 2]]], device=device)
        expected_per_class = [0.4, 0.32, 0.31]
        if case == "absent_classes":
            labels = torch.zeros_like(labels)
            expected_per_class = [0.69, 0.68, 0.69]
        elif case == "single_pixel":
            probabilities = probabilities[..., :1]
            labels = labels[..., :1]
            expected_per_class = [0.4, 0.29, 0.11]
        weight = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype) if weighted else None
        expected = torch.tensor(expected_per_class, device=device, dtype=dtype)
        if weight is not None:
            expected = expected * weight
        logits = probabilities.log()
        loss = kornia.losses.lovasz_softmax_loss(logits, labels, weight)
        self.assert_close(loss.to(dtype), expected.mean())
        self.assert_close(kornia.losses.LovaszSoftmaxLoss(weight)(logits, labels).to(dtype), expected.mean())

    @pytest.mark.parametrize("order", [(2, 0, 1), (1, 2, 0), (0, 2, 1)])
    @pytest.mark.parametrize("absent_classes", [False, True])
    @pytest.mark.parametrize("weighted", [False, True])
    def test_class_permutation(self, device, dtype, order, absent_classes, weighted):
        logits = torch.tensor(
            [[[[1.2, -0.3, 0.7]], [[0.1, 1.4, -0.2]], [[-0.8, 0.2, 1.6]]]],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        labels = torch.tensor([[[0, 1, 2]]], device=device)
        if absent_classes:
            labels = torch.zeros_like(labels)
        permutation = torch.tensor(order, device=device)
        inverse = permutation.argsort()
        weight = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype) if weighted else None
        original = kornia.losses.lovasz_softmax_loss(logits, labels, weight)
        permuted = kornia.losses.lovasz_softmax_loss(
            logits[:, permutation], inverse[labels], weight[permutation] if weight is not None else None
        )
        self.assert_close(permuted, original)
        self.assert_close(torch.autograd.grad(permuted, logits)[0], torch.autograd.grad(original, logits)[0])

    def test_foreground_gradient_reference(self, device, dtype):
        # Same reference call as test_foreground_reference, differentiated through softmax.
        # Errors have distinct maxima so rounding does not select a different subgradient at a tie.
        probabilities = torch.tensor(
            [[[[0.6, 0.21, 0.12]], [[0.29, 0.68, 0.19]], [[0.11, 0.11, 0.69]]]], device=device, dtype=dtype
        )
        logits = probabilities.log().requires_grad_()
        labels = torch.tensor([[[0, 1, 2]]], device=device)
        expected = torch.tensor(
            [[[[-0.08, 0.0476, 0.0276]], [[0.058, -0.0725333333, 0.0437]], [[0.022, 0.0249333333, -0.0713]]]],
            device=device,
            dtype=dtype,
        )
        loss = kornia.losses.lovasz_softmax_loss(logits, labels)
        self.assert_close(torch.autograd.grad(loss, logits)[0], expected)

    def test_dynamo_foreground(self, device, dtype, torch_optimizer):
        probabilities = torch.tensor(
            [[[[0.6, 0.21, 0.12]], [[0.29, 0.68, 0.19]], [[0.11, 0.11, 0.69]]]], device=device, dtype=dtype
        )
        logits = probabilities.log().requires_grad_()
        labels = torch.tensor([[[0, 1, 2]]], device=device)
        op = kornia.losses.lovasz_softmax_loss
        loss = op(logits, labels)
        optimized = torch_optimizer(op)(logits, labels)
        self.assert_close(optimized, torch.tensor(1.03 / 3, device=device, dtype=dtype))
        self.assert_close(torch.autograd.grad(optimized, logits)[0], torch.autograd.grad(loss, logits)[0])

    def test_large_foreground_counts(self, device, dtype):
        # More than 65504 foreground pixels exceed float16's finite range.
        logits = torch.zeros((1, 3, 257, 257), device=device, dtype=dtype, requires_grad=True)
        labels = torch.zeros((1, 257, 257), device=device, dtype=torch.int64)
        loss = kornia.losses.lovasz_softmax_loss(logits, labels)
        self.assert_close(loss.to(dtype), torch.tensor(4 / 9, device=device, dtype=dtype))
        assert torch.isfinite(torch.autograd.grad(loss, logits)[0]).all()

    def test_output_dtype_follows_the_prediction(self, device, dtype):
        logits = torch.randn(2, 3, 4, 5, device=device, dtype=dtype)
        labels = torch.randint(0, 3, (2, 4, 5), device=device)
        weight = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        assert kornia.losses.lovasz_softmax_loss(logits, labels).dtype == dtype
        assert kornia.losses.lovasz_softmax_loss(logits, labels, weight).dtype == dtype
        assert kornia.losses.LovaszSoftmaxLoss(weight)(logits, labels).dtype == dtype
        # the weight dtype is promoted into the output, as in dice_loss
        promoted = torch.promote_types(dtype, torch.float32)
        assert kornia.losses.lovasz_softmax_loss(logits, labels, weight.float()).dtype == promoted

    def test_default_dtype_does_not_change_the_loss(self, device, dtype):
        logits = torch.randn(2, 3, 4, 5, device=device, dtype=dtype)
        labels = torch.randint(0, 3, (2, 4, 5), device=device)
        expected = kornia.losses.lovasz_softmax_loss(logits, labels)
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32 if original == torch.float64 else torch.float64)
            loss = kornia.losses.lovasz_softmax_loss(logits, labels)
        finally:
            torch.set_default_dtype(original)
        assert loss.dtype == dtype
        self.assert_close(loss, expected)

    @pytest.mark.parametrize("weighted", [False, True])
    def test_float64_matches_a_float64_reference(self, device, weighted):
        if device.type == "mps":
            pytest.skip("MPS does not support float64")
        torch.manual_seed(0)
        logits = torch.randn(2, 3, 16, 24, device=device, dtype=torch.float64, requires_grad=True)
        labels = torch.randint(0, 3, (2, 16, 24), device=device)
        weight = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=torch.float64) if weighted else None
        reference_logits = logits.detach().cpu().clone().requires_grad_()
        expected = _lovasz_softmax_reference(reference_logits, labels.cpu(), None if weight is None else weight.cpu())
        expected_grad = torch.autograd.grad(expected, reference_logits)[0]
        loss = kornia.losses.lovasz_softmax_loss(logits, labels, weight)
        # the Jaccard weights are exact in float64: only the summation order differs from the reference
        self.assert_close(loss, expected.to(device), rtol=1e-12, atol=1e-12)
        self.assert_close(torch.autograd.grad(loss, logits)[0], expected_grad.to(device), rtol=1e-12, atol=1e-15)

    def test_gradcheck(self, device, dtype):
        num_classes = 4
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=torch.float64)
        labels = torch.randint(0, num_classes, (2, 3, 2), device=device)
        self.gradcheck(kornia.losses.lovasz_softmax_loss, (logits, labels), dtypes=[torch.float64, torch.int64])

    @pytest.mark.skip(reason="Not matching results")
    def test_dynamo(self, device, dtype, torch_optimizer):
        # TODO: investigate if we can fix it or report the issue
        num_classes = 6
        logits = torch.rand(2, num_classes, 1, 2, device=device, dtype=dtype)
        labels = torch.randint(0, num_classes, (2, 1, 2), device=device)

        op = kornia.losses.lovasz_softmax_loss
        op_optimized = torch_optimizer(op)

        self.assert_close(op(logits, labels), op_optimized(logits, labels))

    def test_module(self, device, dtype):
        num_classes = 5
        logits = torch.rand(2, num_classes, 1, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 1, 2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.lovasz_softmax_loss
        op_module = kornia.losses.LovaszSoftmaxLoss()

        self.assert_close(op(logits, labels), op_module(logits, labels))


class TestConventionsLovaszSoftmaxLoss(BaseTester):
    """Pins for the batch, class and weight reduction of :func:`lovasz_softmax_loss`."""

    @staticmethod
    def _logits_and_labels(device, dtype):
        g = torch.Generator().manual_seed(0)
        logits = torch.randn(2, 4, 4, 6, generator=g)
        labels = torch.randint(0, 3, (2, 4, 6), generator=g)
        labels[1, 1:3, 2:5] = 3  # class 3 is absent from image 0
        logits = logits + 3.0 * (labels[:, None] == torch.arange(4)[:, None, None])
        logits[0, 3, 0, :2] += 6.0  # image 0 predicts the absent class 3 on two pixels
        return logits.to(device=device, dtype=dtype), labels.to(device)

    def test_convention_lovasz_softmax_loss_averages_all_classes_per_image(self, device, dtype):
        # Berman's lovasz_softmax(softmax(pred), classes='all', per_image=True): each image is scored on its own (the
        # batch flattened into one image, per_image=False, gives another value), and a class absent from an image
        # still enters that image's mean over the C classes, with the image's largest probability for it (Berman's
        # default classes='present' skips it)
        logits, labels = self._logits_and_labels(device, dtype)
        loss = kornia.losses.lovasz_softmax_loss
        per_image = (loss(logits[:1], labels[:1]) + loss(logits[1:], labels[1:])) / 2
        self.assert_close(loss(logits, labels), per_image)
        flattened = loss(torch.cat([logits[0], logits[1]], -1)[None], torch.cat([labels[0], labels[1]], -1)[None])
        assert (flattened - per_image).abs() > 0.02
        image_0 = logits[:1].cpu(), labels[:1].cpu()
        without_class_3 = _lovasz_softmax_reference(*image_0, torch.tensor([1.0, 1.0, 1.0, 0.0]))
        largest_p_3 = image_0[0].double().softmax(1)[:, 3].max()
        expected = without_class_3 + largest_p_3 / 4
        self.assert_close(loss(logits[:1], labels[:1]), expected.to(device=device, dtype=dtype))
        assert largest_p_3 / 4 - without_class_3 / 3 > 0.1  # the mean over the present classes would differ

    def test_convention_lovasz_softmax_loss_weight_is_not_normalised(self, device, dtype):
        # weight scales each class's term and the mean over classes still divides by C, not by sum(weight): a uniform
        # weight of 2 doubles the loss, where dice_loss 'macro' divides by sum(weight)
        logits, labels = self._logits_and_labels(device, dtype)
        loss = kornia.losses.lovasz_softmax_loss
        self.assert_close(
            loss(logits, labels, torch.full((4,), 2.0, device=device, dtype=dtype)), 2 * loss(logits, labels)
        )
        weight = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device, dtype=dtype)
        one_class = torch.eye(4, device=device, dtype=dtype)
        self.assert_close(
            loss(logits, labels, weight), sum(w * loss(logits, labels, e) for w, e in zip(weight, one_class))
        )
