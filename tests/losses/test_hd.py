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

import numpy as np
import pytest
import torch

import kornia
from kornia.core._compat import torch_version_lt

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, dynamo_is_available

# The guard these tests pin (#5037) is only exercised where the two capture flags differ: from torch 2.14, Dynamo
# no longer folds the export flag to True inside torch.compile. On older releases it already did, as on 2.5.1 where
# kornia's helper falls back to is_compiling().
_CAPTURE_UNAVAILABLE_REASON = (
    f"{DYNAMO_UNAVAILABLE_REASON}, and before torch 2.14 the export flag was already True under torch.compile, "
    "so the guard made no difference there (#5037)"
)


class TestHausdorffLoss(BaseTester):
    @pytest.mark.parametrize("reduction", ["mean", "none", "sum"])
    @pytest.mark.parametrize(
        "hd,shape", [(kornia.losses.HausdorffERLoss, (10, 10)), (kornia.losses.HausdorffERLoss3D, (10, 10, 10))]
    )
    def test_smoke_none(self, hd, shape, reduction, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, *shape, dtype=dtype, device=device)
        labels = (torch.rand(2, 1, *shape, dtype=dtype, device=device) * (num_classes - 1)).long()
        loss = hd(reduction=reduction)

        loss(logits, labels)

    @pytest.mark.parametrize("loss_cls", [kornia.losses.HausdorffERLoss, kornia.losses.HausdorffERLoss3D])
    def test_input_type_validation(self, loss_cls):
        with pytest.raises(TypeError) as errinf:
            loss_cls()(None, torch.zeros(1, 1, 1, 1, dtype=torch.long))
        assert "pred type is not a torch.Tensor" in str(errinf)

        with pytest.raises(TypeError) as errinf:
            loss_cls()(torch.zeros(1, 2, 1, 1), [1, 2, 3])
        assert "target type is not a torch.Tensor" in str(errinf)

    def test_integer_like_k_negative_alpha_and_zero_k(self, device, dtype):
        # The loss sums k erosions weighted by (i + 1) ** alpha: any integer type of k, k = 0 (an empty sum), alpha < 0,
        # alpha = -inf (only the first erosion counts) and a per-sample alpha tensor are valid.
        hd = kornia.losses.HausdorffERLoss
        # Two offset squares whose differences are 8 to 10 pixels wide, so every erosion up to k = 4 adds to the loss.
        labels = torch.zeros(2, 1, 32, 32, dtype=torch.long, device=device)
        labels[..., 2:22, 2:22] = 1
        fg = torch.zeros(2, 1, 32, 32, device=device, dtype=dtype)
        fg[..., 12:30, 12:30] = 1
        pred = torch.cat([1 - fg, fg], 1)

        expected = hd(k=3)(pred, labels)
        assert hd(k=4)(pred, labels) > expected > hd(k=2)(pred, labels) > 0
        for k in (np.int64(3), np.int32(3), torch.tensor(3), torch.tensor(3, dtype=torch.int32)):
            self.assert_close(hd(k=k)(pred, labels), expected, rtol=0, atol=0)
        self.assert_close(hd(k=0)(pred, labels), torch.zeros_like(expected), rtol=0, atol=0)
        assert 0 < hd(k=3, alpha=-1.0)(pred, labels) < expected
        self.assert_close(hd(alpha=float("-inf"))(pred, labels), hd(k=1)(pred, labels), rtol=0, atol=0)

        alpha = torch.tensor([2.0, -1.0], device=device, dtype=dtype).view(2, 1, 1, 1)
        per_sample = hd(k=3, alpha=alpha, reduction="none")(pred, labels)
        self.assert_close(per_sample[:, :1], hd(k=3, alpha=2.0, reduction="none")(pred, labels)[:, :1])
        self.assert_close(per_sample[:, 1:], hd(k=3, alpha=-1.0, reduction="none")(pred, labels)[:, 1:])

    @pytest.mark.parametrize("reduction", ["mean", "none", "sum"])
    @pytest.mark.parametrize(
        "hd,shape", [(kornia.losses.HausdorffERLoss, (4, 4)), (kornia.losses.HausdorffERLoss3D, (3, 4, 4))]
    )
    def test_empty_batch(self, hd, shape, reduction, device, dtype):
        pred = torch.empty(0, 2, *shape, device=device, dtype=dtype)
        target = torch.empty(0, 1, *shape, device=device, dtype=torch.long)

        actual = hd(reduction=reduction)(pred, target)

        if reduction == "none":
            assert actual.shape == (2, 0, 1, *shape)
        elif reduction == "sum":
            self.assert_close(actual, torch.tensor(0.0, device=device, dtype=dtype))
        else:
            assert actual.ndim == 0
            assert torch.isnan(actual)

    @pytest.mark.parametrize(
        "hd,shape", [(kornia.losses.HausdorffERLoss, (4, 4)), (kornia.losses.HausdorffERLoss3D, (3, 4, 4))]
    )
    @pytest.mark.parametrize("label", [-1, 2])
    def test_nonempty_target_out_of_range(self, hd, shape, label, device, dtype):
        pred = torch.rand(1, 2, *shape, device=device, dtype=dtype)
        target = torch.full((1, 1, *shape), label, device=device, dtype=torch.long)

        with pytest.raises(ValueError, match="Expect long type target value in range"):
            hd()(pred, target)

    def test_exception_2d(self):

        with pytest.raises(ValueError) as errinf:
            kornia.losses.HausdorffERLoss()(
                torch.zeros(1, 2, 1, 1), torch.zeros(1, 1, 1, 1, dtype=torch.long, device="meta")
            )
        assert "Device mismatch" in str(errinf)

        with pytest.raises(TypeError) as errinf:
            kornia.losses.HausdorffERLoss(k=1.5)
        assert "k must be an integer" in str(errinf)

        with pytest.raises(TypeError) as errinf:
            kornia.losses.HausdorffERLoss(k="10")
        assert "k must be an integer" in str(errinf)

        for k in (True, False, np.bool_(True), torch.tensor(True), torch.tensor(False)):
            with pytest.raises(TypeError) as errinf:
                kornia.losses.HausdorffERLoss(k=k)
            assert "k must be an integer" in str(errinf)

        for k in (-1, torch.tensor(-1)):
            with pytest.raises(ValueError) as errinf:
                kornia.losses.HausdorffERLoss(k=k)
            assert "k must be non-negative" in str(errinf)

        for alpha in (float("nan"), float("inf"), torch.tensor(float("nan")), torch.tensor([2.0, float("inf")])):
            with pytest.raises(ValueError) as errinf:
                kornia.losses.HausdorffERLoss(alpha=alpha)
            assert "alpha must not be NaN or +inf" in str(errinf)

        with pytest.raises(NotImplementedError) as errinf:
            kornia.losses.HausdorffERLoss(reduction="invalid")
        assert "reduction `invalid` has not been implemented yet" in str(errinf)

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

        # The 3-D loss validates its target like the 2-D one: long dtype and labels in [0, C) (#5545).
        with pytest.raises(ValueError) as errinf:
            kornia.losses.HausdorffERLoss3D()(
                (torch.rand(1, 2, 1, 1, 1) > 0.5) * 1, torch.tensor([[[[[1]]]]], dtype=torch.float32)
            )
        assert "Expect long type target value in range [0, 2). Got torch.float32." in str(errinf)

        for label in (-1, 2, 5):
            with pytest.raises(ValueError) as errinf:
                kornia.losses.HausdorffERLoss3D()(torch.rand(1, 2, 1, 1, 1), torch.tensor([[[[[label]]]]]))
            assert f"Expect long type target value in range [0, 2). ({label}, {label})" in str(errinf)

    def test_numeric(self, device, dtype):
        num_classes = 3
        shape = (50, 50)
        hd = kornia.losses.HausdorffERLoss
        generator = torch.Generator().manual_seed(0)
        logits = torch.rand(2, num_classes, *shape, generator=generator).to(device, dtype)
        labels = (torch.rand(2, 1, *shape, generator=generator) * (num_classes - 1)).long().to(device)
        loss = hd(k=10)

        # The reference, in float64 with SciPy, for each class c and image b, with the cross structure
        # cross = scipy.ndimage.generate_binary_structure(ndim, 1); the loss is the mean of `out` over classes, images
        # and pixels:
        #   e = np.maximum(((pred[b, c] - (target[b, 0] == c)) ** 2 - 0.25) / 0.75, 0)
        #   m, w = e > 0, 0
        #   for i in range(1, k + 1):
        #       m = scipy.ndimage.binary_erosion(m, structure=cross, border_value=1)
        #       w += i ** alpha * m
        #   out[b, c] = e * w
        expected = torch.tensor(0.01620712955727422, device=device, dtype=dtype)

        actual = loss(logits, labels)
        self.assert_close(actual, expected)

    def test_numeric_every_erosion_counts(self, device, dtype):
        # Random inputs as in test_numeric erode to zero after a few steps. These mismatches are 24 to 26 pixels wide
        # and stay nonzero through eleven erosions: k = 9 gives 17.93455078125 and k = 11 gives 22.183046875.
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("the weighted sum of ten erosions is not exact in half precision")
        target = torch.zeros(2, 1, 64, 64, dtype=torch.long, device=device)
        target[0, :, 6:58, 6:58] = 1  # a 52 x 52 object
        fg = torch.zeros(2, 64, 64, device=device, dtype=dtype)
        fg[0, 6:58, 6:32] = 0.9  # predicted over the left half of the object only
        fg[1, 8:32, 30:54] = 0.8  # a 24 x 24 false positive
        pred = torch.stack([1 - fg, fg], 1)

        # The SciPy reference of test_numeric.
        expected = torch.tensor(20.37986328125, device=device, dtype=dtype)
        self.assert_close(kornia.losses.HausdorffERLoss(k=10)(pred, target), expected)

    def test_numeric_3d(self, device, dtype):
        num_classes = 3
        shape = (20, 20, 20)
        hd = kornia.losses.HausdorffERLoss3D
        generator = torch.Generator().manual_seed(0)
        logits = torch.rand(2, num_classes, *shape, generator=generator).to(device, dtype)
        labels = (torch.rand(2, 1, *shape, generator=generator) * (num_classes - 1)).long().to(device)
        loss = hd(k=10)

        # The SciPy reference of test_numeric.
        expected = torch.tensor(0.004341368957147833, device=device, dtype=dtype)
        actual = loss(logits, labels)
        self.assert_close(actual, expected)

    def test_get_kernel_override_sets_the_structuring_element(self, device, dtype):
        # Only the nonzero entries of get_kernel count, as the window of the minimum; an override that takes no
        # arguments still works. A graded diamond erodes differently under the 3 x 3 square and the default cross.
        class Square(kornia.losses.HausdorffERLoss):
            def get_kernel(self):
                return torch.ones(1, 1, 3, 3) / 9

        i, j = torch.meshgrid(torch.arange(11.0), torch.arange(11.0), indexing="ij")
        fg = (1 - ((i - 5).abs() + (j - 5).abs()) / 8).clamp(0, 1).to(device, dtype)
        pred = torch.stack([1 - fg, fg])[None]
        target = torch.zeros(1, 1, 11, 11, dtype=torch.long, device=device)
        # The reference of test_numeric with structure=np.ones((3, 3), bool) and with the cross, k = 4.
        self.assert_close(Square(k=4)(pred, target), torch.tensor(0.030991735537190084, device=device, dtype=dtype))
        self.assert_close(
            kornia.losses.HausdorffERLoss(k=4)(pred, target),
            torch.tensor(0.2568870523415978, device=device, dtype=dtype),
        )

    @pytest.mark.parametrize("p_true", [0.45, 0.35, 0.3, 0.28])
    def test_misclassified_band_has_a_loss_and_a_gradient(self, p_true, device, dtype):
        # #6676: a two-class prediction wrong at every pixel with a true-class probability between 0.29 and 0.5 gave
        # zero loss and zero gradient. A uniform error survives every erosion (the border is not an edge), so each
        # pixel adds 385 e, with the weight sum_i i ** 2 = 385 and e = ((1 - p_true) ** 2 - 1 / 4) / (3 / 4), and gets
        # the gradient 385 * 2 (p - q) / (3 / 4) over the 512 elements of the mean.
        generator = torch.Generator().manual_seed(0)
        target = torch.randint(2, (1, 1, 16, 16), generator=generator).to(device)
        onehot = torch.cat([target == 0, target == 1], 1).to(dtype)
        pred = (onehot * p_true + (1 - onehot) * (1 - p_true)).requires_grad_()

        loss = kornia.losses.HausdorffERLoss()(pred, target)
        loss.backward()

        # The probabilities as stored: in half precision 1 - p_true is rounded too.
        wrong = (1 - torch.tensor(p_true, dtype=dtype)).item()
        expected = torch.tensor(385 * (wrong**2 - 0.25) / 0.75, device=device, dtype=dtype)
        expected_grad = 385 * 2 * (pred.detach() - onehot) / 0.75 / pred.numel()
        # (p - q) ** 2 - 1/4 cancels: at p_true = 0.45, 0.3025 - 0.25 turns bfloat16's rounding of the square into a
        # 1.4 % error of the loss.
        tolerance = {"rtol": 2e-2, "atol": 0.0} if dtype in (torch.float16, torch.bfloat16) else {}
        self.assert_close(loss, expected, **tolerance)
        self.assert_close(pred.grad, expected_grad)

    @pytest.mark.parametrize("p_true", [0.5, 0.7, 0.99, 1.0])
    def test_correct_prediction_scores_zero(self, p_true, device, dtype):
        # Thresholding at 0.5 gets every pixel right, so nothing counts, however uncertain; exactly 0.5 counts as
        # correct.
        target = torch.zeros(1, 1, 8, 8, dtype=torch.long, device=device)
        target[..., 2:6, 2:6] = 1
        onehot = torch.cat([target == 0, target == 1], 1).to(dtype)
        pred = (onehot * p_true + (1 - onehot) * (1 - p_true)).requires_grad_()
        loss = kornia.losses.HausdorffERLoss()(pred, target)
        loss.backward()
        self.assert_close(loss, torch.zeros_like(loss), rtol=0, atol=0)
        self.assert_close(pred.grad, torch.zeros_like(pred.grad), rtol=0, atol=0)

    def test_gradient_reaches_every_misclassified_pixel_by_its_depth(self, device, dtype):
        # The depth weight is a constant, so d loss / d p = weight * 2 (p - q) / (3 / 4) at every misclassified pixel,
        # not only at the minimum of each erosion window. A 5 x 7 error against the left border, predicted at 0.8: the
        # border is not an edge, so a pixel's depth d is 1 plus its distance to the nearest of the other three sides,
        # and its weight is sum_{i=1}^{d - 1} i ** 2. Both channels are wrong there, with e = (0.8 ** 2 - 1/4) / (3/4).
        weight = torch.zeros(9, 11, dtype=torch.float64)
        weight[3] = weight[5] = torch.tensor([1, 1, 1, 1, 1, 1, 0] + [0] * 4)
        weight[4] = torch.tensor([5, 5, 5, 5, 5, 1, 0] + [0] * 4)
        fg = torch.zeros(1, 9, 11, device=device, dtype=dtype)
        fg[:, 2:7, 0:7] = 0.8
        pred = torch.stack([1 - fg, fg], 1).requires_grad_()
        target = torch.zeros(1, 1, 9, 11, dtype=torch.long, device=device)

        loss = kornia.losses.HausdorffERLoss(reduction="sum")(pred, target)
        loss.backward()

        weight = weight.to(device, dtype)
        e = (0.8**2 - 0.25) / 0.75
        self.assert_close(loss, 2 * e * weight.sum())
        onehot = torch.stack([torch.ones_like(weight), torch.zeros_like(weight)])[None]
        self.assert_close(pred.grad, weight * 2 * (pred.detach() - onehot) / 0.75)

    @pytest.mark.parametrize(
        "hd,shape", [(kornia.losses.HausdorffERLoss, (5, 5)), (kornia.losses.HausdorffERLoss3D, (5, 5, 5))]
    )
    def test_gradcheck(self, hd, shape, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, *shape, device=device)
        labels = (torch.rand(2, 1, *shape, device=device) * (num_classes - 1)).long()
        loss = hd(k=2)

        self.gradcheck(loss, (logits, labels), dtypes=[torch.float64, torch.int64])

    @pytest.mark.skipif(not dynamo_is_available() or torch_version_lt(2, 14, 0), reason=_CAPTURE_UNAVAILABLE_REASON)
    @pytest.mark.parametrize(
        "hd,shape", [(kornia.losses.HausdorffERLoss, (10, 10)), (kornia.losses.HausdorffERLoss3D, (5, 5, 5))]
    )
    def test_trace_labels_like_eager(self, hd, shape, device, dtype):
        # The range check reads the data, so any capture has to skip it, not only export (#5037).
        from torch._dynamo.testing import CompileCounter

        num_classes = 3
        logits = torch.rand(2, num_classes, *shape, device=device, dtype=dtype)
        labels = (torch.rand(2, 1, *shape, device=device) * (num_classes - 1)).long()
        counter = CompileCounter()
        torch._dynamo.reset()
        traced = torch.compile(hd(), backend=counter, fullgraph=True)
        self.assert_close(traced(logits, labels), hd()(logits, labels), rtol=0, atol=0)
        assert counter.frame_count == 1

    @pytest.mark.skipif(not dynamo_is_available() or torch_version_lt(2, 14, 0), reason=_CAPTURE_UNAVAILABLE_REASON)
    def test_trace_scores_out_of_range_like_export(self, device, dtype):
        # Export already scored an out-of-range label in no class, and a traced call now does the same instead of
        # raising; that is the behaviour change #5037 accepts. An eager call keeps raising.
        num_classes = 3
        logits = torch.rand(2, num_classes, 10, 10, device=device, dtype=dtype)
        labels = torch.full((2, 1, 10, 10), num_classes, dtype=torch.long, device=device)
        with pytest.raises(ValueError, match="Expect long type target value in range"):
            kornia.losses.HausdorffERLoss()(logits, labels)
        exported = torch.export.export(kornia.losses.HausdorffERLoss(), (logits, labels)).module()(logits, labels)
        torch._dynamo.reset()
        traced = torch.compile(kornia.losses.HausdorffERLoss(), backend="eager", fullgraph=True)(logits, labels)
        self.assert_close(traced, exported, rtol=0, atol=0)


class TestConventionsHausdorffERLoss(BaseTester):
    """Pins for the input form, output layout and per-image scoring of the Hausdorff erosion losses."""

    @pytest.mark.parametrize(
        "hd,spatial", [(kornia.losses.HausdorffERLoss, (9, 13)), (kornia.losses.HausdorffERLoss3D, (5, 7, 9))]
    )
    def test_convention_hausdorff_er_loss_none_is_class_first_and_channel_c_scores_label_c(
        self, hd, spatial, device, dtype
    ):
        # pred holds per-class probabilities and nothing is applied to it; channel c is compared with target == c,
        # class 0 included, and reduction='none' returns (C, B, 1, *spatial), class axis first; the default 'mean' is
        # the plain mean of that tensor
        labels = torch.zeros(2, 1, *spatial, dtype=torch.long, device=device)
        labels[0, 0, ..., 1:4, 1:4] = 1
        labels[0, 0, ..., 3:6, 5:8] = 2
        labels[1, 0, ..., 2:6, 4:8] = 1
        labels[1, 0, ..., 0:2, 0:3] = 2
        pred = (labels == torch.arange(3, device=device).view(1, 3, *[1] * len(spatial))).to(dtype)
        none = hd(reduction="none")
        self.assert_close(none(pred, labels), torch.zeros(3, 2, 1, *spatial, device=device, dtype=dtype))
        pred[1, 0, ..., 3:6, 0:4] = 1 - pred[1, 0, ..., 3:6, 0:4]  # a 3 x 4 error in class 0 of image 1
        pred[0, 2, ..., 0:3, 5:8] = 1 - pred[0, 2, ..., 0:3, 5:8]  # a 3 x 3 error in class 2 of image 0
        out = none(pred, labels)
        assert out.shape == (3, 2, 1, *spatial)
        assert (out.flatten(2).sum(-1) > 0).tolist() == [[False, True], [False, False], [True, False]]
        self.assert_close(hd()(pred, labels), out.mean())

    @pytest.mark.parametrize(
        "hd,spatial", [(kornia.losses.HausdorffERLoss, (9, 13)), (kornia.losses.HausdorffERLoss3D, (5, 7, 9))]
    )
    def test_convention_hausdorff_er_loss_scores_each_image_on_its_own(self, hd, spatial, device, dtype):
        # The erosion is local to each image and class, so an image's 'none' slice does not depend on the other images
        # of the batch: a thin error (image 0) and a thick one (image 1), scored together and alone
        labels = torch.zeros(2, 1, *spatial, dtype=torch.long, device=device)
        labels[0, 0, ..., 1:4, 1:4] = 1
        labels[1, 0, ..., 2:6, 4:8] = 1
        pred = (labels == torch.arange(2, device=device).view(1, 2, *[1] * len(spatial))).to(dtype)
        pred[0, 1, ..., 2:3, 1:4] = 0.0  # image 0 misses one row of its object: a thin error
        pred[1, 1, ..., 2:6, 4:8] = 0.0  # image 1 misses its whole object
        none = hd(reduction="none")
        batch = none(pred, labels)
        self.assert_close(batch[:, :1], none(pred[:1], labels[:1]))
        self.assert_close(batch[:, 1:], none(pred[1:], labels[1:]))

    @pytest.mark.parametrize(
        "hd,spatial,box,expected",
        [
            # A 5 x 7 error against the left border: the border is not an edge, so a pixel's depth d is 1 plus its
            # distance to the nearest of the other three sides. Rows at depth 1 add 0, the two rows at depth 2 add 1
            # on six pixels, and the middle row adds 1 + 4 on five pixels and 1 on one: 12 + 26 = 38 per channel.
            (kornia.losses.HausdorffERLoss, (9, 11), (slice(2, 7), slice(0, 7)), 76.0),
            # A 5 x 5 x 7 error inside the volume: 42 voxels at depth 2 add 1 and 3 at depth 3 add 1 + 4, 57 per
            # channel.
            (kornia.losses.HausdorffERLoss3D, (7, 9, 11), (slice(1, 6), slice(2, 7), slice(2, 9)), 114.0),
        ],
    )
    def test_convention_hausdorff_er_loss_binary_error_scores_its_city_block_depth(
        self, hd, spatial, box, expected, device, dtype
    ):
        # For a binary prediction, a wrong pixel at city-block distance d from the nearest correct pixel adds
        # sum_{i=1}^{min(k, d - 1)} i ** alpha; both channels of a two-class prediction are wrong in the box.
        fg = torch.zeros(1, *spatial, device=device, dtype=dtype)
        fg[(slice(None), *box)] = 1
        pred = torch.stack([1 - fg, fg], 1)
        target = torch.zeros(1, 1, *spatial, dtype=torch.long, device=device)
        self.assert_close(hd(reduction="sum")(pred, target), torch.tensor(expected, device=device, dtype=dtype))
