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
        "hd,shape", [(kornia.losses.HausdorffERLoss, (10, 10)), (kornia.losses.HausdorffERLoss3D, (10, 10, 10))]
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

    def test_numeric(self, device, dtype, cudnn_tf32_follows_option):
        # `cudnn_tf32_follows_option` keeps CUDA's float32 `conv2d` out of TF32: rounding its inputs to 10 mantissa bits
        # moves the value by 3e-5 to 8e-5, above the float32 tolerance.
        num_classes = 3
        shape = (50, 50)
        hd = kornia.losses.HausdorffERLoss
        generator = torch.Generator().manual_seed(0)
        logits = torch.rand(2, num_classes, *shape, generator=generator).to(device, dtype)
        labels = (torch.rand(2, 1, *shape, generator=generator) * (num_classes - 1)).long().to(device)
        loss = hd(k=10)

        # The value at this seed. Unseeded draws spread from 0.0205 to 0.0316 around the old constant 0.025 (#5577).
        expected = torch.tensor(0.0265066, device=device, dtype=dtype)

        actual = loss(logits, labels)
        self.assert_close(actual, expected)

    def test_numeric_every_erosion_counts(self, device, dtype, cudnn_tf32_follows_option):
        # Random inputs as in test_numeric erode to zero after two steps, so that value is the same for any k >= 2.
        # These mismatches are 10 to 20 pixels wide and stay nonzero through all ten erosions: k = 9 gives 11.567 and
        # k = 11 gives 13.223.
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("the weighted sum of ten erosions is 0.3 % (float16) to 1.2 % (bfloat16) off in half precision")
        target = torch.zeros(2, 1, 32, 32, dtype=torch.long, device=device)
        target[0, :, 6:26, 6:26] = 1  # a 20 x 20 object
        fg = torch.zeros(2, 32, 32, device=device, dtype=dtype)
        fg[0, 6:26, 6:16] = 0.9  # predicted over the left half of the object only
        fg[1, 4:20, 10:26] = 0.8  # a 16 x 16 false positive
        pred = torch.stack([1 - fg, fg], 1)

        # The reference algorithm (PatRyg99/HausdorffLoss, HausdorffERLoss.perform_erosion) in float64 with SciPy, for
        # each class c and image b, with cross = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]; the loss is the mean of `eroded`
        # over classes, images and pixels:
        #   bound, eroded = (pred[b, c] - (target[b, 0] == c)) ** 2, 0
        #   for k in range(10):
        #       e = np.maximum(scipy.ndimage.convolve(bound, 0.2 * cross, mode="constant") - 0.5, 0)
        #       e = (e - e.min()) / np.ptp(e) if np.ptp(e) else e
        #       eroded += e * (k + 1) ** 2
        #       bound = e
        expected = torch.tensor(12.540735900430166, device=device, dtype=dtype)
        self.assert_close(kornia.losses.HausdorffERLoss(k=10)(pred, target), expected)

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
        "hd,shape", [(kornia.losses.HausdorffERLoss, (8, 8)), (kornia.losses.HausdorffERLoss3D, (6, 6, 6))]
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
        "hd,shape", [(kornia.losses.HausdorffERLoss, (5, 5)), (kornia.losses.HausdorffERLoss3D, (5, 5, 5))]
    )
    def test_gradcheck(self, hd, shape, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, *shape, device=device)
        labels = (torch.rand(2, 1, *shape, device=device) * (num_classes - 1)).long()
        loss = hd(k=2)

        self.gradcheck(loss, (logits, labels), dtypes=[torch.float64, torch.int64])


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
        # The erosions are min-max normalised per image and class, so an image's 'none' slice does not depend on the
        # other images of the batch: a thin error (image 0) and a thick one (image 1), scored together and alone
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
