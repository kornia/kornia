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

from __future__ import annotations

import pytest
import torch

import kornia
from kornia.core.exceptions import BaseError, DeviceError, ShapeError, TypeCheckError
from kornia.metrics import accuracy

from testing.base import BaseTester, supports_topk


class TestAccuracy:
    def test_top1_perfect(self):
        logits = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        target = torch.tensor([[1], [0]])
        result = accuracy(logits, target, topk=(1,))
        assert len(result) == 1
        assert result[0].item() == 100.0

    def test_top1_zero(self):
        logits = torch.tensor([[1.0, 0.0, 0.0]])
        target = torch.tensor([[2]])
        result = accuracy(logits, target, topk=(1,))
        assert result[0].item() == 0.0

    def test_topk(self):
        # 3 classes; true class is index 2 (lowest logit) — wrong for both top-1 and top-2
        logits = torch.tensor([[0.3, 0.5, 0.2]])
        target = torch.tensor([[2]])  # true class is index 2 (third highest)
        top1, top2 = accuracy(logits, target, topk=(1, 2))
        assert top1.item() == 0.0
        assert top2.item() == 0.0

        logits2 = torch.tensor([[0.3, 0.2, 0.5]])
        target2 = torch.tensor([[2]])
        top1b, _ = accuracy(logits2, target2, topk=(1, 2))
        assert top1b.item() == 100.0

    def test_topk_exceeds_num_classes_is_clipped(self):
        # topk=5 but only 3 classes — should not crash
        logits = torch.tensor([[0.1, 0.8, 0.1]])
        target = torch.tensor([[1]])
        result = accuracy(logits, target, topk=(5,))
        assert result[0].item() == 100.0

    @pytest.mark.parametrize(("pred_batch", "target_batch"), [(1, 2), (2, 1), (2, 3), (3, 2)])
    def test_mismatched_batch(self, pred_batch, target_batch):
        """Reject mismatched batch sizes between pred and target."""
        logits = torch.zeros(pred_batch, 3)
        target = torch.zeros(target_batch, dtype=torch.int64)
        with pytest.raises(ShapeError):
            accuracy(logits, target)

    def test_singleton_target_batch_mismatch_does_not_broadcast(self):
        """Ensure singleton target batch size is not silently broadcast across pred batch."""
        logits = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        target = torch.tensor([1])
        with pytest.raises(ShapeError):
            accuracy(logits, target)

    @pytest.mark.parametrize("pred_shape", [(3,), (2, 3, 4)])
    def test_invalid_pred_shape(self, pred_shape):
        """Reject pred tensors that are not 2-dimensional (B, C)."""
        logits = torch.zeros(pred_shape)
        target = torch.zeros(logits.shape[0] if logits.ndim > 0 else 1, dtype=torch.int64)
        with pytest.raises(ShapeError):
            accuracy(logits, target)

    @pytest.mark.parametrize("target_shape", [(2, 2), (2, 1, 1), ()])
    def test_invalid_target_shape(self, target_shape):
        """Reject target tensors that are not (B,) or (B, 1)."""
        logits = torch.zeros(2, 3)
        target = torch.zeros(target_shape, dtype=torch.int64)
        with pytest.raises(ShapeError):
            accuracy(logits, target)

    def test_non_tensor_inputs(self):
        """Reject non-tensor pred and target inputs."""
        with pytest.raises(TypeCheckError):
            accuracy([[0.0, 1.0]], torch.tensor([1]))  # type: ignore[arg-type]
        with pytest.raises(TypeCheckError):
            accuracy(torch.tensor([[0.0, 1.0]]), [1])  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_topk", [(), (0,), (-1,), (1, 0), (1.5,), (True,)])
    def test_invalid_topk(self, invalid_topk):
        """Reject empty, non-positive, or non-integer topk sequences."""
        logits = torch.tensor([[0.0, 1.0]])
        target = torch.tensor([1])
        with pytest.raises(BaseError, match="topk must be a non-empty sequence of positive integers"):
            accuracy(logits, target, topk=invalid_topk)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_mismatch(self):
        """Reject pred and target tensors located on different devices."""
        logits = torch.zeros(2, 3, device="cpu")
        target = torch.zeros(2, dtype=torch.int64, device="cuda")
        with pytest.raises(DeviceError):
            accuracy(logits, target)


class TestConventionsAccuracy(BaseTester):
    def test_convention_accuracy_is_a_percentage_per_topk(self, device, dtype):
        """accuracy returns a list of 0-d float32 percentages, one per entry of topk and in its order."""
        if not supports_topk(device, dtype):
            pytest.skip(f"this torch build has no topk kernel for {dtype} on {device.type}")
        # Four samples, three classes. The target is the 1st, 3rd, 1st and 2nd highest score, so top-1 hits 2 of 4,
        # top-2 hits 3 and top-3 hits 4. scikit-learn 1.9.0 gives the fractions: accuracy_score -> 0.5,
        # top_k_accuracy_score(k=2) -> 0.75; kornia reports them as percentages.
        logits = torch.tensor(
            [[0.1, 0.7, 0.2], [0.6, 0.3, 0.1], [0.2, 0.3, 0.5], [0.5, 0.1, 0.4]], device=device, dtype=dtype
        )
        target = torch.tensor([1, 2, 2, 2], device=device)
        out = kornia.metrics.accuracy(logits, target, topk=(1, 2, 3, 5))
        assert isinstance(out, list)
        assert len(out) == 4
        assert all(value.shape == () and value.dtype == torch.float32 for value in out)
        # k larger than the number of classes is clipped to it, so top-5 of 3 classes is 100
        self.assert_close(torch.stack(out), torch.tensor([50.0, 75.0, 100.0, 100.0], device=device))
        # the list follows the order of topk; a (B, 1) target is read like a (B,) one
        swapped = kornia.metrics.accuracy(logits, target[:, None], topk=(2, 1))
        self.assert_close(torch.stack(swapped), torch.tensor([75.0, 50.0], device=device))
        # relabel: renaming class c to perm[c] in the scores and the targets alike leaves the result unchanged
        perm = torch.tensor([2, 0, 1], device=device)
        relabelled = torch.empty_like(logits)
        relabelled[:, perm] = logits
        out_relabelled = kornia.metrics.accuracy(relabelled, perm[target], topk=(1, 2))
        self.assert_close(torch.stack(out_relabelled), torch.tensor([50.0, 75.0], device=device))
