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

import copy
import pickle

import pytest
import torch

import kornia.augmentation as K
from kornia.core.exceptions import BaseError

from testing.base import BaseTester


def _labelled_batch(batch: int = 4, spatial: tuple[int, ...] = (4, 6), channels: int = 2):
    """Each image is filled with its own index and its mask carries its own label, so the donor is visible."""
    mask = torch.stack([torch.full(spatial, i + 1, dtype=torch.long) for i in range(batch)])
    image = torch.stack([torch.full((channels, *spatial), float(i + 1)) for i in range(batch)])
    return image, mask


class TestTransplantationConventions(BaseTester):
    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("n_spatial", [0, 1, 2, 3])
    def test_convention_any_spatial_rank_is_accepted(self, n_spatial):
        spatial = (4, 5, 6)[:n_spatial]
        image = torch.rand(3, 2, *spatial)
        mask = torch.randint(0, 3, (3, *spatial))
        out_image, out_mask = K.RandomTransplantation(p=1.0)(image, mask)
        assert out_image.shape == image.shape and out_mask.shape == mask.shape

    @pytest.mark.device_agnostic
    def test_convention_first_axis_is_always_the_batch(self):
        # There is no (C, H, W) form: the leading axis is read as the batch, so the ranks disagree.
        with pytest.raises(BaseError, match="must match except for the channel dimension"):
            K.RandomTransplantation(p=1.0)(torch.rand(2, 4, 5), torch.randint(0, 3, (4, 5)))

    @pytest.mark.device_agnostic
    def test_convention_image_rank_must_be_mask_rank_plus_one(self):
        mask = torch.randint(0, 3, (3, 4, 5))
        with pytest.raises(BaseError, match="one additional dimension"):
            K.RandomTransplantation(p=1.0)(torch.rand(3, 4, 5), mask)
        with pytest.raises(BaseError, match="one additional dimension"):
            K.RandomTransplantation(p=1.0)(torch.rand(3, 2, 4, 5, 6), mask)

    @pytest.mark.device_agnostic
    def test_convention_gate_is_per_sample_and_p_batch_is_call_wide(self):
        aug = K.RandomTransplantation(p=0.5)
        subsets = 0
        for _ in range(40):
            gate = aug.forward_parameters(torch.Size([8, 4, 5]))["batch_prob"] > 0.5
            subsets += 0 < int(gate.sum()) < 8
        assert subsets > 0  # a batch-wide gate can never select a strict subset
        closed = K.RandomTransplantation(p=1.0, p_batch=0.0)
        for _ in range(8):
            assert not bool((closed.forward_parameters(torch.Size([8, 4, 5]))["batch_prob"] > 0.5).any())

    @pytest.mark.device_agnostic
    def test_convention_donor_is_the_previous_image_of_the_full_batch(self):
        image, mask = _labelled_batch(batch=5)
        torch.manual_seed(0)
        aug = K.RandomTransplantation(p=1.0)
        _, out_mask = aug(image, mask)
        # Every image is one label, so the whole mask row takes the donor's label.
        observed = [int(out_mask[i].flatten()[0]) for i in range(5)]
        assert observed == [(i - 1) % 5 + 1 for i in range(5)]
        assert observed != [(i + 1) % 5 + 1 for i in range(5)]  # not the next image
        assert observed != [i + 1 for i in range(5)]  # not the identity
        assert aug._params["donor_indices"].tolist() == [4, 0, 1, 2, 3]

    @pytest.mark.device_agnostic
    def test_convention_donor_need_not_be_an_acceptor(self):
        aug = K.RandomTransplantation(p=0.5)
        for seed in range(40):
            torch.manual_seed(seed)
            params = aug.forward_parameters(torch.Size([5, 4, 5]))
            acceptors = torch.where(params["batch_prob"] > 0.5)[0]
            if acceptors.numel() and acceptors.numel() < 5:
                donors = (acceptors - 1) % 5
                if not set(donors.tolist()) <= set(acceptors.tolist()):
                    return
        pytest.fail("no draw produced a donor outside the acceptor set")

    @pytest.mark.device_agnostic
    def test_convention_single_image_batch_is_its_own_donor_and_an_identity(self):
        image, mask = _labelled_batch(batch=1)
        for seed in range(4):
            torch.manual_seed(seed)
            aug = K.RandomTransplantation(p=1.0)
            out_image, out_mask = aug(image, mask)
            assert aug._params["donor_indices"].tolist() == aug._params["acceptor_indices"].tolist() == [0]
            self.assert_close(out_image, image, rtol=0, atol=0)
            assert torch.equal(out_mask, mask)

    @pytest.mark.device_agnostic
    def test_convention_label_draw_is_uniform_over_distinct_labels_not_area(self):
        # The donor holds label 1 on a single position and label 2 on the rest: area-weighted sampling
        # would pick 2 almost always.
        mask = torch.full((2, 8, 8), 2, dtype=torch.long)
        mask[0, 0, 0] = 1
        image = torch.rand(2, 1, 8, 8)
        drawn = []
        for seed in range(200):
            torch.manual_seed(seed)
            aug = K.RandomTransplantation(p=1.0)
            aug(image, mask)
            drawn.append(int(aug._params["selected_labels"][1]))  # acceptor 1's donor is image 0
        share = drawn.count(1) / len(drawn)
        assert 0.3 < share < 0.7

    @pytest.mark.device_agnostic
    def test_convention_excluded_label_is_never_transplanted(self):
        mask = torch.zeros(3, 4, 6, dtype=torch.long)
        for i in range(3):
            mask[i, :2, :3] = i + 1
        image = torch.rand(3, 1, 4, 6)
        torch.manual_seed(0)
        aug = K.RandomTransplantation(p=1.0, excluded_labels=[0])
        _, out_mask = aug(image, mask)
        assert aug._params["selected_labels"].tolist() == [3, 1, 2]
        assert int((out_mask == 0).sum()) == int((mask == 0).sum())

    @pytest.mark.device_agnostic
    def test_convention_a_donor_without_an_eligible_label_only_skips_its_own_acceptor(self):
        # Donor image 1 holds nothing but the excluded label. Its acceptor gets nothing; the others still
        # receive the region their OWN donor supplied.
        mask = torch.full((3, 4, 6), 9, dtype=torch.long)
        mask[0] = 5
        mask[0, :2, :3] = 7
        mask[1] = 7
        image = torch.stack([torch.full((2, 4, 6), float(i + 1)) for i in range(3)])
        torch.manual_seed(0)
        aug = K.RandomTransplantation(p=1.0, excluded_labels=[9])
        aug(image, mask)
        params = aug._params
        assert len(params["selected_labels"]) == len(params["acceptor_indices"]) == 3
        for d in range(3):
            donor_mask = mask[int(params["donor_indices"][d])]
            assert torch.equal(params["selection"][d], donor_mask == int(params["selected_labels"][d]))
        assert int(params["selection"][0].sum()) == 0  # donor 2 is entirely the excluded label
        assert int(params["selection"][1].sum()) > 0
        assert int(params["selection"][2].sum()) > 0

    @pytest.mark.device_agnostic
    def test_convention_every_donor_excluded_is_a_clean_no_op(self):
        mask = torch.zeros(3, 4, 6, dtype=torch.long)
        image = torch.rand(3, 1, 4, 6)
        aug = K.RandomTransplantation(p=1.0, excluded_labels=[0])
        out_image, out_mask = aug(image, mask)
        assert aug._params["selected_labels"].dtype == mask.dtype
        assert not bool(aug._params["selection"].any())
        self.assert_close(out_image, image, rtol=0, atol=0)
        assert torch.equal(out_mask, mask)

    @pytest.mark.device_agnostic
    def test_convention_params_keys_and_selection_semantics(self):
        image, mask = _labelled_batch(batch=4)
        torch.manual_seed(0)
        aug = K.RandomTransplantation(p=1.0)
        aug(image, mask)
        params = aug._params
        assert set(params) == {
            "batch_prob",
            "forward_input_shape",
            "acceptor_indices",
            "donor_indices",
            "selected_labels",
            "selection",
        }
        assert params["selection"].dtype is torch.bool
        assert params["selection"].shape == (4, 4, 6)
        # A hand-written selection controls exactly which positions move.
        replay = copy.deepcopy(dict(params))
        chosen = torch.zeros_like(replay["selection"])
        chosen[:, 0, 0] = True
        replay["selection"] = chosen
        out_image, _ = aug(image, mask, params=replay)
        donors = replay["donor_indices"]
        self.assert_close(out_image[:, 0, 0, 0], donors.to(image) + 1, rtol=0, atol=0)
        assert bool((out_image[:, 0, 1, 1] == torch.arange(1.0, 5.0)).all())  # everything else is untouched

    @pytest.mark.device_agnostic
    def test_convention_replay_reproduces_and_selected_labels_alone_is_inert(self):
        image, mask = _labelled_batch(batch=4)
        torch.manual_seed(0)
        aug = K.RandomTransplantation(p=1.0)
        expected_image, expected_mask = aug(image, mask)
        full = copy.deepcopy(dict(aug._params))
        torch.manual_seed(999)
        replayed_image, replayed_mask = aug(image, mask, params=copy.deepcopy(full))
        self.assert_close(replayed_image, expected_image, rtol=0, atol=0)
        assert torch.equal(replayed_mask, expected_mask)
        # Dropping selected_labels alone changes nothing: it is not even recomputed.
        without_labels = {k: v for k, v in copy.deepcopy(full).items() if k != "selected_labels"}
        again_image, _ = aug(image, mask, params=without_labels)
        self.assert_close(again_image, expected_image, rtol=0, atol=0)
        assert "selected_labels" not in without_labels
        # Dropping both selected_labels and selection redraws from the RNG.
        redrawn = {k: v for k, v in copy.deepcopy(full).items() if k not in ("selected_labels", "selection")}
        torch.manual_seed(1)
        aug(image, mask, params=redrawn)
        assert "selected_labels" in redrawn

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_convention_image_dtype_guard_and_free_mask_dtype(self, p):
        mask = torch.randint(0, 3, (2, 4, 5))
        with pytest.raises(TypeError, match="float16"):
            K.RandomTransplantation(p=p)(torch.ones(2, 1, 4, 5, dtype=torch.int64), mask)
        for dtype in (torch.int64, torch.int32, torch.bool, torch.uint8, torch.float32):
            _, out_mask = K.RandomTransplantation(p=p)(torch.rand(2, 1, 4, 5), mask.to(dtype))
            assert out_mask.dtype is dtype

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("p", [0.0, 1.0])
    @pytest.mark.parametrize("key", ["bbox_xyxy", "keypoints", "class"])
    def test_convention_unsupported_key_raises_whatever_the_gate(self, p, key):
        image = torch.rand(2, 1, 8, 8)
        mask = torch.randint(0, 3, (2, 8, 8))
        annotations = {
            "bbox_xyxy": torch.tensor([[[1.0, 1.0, 4.0, 4.0]]] * 2),
            "keypoints": torch.rand(2, 2, 2),
            "class": torch.arange(2),
        }
        # The raise carries no message at all, so no `match=` is possible here.
        with pytest.raises(NotImplementedError):
            K.RandomTransplantation(p=p)(image, mask, annotations[key], data_keys=["input", "mask", key])

    @pytest.mark.device_agnostic
    def test_convention_a_mask_key_is_required(self):
        with pytest.raises(ValueError, match="MASK"):
            K.RandomTransplantation(p=1.0)(torch.rand(2, 1, 4, 5), data_keys=["input"])

    @pytest.mark.device_agnostic
    def test_convention_mask_only_call_returns_a_bare_tensor(self):
        _, mask = _labelled_batch(batch=3)
        out = K.RandomTransplantation(p=1.0)(mask, data_keys=["mask"])
        assert isinstance(out, torch.Tensor) and out.shape == mask.shape

    @pytest.mark.device_agnostic
    def test_convention_no_matrix_and_no_inverse(self):
        for cls in (K.RandomTransplantation, K.RandomTransplantation3D):
            aug = cls(p=1.0)
            with pytest.raises(RuntimeError, match="Transformation matrices"):
                _ = aug.transform_matrix
            with pytest.raises(RuntimeError, match="Inverse"):
                aug.inverse()

    @pytest.mark.device_agnostic
    def test_convention_selected_labels_length_is_validated(self):
        image, mask = _labelled_batch(batch=2)
        aug = K.RandomTransplantation(p=1.0)
        aug(image, mask)
        params = dict(aug._params)
        params["selected_labels"] = torch.tensor([1, 2, 1, 2])
        del params["selection"]
        with pytest.raises(BaseError, match="than images where this augmentation should be applied"):
            aug(image, mask, params=params)

    @pytest.mark.device_agnostic
    def test_convention_serialization_is_state_free_and_replays(self):
        image, mask = _labelled_batch(batch=3)
        torch.manual_seed(0)
        aug = K.RandomTransplantation(p=1.0, excluded_labels=[0])
        expected_image, _ = aug(image, mask)
        assert not aug.state_dict() and not list(aug.parameters()) and not list(aug.buffers())
        for restored in (pickle.loads(pickle.dumps(aug)), copy.deepcopy(aug)):  # noqa: S301
            assert torch.equal(restored.excluded_labels, aug.excluded_labels)
            replayed, _ = restored(image, mask, params=copy.deepcopy(dict(restored._params)))
            self.assert_close(replayed, expected_image, rtol=0, atol=0)

    @pytest.mark.device_agnostic
    def test_convention_3d_subclass_matches_the_2d_class_on_a_direct_call(self):
        image, mask = _labelled_batch(batch=3, spatial=(3, 4, 6))
        torch.manual_seed(42)
        flat_image, flat_mask = K.RandomTransplantation(p=1.0)(image, mask)
        torch.manual_seed(42)
        volume_image, volume_mask = K.RandomTransplantation3D(p=1.0)(image, mask)
        self.assert_close(flat_image, volume_image, rtol=0, atol=0)
        assert torch.equal(flat_mask, volume_mask)

    @pytest.mark.device_agnostic
    def test_wart_container_dispatch_reports_only_one_of_the_two_rank_mistakes_4692(self):
        image, mask = _labelled_batch(batch=3, spatial=(4, 6))
        volume, volume_mask = _labelled_batch(batch=3, spatial=(3, 4, 6))
        # The 2D class on a volume raises.
        with pytest.raises(RuntimeError, match="input shape expected to be in"):
            K.AugmentationSequential(K.RandomTransplantation(p=1.0), data_keys=["image", "mask"])(volume, volume_mask)
        # The 3D class on a 2D batch is a silent no-op: correct shapes, nothing moved.
        inner = K.RandomTransplantation3D(p=1.0)
        torch.manual_seed(7)
        out_image, out_mask = K.AugmentationSequential(inner, data_keys=["image", "mask"])(image, mask)
        assert out_image.shape == image.shape
        assert torch.equal(out_mask, mask)
        assert inner._params["batch_prob"].numel() == 1  # a one-element gate for a batch of 3
        assert inner._params["donor_indices"].tolist() == [0]

    @pytest.mark.device_agnostic
    def test_wart_container_inverse_returns_its_input_4693(self):
        image, mask = _labelled_batch(batch=3)
        container = K.AugmentationSequential(K.RandomTransplantation(p=1.0), data_keys=["image", "mask"])
        torch.manual_seed(7)
        out_image, out_mask = container(image, mask)
        assert not torch.equal(out_mask, mask)  # the forward really did transplant
        inverted_image, _ = container.inverse(out_image, out_mask)
        self.assert_close(inverted_image, out_image, rtol=0, atol=0)  # handed straight back
        assert not torch.equal(inverted_image, image)  # the original is not restored
