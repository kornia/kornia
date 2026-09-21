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


def _multi_label_batch(batch: int = 4, size: int = 6, labels: int = 5):
    """Every donor holds several labels, so a redraw is distinguishable from a replay; channels differ too."""
    generator = torch.Generator().manual_seed(1234)
    mask = torch.randint(0, labels, (batch, size, size + 1), generator=generator)
    image = torch.rand(batch, 3, size, size + 1, generator=generator)
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
    def test_convention_the_first_mask_drives_the_transplant(self):
        first = torch.stack([torch.full((4, 6), i + 1, dtype=torch.long) for i in range(3)])
        second = torch.zeros_like(first)
        second[:, 2:] = 1  # read as the driver, this one would move only its lower half
        aug = K.RandomTransplantation(p=1.0)
        out_first, out_second = aug(first, second, data_keys=["mask", "mask"])
        assert bool(aug._params["selection"].all())
        assert torch.equal(out_first, first.roll(1, dims=0))
        assert torch.equal(out_second, second)  # moved at the first mask's positions: identical rows, no change

    def test_convention_transplant_follows_the_device_and_dtype_of_its_inputs(self, device, dtype):
        image, mask = _labelled_batch(batch=4)
        image, mask = image.to(device=device, dtype=dtype), mask.to(device)
        aug = K.RandomTransplantation(p=1.0, excluded_labels=[0])  # the exclusion list starts on the CPU
        out_image, out_mask = aug(image, mask)
        assert out_image.dtype is dtype and out_image.device == image.device and out_mask.device == mask.device
        assert torch.equal(out_image, image.roll(1, dims=0)) and torch.equal(out_mask, mask.roll(1, dims=0))
        assert aug._params["selection"].device == mask.device
        assert aug._params["selected_labels"].device == mask.device

    def test_convention_label_draw_uses_the_cpu_generator_whatever_the_device(self, device):
        _, mask = _multi_label_batch()
        torch.manual_seed(11)
        on_cpu = K.RandomTransplantation(p=1.0)
        on_cpu(mask, data_keys=["mask"])
        torch.manual_seed(11)
        state = torch.get_rng_state()
        on_device = K.RandomTransplantation(p=1.0)
        on_device(mask.to(device), data_keys=["mask"])
        assert not torch.equal(torch.get_rng_state(), state)  # the CPU generator moved
        assert on_device._params["selected_labels"].cpu().tolist() == on_cpu._params["selected_labels"].tolist()

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
        # A hand-written gate selects images 1 and 3 only. Their donors are 0 and 2 -- the previous image of the
        # FULL batch, neither of them an acceptor -- and not 3 and 1, the previous ACCEPTOR.
        image, mask = _labelled_batch(batch=5)
        aug = K.RandomTransplantation(p=1.0)
        out_image, out_mask = aug(image, mask, params={"batch_prob": torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])})
        assert aug._params["acceptor_indices"].tolist() == [1, 3]
        assert aug._params["donor_indices"].tolist() == [0, 2]
        assert [int(out_mask[i].flatten()[0]) for i in range(5)] == [1, 1, 3, 3, 5]
        self.assert_close(out_image[1], image[0], rtol=0, atol=0)
        self.assert_close(out_image[3], image[2], rtol=0, atol=0)

    @pytest.mark.device_agnostic
    def test_convention_a_donor_gives_its_original_content_and_every_channel_moves(self):
        # Image i is its own index in every position, offset per channel. With every image an acceptor, a
        # transplant chained through already-written acceptors would hand image 0's content down the batch.
        _, mask = _labelled_batch(batch=4)
        image = torch.stack([torch.stack([torch.full((4, 6), 10.0 * i + c) for c in range(3)]) for i in range(4)])
        out_image, _ = K.RandomTransplantation(p=1.0)(image, mask)
        self.assert_close(out_image, image.roll(1, dims=0), rtol=0, atol=0)

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
        mask = torch.zeros(3, 5, 7, dtype=torch.long)
        for i in range(3):
            mask[i, i : i + 2, i : i + 3] = i + 1  # a different place per image, so the moved region shows
        image = torch.rand(3, 1, 5, 7)
        torch.manual_seed(0)
        aug = K.RandomTransplantation(p=1.0, excluded_labels=[0])
        _, out_mask = aug(image, mask)
        assert aug._params["selected_labels"].tolist() == [3, 1, 2]
        expected = mask.clone()
        for i in range(3):
            donor = (i - 1) % 3
            expected[i][mask[donor] == donor + 1] = donor + 1
        assert torch.equal(out_mask, expected)

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
        out_image, out_mask = aug(image, mask)
        params = aug._params
        # Acceptor 0 is dropped -- its donor, image 2, is entirely the excluded label -- and the rest stay aligned.
        assert params["acceptor_indices"].tolist() == [1, 2]
        assert params["donor_indices"].tolist() == [0, 1]
        assert len(params["selected_labels"]) == len(params["selection"]) == 2
        for d in range(2):
            donor_mask = mask[int(params["donor_indices"][d])]
            assert int(params["selected_labels"][d]) in (5, 7)
            assert torch.equal(params["selection"][d], donor_mask == int(params["selected_labels"][d]))
            assert bool(params["selection"][d].any())
        assert torch.equal(out_mask[0], mask[0])
        self.assert_close(out_image[0], image[0], rtol=0, atol=0)
        # Acceptor 2 takes ITS OWN donor's region: image 1 is label 7 throughout, so all of image 2 becomes 7.
        assert bool((out_mask[2] == 7).all())
        self.assert_close(out_image[2], image[1], rtol=0, atol=0)

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize(
        "mask_dtype, top",
        [(torch.uint8, 255), (torch.int8, 127), (torch.float16, 2048.0), (torch.float32, float("inf"))],
    )
    def test_convention_excluded_label_never_leaks_at_the_top_of_the_mask_dtype(self, mask_dtype, top):
        # Image 1 holds only excluded labels, one of them the largest its dtype can tell apart from `top + 1`.
        # No value of a bounded dtype can stand in for "nothing", so that acceptor is dropped instead.
        mask = torch.zeros(2, 4, 4, dtype=mask_dtype)
        mask[0, :2, :2] = 7
        mask[1, 2:, 1:] = top
        image = torch.stack([torch.full((1, 4, 4), float(i + 1)) for i in range(2)])
        aug = K.RandomTransplantation(p=1.0, excluded_labels=[0, top])
        out_image, out_mask = aug(image, mask)
        assert aug._params["acceptor_indices"].tolist() == [1]
        assert aug._params["batch_prob"].tolist() == [0.0, 1.0]  # the gate agrees with the pruned acceptors
        assert aug._params["selected_labels"].tolist() == [7]
        assert aug._params["selected_labels"].dtype is mask_dtype
        assert torch.equal(out_mask[0], mask[0])  # image 0 received nothing from the all-excluded image 1
        self.assert_close(out_image[0], image[0], rtol=0, atol=0)
        assert int((out_mask[1] == 7).sum()) == 4  # image 1 still received label 7 from image 0

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("mask_dtype", [torch.int64, torch.int32, torch.uint8, torch.bool, torch.float16])
    def test_convention_every_donor_excluded_is_a_clean_no_op(self, mask_dtype):
        mask = torch.zeros(3, 4, 6, dtype=mask_dtype)
        image = torch.rand(3, 1, 4, 6)
        aug = K.RandomTransplantation(p=1.0, excluded_labels=[0])
        out_image, out_mask = aug(image, mask)
        assert aug._params["acceptor_indices"].numel() == 0  # no donor had a label to give ...
        assert not bool((aug._params["batch_prob"] > 0.5).any())  # ... so the p=1 gate was closed for all three
        assert aug._params["selected_labels"].shape == (0,)
        assert aug._params["selected_labels"].dtype is mask_dtype  # not the float32 of a bare torch.empty(0)
        assert aug._params["selection"].shape == (0, 4, 6)
        self.assert_close(out_image, image, rtol=0, atol=0)
        assert torch.equal(out_mask, mask)

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("excluded", [None, [0]])
    def test_convention_zero_sized_spatial_axis_is_a_no_op(self, excluded):
        # An empty mask has no label at all, so nothing is eligible: empty in, empty out, no raise.
        mask = torch.zeros(2, 0, 4, dtype=torch.long)
        image = torch.rand(2, 3, 0, 4)
        out_image, out_mask = K.RandomTransplantation(p=1.0, excluded_labels=excluded)(image, mask)
        assert out_image.shape == image.shape and out_mask.shape == mask.shape

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
        image, mask = _multi_label_batch()
        torch.manual_seed(0)
        aug = K.RandomTransplantation(p=1.0)
        expected_image, expected_mask = aug(image, mask)
        full = copy.deepcopy(dict(aug._params))
        # The fixture discriminates: a fresh draw under the replay's seed picks other labels and moves other pixels.
        torch.manual_seed(999)
        fresh_image, _ = K.RandomTransplantation(p=1.0)(image, mask)
        assert not torch.equal(fresh_image, expected_image)
        torch.manual_seed(999)
        replayed_image, replayed_mask = aug(image, mask, params=copy.deepcopy(full))
        self.assert_close(replayed_image, expected_image, rtol=0, atol=0)
        assert torch.equal(replayed_mask, expected_mask)
        # Dropping selected_labels alone changes nothing: it is not even recomputed.
        without_labels = {k: v for k, v in copy.deepcopy(full).items() if k != "selected_labels"}
        again_image, _ = aug(image, mask, params=without_labels)
        self.assert_close(again_image, expected_image, rtol=0, atol=0)
        assert "selected_labels" not in without_labels
        # Dropping both selected_labels and selection redraws from the RNG, into the caller's own dict.
        redrawn = {k: v for k, v in copy.deepcopy(full).items() if k not in ("selected_labels", "selection")}
        torch.manual_seed(999)
        redrawn_image, _ = aug(image, mask, params=redrawn)
        self.assert_close(redrawn_image, fresh_image, rtol=0, atol=0)
        assert "selected_labels" in redrawn and aug._params is redrawn

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("excluded_donor", [False, True])
    @pytest.mark.parametrize("also_dropped", ["donor_indices", "acceptor_indices"])
    def test_convention_labels_are_not_drawn_beside_an_explicit_selection(self, also_dropped, excluded_donor):
        # With an index key missing the parameters are derived again, but a given selection is never
        # second-guessed: no label is drawn, the RNG is left alone and the same positions move. With a donor
        # that has no eligible label its acceptor was dropped, and the rebuilt indices must still line up.
        image, mask = _multi_label_batch()
        excluded = None
        if excluded_donor:
            mask[1] = 0  # image 1 donates to image 2 and holds only the excluded label
            excluded = [0]
        torch.manual_seed(0)
        aug = K.RandomTransplantation(p=1.0, excluded_labels=excluded)
        expected_image, _ = aug(image, mask)
        partial = {k: v.clone() for k, v in aug._params.items() if k not in ("selected_labels", also_dropped)}
        torch.manual_seed(5)
        before = torch.get_rng_state()
        replayed_image, _ = aug(image, mask, params=partial)
        assert torch.equal(torch.get_rng_state(), before)
        assert "selected_labels" not in partial and also_dropped in partial
        assert partial["acceptor_indices"].tolist() == ([0, 1, 3] if excluded_donor else [0, 1, 2, 3])
        self.assert_close(replayed_image, expected_image, rtol=0, atol=0)

    @pytest.mark.device_agnostic
    def test_convention_a_missing_selection_is_rebuilt_from_the_given_labels(self):
        # The given labels are used as they are: `excluded_labels` is not consulted, and a list shorter than the
        # acceptors leaves the trailing acceptors untouched.
        mask = torch.zeros(3, 4, 6, dtype=torch.long)
        for i in range(3):
            mask[i, i, :] = i + 1  # a different row per image, so the moved region names its donor
        image = torch.stack([torch.full((1, 4, 6), float(i + 1)) for i in range(3)])
        aug = K.RandomTransplantation(p=1.0, excluded_labels=[0])
        params = {"batch_prob": torch.ones(3), "selected_labels": torch.tensor([0, 1])}
        _, out_mask = aug(image, mask, params=params)
        assert params["selection"].flatten(1).sum(1).tolist() == [18, 6, 0]
        assert torch.equal(params["selection"][0], mask[2] == 0)  # the EXCLUDED label 0 of donor 2 was moved
        assert torch.equal(params["selection"][1], mask[0] == 1)
        assert torch.equal(out_mask[2], mask[2])  # no third label: acceptor 2 is untouched
        assert int((out_mask[1] == 1).sum()) == 6 and int((out_mask[0] == 3).sum()) == 0

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_convention_image_dtype_guard_and_free_mask_dtype(self, p):
        mask = torch.randint(0, 3, (2, 4, 5))
        for image_dtype in (torch.int64, torch.int32, torch.uint8, torch.bool):
            with pytest.raises(TypeError, match="Expected input of"):
                K.RandomTransplantation(p=p)(torch.ones(2, 1, 4, 5, dtype=image_dtype), mask)
        signed = (torch.int8, torch.int16, torch.int32, torch.int64)
        floating = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        for dtype in (torch.bool, torch.uint8, *signed, *floating):
            aug = K.RandomTransplantation(p=p, excluded_labels=[0])
            _, out_mask = aug(torch.rand(2, 1, 4, 5), mask.to(dtype))
            assert out_mask.dtype is dtype and aug._params["selected_labels"].dtype is dtype

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("image_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
    def test_convention_each_floating_image_dtype_is_accepted_and_kept(self, image_dtype):
        image, mask = _labelled_batch(batch=3)
        out_image, _ = K.RandomTransplantation(p=1.0)(image.to(image_dtype), mask)
        assert out_image.dtype is image_dtype
        assert torch.equal(out_image, image.to(image_dtype).roll(1, dims=0))

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
    def test_convention_a_mask_key_is_required_only_to_derive_the_parameters(self):
        with pytest.raises(ValueError, match="MASK"):
            K.RandomTransplantation(p=1.0)(torch.rand(2, 1, 4, 5), data_keys=["input"])
        # With complete parameters no mask is looked up: this is the call AugmentationSequential makes per input.
        image, mask = _multi_label_batch()
        torch.manual_seed(0)
        aug = K.RandomTransplantation(p=1.0)
        expected_image, expected_mask = aug(image, mask)
        recorded = copy.deepcopy(dict(aug._params))
        image_only = K.RandomTransplantation(p=1.0)(image, params=copy.deepcopy(recorded), data_keys=["input"])
        self.assert_close(image_only, expected_image, rtol=0, atol=0)
        # Outputs follow the input order, whatever it is.
        swapped = K.RandomTransplantation(p=1.0)(mask, image, params=recorded, data_keys=["mask", "input"])
        assert torch.equal(swapped[0], expected_mask) and torch.equal(swapped[1], expected_image)

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
        image, mask = _multi_label_batch()
        torch.manual_seed(0)
        aug = K.RandomTransplantation(p=1.0, excluded_labels=[0])
        expected_image, _ = aug(image, mask)
        torch.manual_seed(999)
        fresh_image, _ = K.RandomTransplantation(p=1.0, excluded_labels=[0])(image, mask)
        assert not torch.equal(fresh_image, expected_image)  # a redraw under the replay's seed would show
        assert not aug.state_dict() and not list(aug.parameters()) and not list(aug.buffers())
        for restored in (pickle.loads(pickle.dumps(aug)), copy.deepcopy(aug)):  # noqa: S301
            assert torch.equal(restored.excluded_labels, aug.excluded_labels)
            torch.manual_seed(999)
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
    def test_wart_container_runs_the_transplant_only_as_the_first_step_4707(self):
        image, mask = _labelled_batch(batch=3)
        first = K.AugmentationSequential(
            K.RandomTransplantation(p=1.0), K.RandomHorizontalFlip(p=0.0), data_keys=["image", "mask"]
        )
        _, out_mask = first(image, mask)
        assert out_mask.shape == (3, 1, 4, 6)  # the later step promoted the mask ...
        assert not torch.equal(out_mask[:, 0], mask)
        later = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.0), K.RandomTransplantation(p=1.0), data_keys=["image", "mask"]
        )
        with pytest.raises(BaseError, match="one additional dimension"):  # ... and that layout is refused
            later(image, mask)

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
