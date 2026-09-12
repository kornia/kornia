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
from torch import nn

import kornia.augmentation as K
from kornia.color import RgbToBgr

from testing.augmentation.utils import reproducibility_test
from testing.base import BaseTester


@pytest.fixture(autouse=True)
def _restore_global_rng():
    # The restored legacy test now draws parameters; keep it from shifting unrelated tests' RNG state.
    with torch.random.fork_rng(devices=[]):
        yield


class TestPatchSequential:
    @pytest.mark.parametrize(
        "error_param",
        [
            {"random_apply": False, "patchwise_apply": True, "grid_size": (2, 3)},
            {"random_apply": 2, "patchwise_apply": True},
            {"random_apply": (2, 3), "patchwise_apply": True},
        ],
    )
    def test_exception(self, error_param):
        with pytest.raises(Exception):  # AssertError and NotImplementedError
            K.PatchSequential(
                K.ImageSequential(
                    K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5),
                    K.RandomPerspective(0.2, p=0.5),
                    K.RandomSolarize(0.1, 0.1, p=0.5),
                ),
                K.ColorJiggle(0.1, 0.1, 0.1, 0.1),
                K.ImageSequential(
                    K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5),
                    K.RandomPerspective(0.2, p=0.5),
                    K.RandomSolarize(0.1, 0.1, p=0.5),
                ),
                K.ColorJiggle(0.1, 0.1, 0.1, 0.1),
                **error_param,
            )

    @pytest.mark.parametrize("shape", [(2, 3, 24, 24)])
    @pytest.mark.parametrize("padding", ["same", "valid"])
    @pytest.mark.parametrize("patchwise_apply", [True, False])
    @pytest.mark.parametrize("same_on_batch", [True, False, None])
    @pytest.mark.parametrize("keepdim", [True, False, None])
    @pytest.mark.parametrize("random_apply", [1, (2, 2), (1, 2), (2,), 10, True, False])
    def test_forward(self, shape, padding, patchwise_apply, same_on_batch, keepdim, random_apply, device, dtype):
        if patchwise_apply and not isinstance(random_apply, bool):
            pytest.skip("patchwise_apply=True only supports boolean random_apply")
        torch.manual_seed(11)
        # Exercise nested colour/geometric/mix operations without the unrelated HSV float16
        # zero-pixel path: perspective padding can introduce black pixels before another colour draw.
        seq = K.PatchSequential(
            RgbToBgr(),
            K.ColorJiggle(0.1, 0.1),
            K.ImageSequential(
                K.ColorJiggle(0.1, 0.1, p=0.5),
                K.RandomPerspective(0.2, p=0.5),
                K.RandomSolarize(0.1, 0.1, p=0.5),
            ),
            K.RandomMixUpV2(p=1.0),
            grid_size=(2, 2),
            padding=padding,
            patchwise_apply=patchwise_apply,
            same_on_batch=same_on_batch,
            keepdim=keepdim,
            random_apply=random_apply,
        )

        # Colour transforms expect RGB values in [0, 1], not a normally distributed image.
        input = torch.rand(*shape, device=device, dtype=dtype)
        out = seq(input)
        assert out.shape == input.shape

        reproducibility_test(input, seq)

    def test_intensity_only(self):
        seq = K.PatchSequential(
            K.ImageSequential(
                K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5),
                K.RandomPerspective(0.2, p=0.5),
                K.RandomSolarize(0.1, 0.1, p=0.5),
            ),
            K.ColorJiggle(0.1, 0.1, 0.1, 0.1),
            K.ImageSequential(
                K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5),
                K.RandomPerspective(0.2, p=0.5),
                K.RandomSolarize(0.1, 0.1, p=0.5),
            ),
            K.ColorJiggle(0.1, 0.1, 0.1, 0.1),
            grid_size=(2, 2),
        )
        assert not seq.is_intensity_only()

        seq = K.PatchSequential(
            K.ImageSequential(K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5)),
            K.ColorJiggle(0.1, 0.1, 0.1, 0.1),
            K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5),
            K.ColorJiggle(0.1, 0.1, 0.1, 0.1),
            grid_size=(2, 2),
        )
        assert seq.is_intensity_only()

    def test_autocast(self, device, dtype):
        if not hasattr(torch, "autocast"):
            pytest.skip("PyTorch version without autocast support")

        tfs = (K.RandomAffine(0.5, (0.1, 0.5), (0.5, 1.5), 1.2, p=1.0), K.RandomGaussianBlur((3, 3), (0.1, 3), p=1))
        aug = K.PatchSequential(*tfs, grid_size=(2, 2), random_apply=True)
        imgs = torch.rand(2, 3, 7, 4, dtype=dtype, device=device)

        with torch.autocast(device.type):
            output = aug(imgs)

        assert output.dtype == dtype, "Output image dtype should match the input dtype"


class _ScaleShift(nn.Module):
    def __init__(self, scale=1, shift=0):
        super().__init__()
        self.scale = scale
        self.shift = shift

    def forward(self, input):
        return input * self.scale + self.shift


class TestPatchSequentialRegression(BaseTester):
    """Numerical regressions for the patch layout and padding defects in #4421."""

    @pytest.fixture(autouse=True)
    def _seed_rng(self, _restore_global_rng):
        torch.manual_seed(42)

    @pytest.mark.parametrize("channels", [1, 3, 7])
    @pytest.mark.parametrize("grid", [(1, 1), (2, 2), (2, 3)])
    def test_augments_every_patch_row_4421(self, channels, grid, device, dtype):
        # Manual spatial slices are the oracle, independent of the extraction/restoration helpers.
        x = (torch.arange(2 * channels * 8 * 12, device=device) % 251).to(dtype).reshape(2, channels, 8, 12)
        seq = K.PatchSequential(K.RandomHorizontalFlip(p=1), grid_size=grid, patchwise_apply=False)
        expected = x.clone()
        h, w = 8 // grid[0], 12 // grid[1]
        for row in range(grid[0]):
            for col in range(grid[1]):
                rows, cols = slice(row * h, (row + 1) * h), slice(col * w, (col + 1) * w)
                expected[..., rows, cols] = x[..., rows, cols].flip(-1)
        self.assert_close(seq(x), expected, rtol=0, atol=0)

    @pytest.mark.parametrize("same_on_batch", [True, False, None])
    def test_location_wise_modules_cover_every_sample(self, same_on_batch, device, dtype):
        seq = K.PatchSequential(
            *[_ScaleShift(shift=i) for i in range(1, 7)], grid_size=(2, 3), same_on_batch=same_on_batch
        )
        x = torch.zeros(2, 3, 4, 6, device=device, dtype=dtype)
        x[1] = 10
        expected = x.clone()
        for row in range(2):
            for col in range(3):
                expected[..., row * 2 : row * 2 + 2, col * 2 : col * 2 + 2] += row * 3 + col + 1
        self.assert_close(seq(x), expected, rtol=0, atol=0)

    @pytest.mark.parametrize("same_on_batch", [True, False, None])
    def test_applies_complete_sequence_in_order(self, same_on_batch, device, dtype):
        seq = K.PatchSequential(
            _ScaleShift(shift=1),
            _ScaleShift(scale=2),
            grid_size=(2, 3),
            patchwise_apply=False,
            same_on_batch=same_on_batch,
        )
        x = torch.ones(2, 3, 4, 6, device=device, dtype=dtype)
        self.assert_close(seq(x), (x + 1) * 2, rtol=0, atol=0)

    @pytest.mark.parametrize("same_on_batch", [True, False, None])
    @pytest.mark.parametrize("child_same_on_batch", [True, False])
    def test_location_wise_parameter_batch(self, same_on_batch, child_same_on_batch):
        # Each location draws a batch of parameters, allowing both sharing and per-sample draws.
        seq = K.PatchSequential(
            *[K.RandomBrightness((0.8, 1.2), p=1, same_on_batch=child_same_on_batch) for _ in range(6)],
            grid_size=(2, 3),
            same_on_batch=same_on_batch,
        )
        params = seq.forward_parameters(torch.Size((2, 6, 3, 2, 2)))
        assert len(params) == 6
        for location, item in enumerate(params):
            assert item.indices == [location, location + 6]
            assert item.param.data["brightness_factor"].shape == (2,)
            expected_same = child_same_on_batch if same_on_batch is None else same_on_batch
            assert seq.get_submodule(item.param.name).same_on_batch is expected_same
            if expected_same:
                self.assert_close(item.param.data["brightness_factor"][0], item.param.data["brightness_factor"][1])

    @pytest.mark.parametrize("same_on_batch", [True, False, None])
    def test_random_patch_schedule_and_replay(self, same_on_batch, device, dtype, monkeypatch):
        seq = K.PatchSequential(
            _ScaleShift(shift=1),
            _ScaleShift(scale=2),
            grid_size=(2, 3),
            random_apply=True,
            same_on_batch=same_on_batch,
        )
        # Supply alternating sequences so sharing is tested without relying on random inequality.
        calls = []
        modules = list(seq.named_children())

        def select(with_mix=True):
            order = modules if len(calls) % 4 < 2 else modules[::-1]
            calls.append(order)
            return iter(order), False

        monkeypatch.setattr(seq, "get_random_forward_sequence", select)
        x = torch.ones(2, 3, 4, 6, device=device, dtype=dtype)
        out = seq(x)
        assert len(calls) == (6 if same_on_batch else 12)
        expected = x.clone()
        for batch in range(2):
            for location in range(6):
                call = location if same_on_batch else batch * 6 + location
                row, col = divmod(location, 3)
                expected[batch, :, row * 2 : row * 2 + 2, col * 2 : col * 2 + 2] = 4 if call % 4 < 2 else 3
        self.assert_close(out, expected, rtol=0, atol=0)
        params = seq._params
        monkeypatch.setattr(seq, "forward_parameters", lambda _: pytest.fail("Replay must not draw parameters"))
        self.assert_close(seq(x, params=params), out, rtol=0, atol=0)
        assert len(calls) == (6 if same_on_batch else 12)

    @pytest.mark.parametrize("patchwise_apply", [True, False])
    @pytest.mark.parametrize("same_on_batch", [True, False, None])
    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_random_parameter_replay(self, patchwise_apply, same_on_batch, padding, device, dtype, monkeypatch):
        seq = K.PatchSequential(
            K.RandomBrightness((0.8, 1.2), p=1),
            K.RandomHorizontalFlip(p=0.5),
            grid_size=(2, 3),
            patchwise_apply=patchwise_apply,
            random_apply=True,
            same_on_batch=same_on_batch,
            padding=padding,
        )
        x = torch.full((2, 3, 5, 7), 0.5, device=device, dtype=dtype)
        out = seq(x)
        params = seq._params
        monkeypatch.setattr(seq, "forward_parameters", lambda _: pytest.fail("Replay must not draw parameters"))
        self.assert_close(seq(x, params=params), out, rtol=0, atol=0)
        assert out.shape == ((2, 3, 5, 7) if padding == "same" else (2, 3, 4, 6))

    @pytest.mark.parametrize(
        "height,width,grid,crop",
        [
            (8, 8, (2, 2), (0, 8, 0, 8)),
            (8, 8, (3, 3), (1, 7, 1, 7)),
            (7, 10, (3, 4), (0, 6, 1, 9)),
            (6, 8, (4, 4), (1, 5, 0, 8)),
        ],
    )
    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_padding_preserves_batch_and_pixels_4421(self, height, width, grid, crop, padding, device, dtype):
        x = (torch.arange(2 * height * width, device=device) % 251).to(dtype).reshape(2, 1, height, width)
        seq = K.PatchSequential(nn.Identity(), grid_size=grid, padding=padding, patchwise_apply=False)
        top, bottom, left, right = crop
        expected = x if padding == "same" else x[..., top:bottom, left:right]
        self.assert_close(seq(x), expected, rtol=0, atol=0)

    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_padding_with_flip(self, padding, device, dtype):
        x = torch.arange(1, 16, device=device, dtype=dtype).reshape(1, 1, 3, 5)
        seq = K.PatchSequential(K.RandomHorizontalFlip(p=1), grid_size=(2, 2), padding=padding, patchwise_apply=False)
        # same: pad on right/bottom to 4x6, flip each 2x3 patch, remove that padding.
        # valid: crop right/bottom to 2x4, then flip each 1x2 patch.
        values = (
            [[3, 2, 1, 0, 5], [8, 7, 6, 0, 10], [13, 12, 11, 0, 15]]
            if padding == "same"
            else [[2, 1, 4, 3], [7, 6, 9, 8]]
        )
        expected = torch.tensor(values, device=device, dtype=dtype)[None, None]
        self.assert_close(seq(x), expected, rtol=0, atol=0)

    def test_same_padding_with_grid_larger_than_image(self, device, dtype):
        x = torch.arange(6, device=device, dtype=dtype).reshape(1, 1, 2, 3)
        seq = K.PatchSequential(nn.Identity(), grid_size=(4, 5), patchwise_apply=False)
        self.assert_close(seq(x), x, rtol=0, atol=0)

    @pytest.mark.parametrize("grid", [(0, 2), (-1, 2), (2,), (2, 3, 4), (2.5, 2), (True, 2)])
    def test_invalid_grid(self, grid):
        with pytest.raises(ValueError, match=r"grid_size.*two positive integers"):
            K.PatchSequential(nn.Identity(), grid_size=grid, patchwise_apply=False)

    def test_valid_padding_rejects_empty_patches(self, device, dtype):
        seq = K.PatchSequential(nn.Identity(), grid_size=(3, 4), padding="valid", patchwise_apply=False)
        with pytest.raises(ValueError, match="non-empty patches"):
            seq(torch.ones(2, 1, 2, 3, device=device, dtype=dtype))

    def test_extract_rejects_incompatible_grid(self, device, dtype):
        seq = K.PatchSequential(nn.Identity(), patchwise_apply=False)
        with pytest.raises(ValueError, match="divisible by grid_size"):
            seq.extract_patches(torch.ones(2, 1, 7, 10, device=device, dtype=dtype), grid_size=(3, 4))

    def test_restore_rejects_wrong_patch_count(self, device, dtype):
        seq = K.PatchSequential(nn.Identity(), patchwise_apply=False)
        with pytest.raises(ValueError, match="patch count"):
            seq.restore_from_patches(torch.ones(2, 9, 1, 2, 2, device=device, dtype=dtype), grid_size=(2, 3))

    @pytest.mark.parametrize("padding,patch_shape", [("same", (24, 3, 3, 3)), ("valid", (24, 3, 2, 2))])
    def test_parameters_use_patch_spatial_size(self, padding, patch_shape, device, dtype):
        seq = K.PatchSequential(K.RandomHorizontalFlip(p=1), grid_size=(3, 4), padding=padding, patchwise_apply=False)
        seq(torch.ones(2, 3, 7, 10, device=device, dtype=dtype))
        assert tuple(seq._params[0].param.data["forward_input_shape"].tolist()) == patch_shape

    def test_noncontiguous_input_is_not_modified(self, device, dtype):
        x = torch.arange(96, device=device, dtype=dtype).reshape(2, 1, 12, 4).transpose(-1, -2)
        original = x.clone()
        seq = K.PatchSequential(_ScaleShift(shift=1), grid_size=(2, 3), patchwise_apply=False)
        self.assert_close(seq(x), original + 1, rtol=0, atol=0)
        self.assert_close(x, original, rtol=0, atol=0)

    def test_mix_parameter_dtype_preserves_image_dtype(self, device, dtype):
        seq = K.PatchSequential(K.RandomMixUpV2(p=1), grid_size=(1, 1), patchwise_apply=False)
        params = seq.forward_parameters(torch.Size((2, 1, 3, 2, 2)))
        # MixUp's parameter generator can produce float32 lambdas for a half-precision image.
        params[0].param.data["mixup_pairs"] = torch.tensor([1, 0])
        params[0].param.data["mixup_lambdas"] = torch.full((2,), 0.25, dtype=torch.float32)
        x = torch.zeros(2, 3, 2, 2, device=device, dtype=dtype)
        x[1] = 1
        self.assert_close(seq(x, params=params), x * 0.75 + x.flip(0) * 0.25, rtol=0, atol=0)

    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_gradient_support(self, padding, device, dtype):
        x = torch.ones(2, 1, 3, 5, device=device, dtype=dtype, requires_grad=True)
        seq = K.PatchSequential(_ScaleShift(scale=2), grid_size=(2, 2), padding=padding, patchwise_apply=False)
        grad = torch.autograd.grad(seq(x).sum(), x)[0]
        expected = torch.full_like(x, 2)
        if padding == "valid":
            expected[..., -1, :] = 0
            expected[..., :, -1] = 0
        self.assert_close(grad, expected, rtol=0, atol=0)

    @pytest.mark.slow
    @pytest.mark.parametrize("padding", ["same", "valid"])
    def test_gradcheck(self, padding, device):
        x = torch.arange(1, 16, device=device, dtype=torch.float64).reshape(1, 1, 3, 5) / 16
        seq = K.PatchSequential(
            K.RandomHorizontalFlip(p=1),
            _ScaleShift(scale=2),
            grid_size=(2, 2),
            padding=padding,
            patchwise_apply=False,
        )
        seq(x)
        params = seq._params
        self.gradcheck(lambda value: seq(value, params=params), (x.requires_grad_(),))
