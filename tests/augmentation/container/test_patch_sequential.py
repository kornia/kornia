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

from copy import deepcopy

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
        with pytest.raises(ValueError, match=r"number of processing modules|Only boolean value allowed"):
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

    @pytest.mark.parametrize("shape", [(2, 3, 24, 24), (2, 3, 23, 25)])
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
        expected_shape = shape if padding == "same" else (*shape[:2], shape[2] // 2 * 2, shape[3] // 2 * 2)
        assert out.shape == expected_shape

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

    @pytest.mark.parametrize("empty", [False, True])
    @pytest.mark.parametrize("grid,image_size,tracked_size", [((2, 2), (9, 9), (8, 8)), ((2, 3), (9, 10), (8, 9))])
    def test_nested_valid_padding_updates_tracked_shape_5596(self, empty, grid, image_size, tracked_size):
        patch_seq = K.PatchSequential(
            *([] if empty else [K.RandomHorizontalFlip(p=1.0)]),
            grid_size=grid,
            padding="valid",
            patchwise_apply=False,
        )
        seq = K.ImageSequential(K.ImageSequential(patch_seq), K.RandomCrop((4, 4), p=1.0))
        params = seq.forward_parameters(torch.Size((64, 1, *image_size)))
        crop_input_size = params[1].data["input_size"]
        assert torch.equal(crop_input_size, crop_input_size.new_tensor(tracked_size).expand_as(crop_input_size))

    def test_nested_valid_padding_in_a_video_clip_5596(self, device, dtype):
        x = torch.zeros(2, 3, 1, 9, 9, device=device, dtype=dtype)
        inner = K.PatchSequential(
            K.RandomHorizontalFlip(p=1.0), grid_size=(2, 2), padding="valid", patchwise_apply=False
        )
        seq = K.AugmentationSequential(K.VideoSequential(inner, same_on_frame=False))
        assert seq(x).shape == (2, 3, 1, 8, 8)

    def test_patch_child_resize_does_not_change_tracked_image_shape_5596(self):
        patch_seq = K.PatchSequential(K.RandomResizedCrop((4, 4), p=1.0), grid_size=(2, 2), patchwise_apply=False)
        seq = K.VideoSequential(
            patch_seq,
            K.RandomCrop((2, 2), p=1.0),
            same_on_frame=False,
        )
        params = seq.forward_parameters(torch.Size((2, 3, 1, 8, 8)))
        assert params[1].data["input_size"][0].tolist() == [8, 8]

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

    @pytest.mark.parametrize("padding,patch_size", [("same", 3), ("valid", 2)])
    @pytest.mark.parametrize("patchwise_apply", [True, False])
    @pytest.mark.parametrize("same_on_batch", [True, False, None])
    @pytest.mark.parametrize("random_apply", [True, False])
    def test_image_shape_parameter_replay(
        self, padding, patch_size, patchwise_apply, same_on_batch, random_apply, device, dtype, monkeypatch
    ):
        seq = K.PatchSequential(
            *[K.RandomAffine(degrees=20, translate=(0.1, 0.2), p=1) for _ in range(6)],
            grid_size=(2, 3),
            padding=padding,
            patchwise_apply=patchwise_apply,
            same_on_batch=same_on_batch,
            random_apply=random_apply,
        )
        x = torch.arange(210, device=device, dtype=dtype).reshape(2, 3, 5, 7) / 210
        rng = torch.get_rng_state()
        image_params = seq.forward_parameters(x.shape)
        sampled_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        # Independently specified dimensions: 5x7 pads to 6x9 or crops to 4x6 on a 2x3 grid.
        patch_params = seq.forward_parameters(torch.Size((2, 6, 3, patch_size, patch_size)))
        assert torch.equal(torch.get_rng_state(), sampled_rng)
        assert len(image_params) == len(patch_params)
        for image_item, patch_item in zip(image_params, patch_params):
            assert image_item.indices == patch_item.indices
            assert image_item.param.name == patch_item.param.name
            assert image_item.param.data.keys() == patch_item.param.data.keys()
            for key in image_item.param.data:
                self.assert_close(image_item.param.data[key], patch_item.param.data[key], rtol=0, atol=0)

        monkeypatch.setattr(seq, "forward_parameters", lambda _: pytest.fail("Replay must not draw parameters"))
        out = seq(x, params=image_params)
        self.assert_close(out, seq(x, params=patch_params), rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), sampled_rng)
        assert out.shape == ((2, 3, 5, 7) if padding == "same" else (2, 3, 4, 6))

    @pytest.mark.parametrize("grid", [(1, 1), (4, 5)])
    def test_image_shape_parameters_with_small_image(self, grid, device, dtype):
        seq = K.PatchSequential(K.RandomHorizontalFlip(p=1), grid_size=grid, patchwise_apply=False)
        x = torch.arange(12, device=device, dtype=dtype).reshape(2, 1, 2, 3)
        params = seq.forward_parameters(x.shape)
        expected = x.flip(-1) if grid == (1, 1) else x
        self.assert_close(seq(x, params=params), expected, rtol=0, atol=0)

    @pytest.mark.parametrize("method", ["forward", "compute_padding", "extract_patches"])
    def test_three_dimensional_input_raises_value_error(self, method, device, dtype):
        seq = K.PatchSequential(nn.Identity(), grid_size=(2, 2), patchwise_apply=False)
        kwargs = {"padding": "same"} if method == "compute_padding" else {}
        with pytest.raises(ValueError, match="Expected image shape"):
            getattr(seq, method)(torch.ones(3, 4, 4, device=device, dtype=dtype), **kwargs)

    @pytest.mark.parametrize("shape", [(2, 3, 4), (2, 5, 3, 2, 2), (2, 6, 3, 2, 2, 1)])
    def test_parameters_reject_invalid_shape(self, shape):
        seq = K.PatchSequential(nn.Identity(), grid_size=(2, 3), patchwise_apply=False)
        with pytest.raises(ValueError, match=r"Expected.*shape"):
            seq.forward_parameters(torch.Size(shape))

    @pytest.mark.parametrize("shape", [(2, 3, 1, 2), (2, 3, 0, 4)])
    def test_image_shape_parameters_reject_empty_valid_patches(self, shape):
        seq = K.PatchSequential(nn.Identity(), grid_size=(2, 3), padding="valid", patchwise_apply=False)
        with pytest.raises(ValueError, match="non-empty"):
            seq.forward_parameters(torch.Size(shape))

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

    @pytest.mark.parametrize("container", [K.ImageSequential, K.AugmentationSequential])
    def test_nested_in_a_container_5584(self, container, device, dtype):
        # The parent reads the nested PatchSequential's PatchParamItem list while tracking the image shape; it used
        # to treat the items as ParamItem and raise AttributeError. A patch sequential does not change the image
        # size, so the sibling after it draws its crop for the full image.
        x = torch.arange(2 * 3 * 8 * 8, device=device).to(dtype).reshape(2, 3, 8, 8) / 384
        inner = K.PatchSequential(K.RandomInvert(p=1.0), grid_size=(2, 2), patchwise_apply=False)
        seq = container(inner, K.RandomCrop((6, 6), p=1.0))
        params = seq.forward_parameters(x.shape)
        assert params[1].data["input_size"][0].tolist() == [8, 8]
        out = seq(x, params=params)
        expected = K.RandomCrop((6, 6), p=1.0)(inner(x, params=params[0].data), params=params[1].data)
        self.assert_close(out, expected, rtol=0, atol=0)

    def test_nested_valid_padding_tracks_the_cropped_size_5584(self, device, dtype):
        # padding="valid" crops a 9 x 9 image to 8 x 8 before the patches are taken, and the sibling must see that.
        x = torch.arange(2 * 1 * 9 * 9, device=device).to(dtype).reshape(2, 1, 9, 9)
        inner = K.PatchSequential(
            K.RandomHorizontalFlip(p=1.0), grid_size=(2, 2), padding="valid", patchwise_apply=False
        )
        seq = K.ImageSequential(inner, K.RandomCrop((4, 4), p=1.0))
        params = seq.forward_parameters(x.shape)
        assert params[1].data["input_size"][0].tolist() == [8, 8]
        assert params[1].data["src"].max().item() <= 7
        assert seq(x, params=params).shape == (2, 1, 4, 4)

    def test_nested_in_video_sequential_5584(self, device, dtype):
        x = torch.arange(2 * 3 * 3 * 9 * 9, device=device).to(dtype).reshape(2, 3, 3, 9, 9)
        inner = K.PatchSequential(
            K.RandomHorizontalFlip(p=1.0), grid_size=(2, 2), padding="valid", patchwise_apply=False
        )
        seq = K.VideoSequential(inner, K.RandomCrop((4, 4), p=1.0), same_on_frame=False)
        params = seq.forward_parameters(x.shape)
        assert params[1].data["input_size"][0].tolist() == [8, 8]
        assert seq(x, params=params).shape == (2, 3, 3, 4, 4)

    @pytest.mark.parametrize(
        "children, grid_size, padding, image_size, tracked",
        [
            # a child resizing each 4 x 4 patch to 4 x 4 reports output_size 4 x 4; the image stays 8 x 8
            (lambda: [K.RandomResizedCrop((4, 4), p=1.0)], (2, 2), "same", (8, 8), [8, 8]),
            # a non-square grid crops the rows and the columns by their own remainders: 9 x 10 becomes 8 x 9
            (lambda: [K.RandomHorizontalFlip(p=1.0)], (2, 3), "valid", (9, 10), [8, 9]),
            # an empty patch sequential hands the parent an empty parameter list
            (list, (2, 2), "same", (8, 8), [8, 8]),
        ],
        ids=["patch_size_child", "non_square_valid", "no_children"],
    )
    def test_nested_tracks_the_image_shape_5584(self, children, grid_size, padding, image_size, tracked, device, dtype):
        inner = K.PatchSequential(*children(), grid_size=grid_size, padding=padding, patchwise_apply=False)
        seq = K.ImageSequential(inner, K.RandomCrop((2, 2), p=1.0))
        x = torch.zeros(2, 1, *image_size, device=device, dtype=dtype)
        params = seq.forward_parameters(x.shape)
        assert params[1].data["input_size"][0].tolist() == tracked
        assert seq(x, params=params).shape == (2, 1, 2, 2)


@pytest.mark.usefixtures("restore_torch_rng")
class TestConventionPatchSequential(BaseTester):
    """Convention pins for `PatchSequential` after the #4421 repair (#4460)."""

    def test_convention_random_patch_assignment_allows_fewer_modules_than_patches(self, device, dtype):
        with pytest.raises(ValueError, match="number of processing modules must be equal with grid size"):
            K.PatchSequential(K.RandomHorizontalFlip(p=1.0), grid_size=(2, 2), patchwise_apply=True, random_apply=False)
        seq = K.PatchSequential(
            K.RandomHorizontalFlip(p=1.0), grid_size=(2, 2), patchwise_apply=True, random_apply=True
        )
        image = torch.arange(16, device=device, dtype=dtype).reshape(1, 1, 4, 4)
        assert seq(image).shape == image.shape

    @staticmethod
    def _patch_states(device, dtype, grid, height, width):
        # Per-patch verdict: was this patch horizontally flipped, left untouched, or neither.
        torch.manual_seed(0)
        seq = K.PatchSequential(K.RandomHorizontalFlip(p=1.0), grid_size=grid, patchwise_apply=False)
        x = torch.rand(2, 3, height, width, device=device, dtype=dtype)
        out = seq(x)
        patch_h, patch_w = height // grid[0], width // grid[1]
        states = []
        for b in range(2):
            for row in range(grid[0]):
                for col in range(grid[1]):
                    rows = slice(row * patch_h, (row + 1) * patch_h)
                    cols = slice(col * patch_w, (col + 1) * patch_w)
                    patch_in, patch_out = x[b, :, rows, cols], out[b, :, rows, cols]
                    if torch.equal(patch_out, patch_in.flip(-1)):
                        states.append("flip")
                    elif torch.equal(patch_out, patch_in):
                        states.append("same")
                    else:
                        states.append("other")
        return states

    def test_convention_patch_sequential_augments_every_patch_row(self, device, dtype):
        # Convention pin: the grid splits the image into B x n_patches patch rows and the chain is applied to
        # all of them, for any channel count. Before #4460 the draw was sized B x C, so on a non-square 8x12
        # image with a (2, 3) grid only the first 6 of the 12 patches were augmented (#4421).
        assert self._patch_states(device, dtype, (2, 3), 8, 12) == ["flip"] * 12
        assert self._patch_states(device, dtype, (2, 2), 8, 8) == ["flip"] * 8

    def test_convention_default_patch_mode_augments_each_sample(self, device, dtype):
        # Convention pin: the default (patchwise_apply=True, random_apply=False) processes every sample.
        # Before #4460 the single module flipped sample zero and left sample one unchanged (#4421).
        seq = K.PatchSequential(K.RandomHorizontalFlip(p=1.0), grid_size=(1, 1))
        image = torch.arange(128, device=device, dtype=dtype).reshape(2, 1, 8, 8)
        self.assert_close(seq(image), image.flip(-1))


class TestEmptyPatchSequential(BaseTester):
    """Empty image batches keep patch geometry and the existing container contracts (#4429)."""

    @staticmethod
    def _sequence(padding, patchwise=False, random_apply=False, same_on_batch=None, children=2):
        modules = [K.RandomHorizontalFlip(p=0.5) if i % 2 else K.RandomVerticalFlip(p=1) for i in range(children)]
        return K.PatchSequential(
            *modules,
            grid_size=(2, 3),
            padding=padding,
            patchwise_apply=patchwise,
            random_apply=random_apply,
            same_on_batch=same_on_batch,
        )

    def _assert_params_equal(self, actual, expected):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            assert actual_item.indices == expected_item.indices
            assert actual_item.param.name == expected_item.param.name
            assert actual_item.param.data.keys() == expected_item.param.data.keys()
            for key in actual_item.param.data:
                self.assert_close(actual_item.param.data[key], expected_item.param.data[key], rtol=0, atol=0)

    @pytest.mark.parametrize("padding,patch_size,output_size", [("same", 5, (9, 13)), ("valid", 4, (8, 12))])
    @pytest.mark.parametrize(
        "patchwise,random_apply,same_on_batch,children",
        [
            (False, False, None, 2),
            (False, True, None, 2),
            (False, 2, None, 2),
            (False, (1, 2), None, 2),
            (True, False, None, 6),
            (True, True, False, 2),
            (True, True, True, 2),
            (False, False, None, 0),
        ],
    )
    def test_empty_patch_pipeline_4429(
        self, padding, patch_size, output_size, patchwise, random_apply, same_on_batch, children, device, dtype
    ):
        sequence = self._sequence(padding, patchwise, random_apply, same_on_batch, children)
        image = torch.empty(0, 3, 9, 13, device=device, dtype=dtype, requires_grad=True)
        pad = sequence.compute_padding(image, padding)
        patches = sequence.extract_patches(image, pad=pad)
        assert patches.shape == (0, 6, 3, patch_size, patch_size)
        assert patches.dtype == dtype
        assert patches.device == device
        rng = torch.get_rng_state()
        params = sequence.forward_parameters(image.shape)
        sampled_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        patch_params = sequence.forward_parameters(patches.shape)
        self._assert_params_equal(params, patch_params)
        assert torch.equal(torch.get_rng_state(), sampled_rng)
        for item in params:
            assert item.indices == []
            assert item.param.data["forward_input_shape"].tolist() == [0, 3, patch_size, patch_size]
        transformed = sequence.forward_by_params(patches, params)
        assert transformed.shape == patches.shape
        restored = sequence.restore_from_patches(transformed, pad=pad)
        assert restored.shape == (0, 3, *output_size)
        output = sequence(image, params=params)
        assert output.shape == restored.shape
        assert output.dtype == dtype
        assert output.device == device
        assert output.numel() == 0
        output.sum().backward()
        assert image.grad is not None
        assert image.grad.shape == image.shape

    @pytest.mark.parametrize("padding,output_size", [("same", (9, 13)), ("valid", (8, 12))])
    @pytest.mark.parametrize("patchwise", [False, True])
    @pytest.mark.parametrize("replay_state", ["same_params", "after_forward", "fresh_instance"])
    def test_empty_saved_patch_params_4429(
        self, padding, output_size, patchwise, replay_state, device, dtype, monkeypatch
    ):
        children = 6 if patchwise else 2
        sequence = self._sequence(padding, patchwise, children=children)
        image = torch.empty(0, 3, 9, 13, device=device, dtype=dtype, requires_grad=True)
        sequence(image)
        params = sequence._params if replay_state == "same_params" else deepcopy(sequence._params)
        snapshot = deepcopy(params)
        if replay_state == "after_forward":
            sequence(torch.zeros(2, 3, 7, 11, device=device, dtype=dtype))
        elif replay_state == "fresh_instance":
            sequence = self._sequence(padding, patchwise, children=children)
        monkeypatch.setattr(sequence, "forward_parameters", lambda _: pytest.fail("Replay must not draw parameters"))
        rng = torch.get_rng_state()
        output = sequence(image, params=params)
        assert output.shape == (0, 3, *output_size)
        assert output.dtype == dtype
        assert output.device == device
        assert torch.equal(torch.get_rng_state(), rng)
        self._assert_params_equal(params, snapshot)
        output.sum().backward()
        assert image.grad is not None
        assert image.grad.shape == image.shape

    @pytest.mark.parametrize("container", [K.ImageSequential, K.AugmentationSequential])
    @pytest.mark.parametrize("padding,tracked_size", [("same", (9, 13)), ("valid", (8, 12))])
    @pytest.mark.parametrize("children", [0, 2])
    def test_empty_nested_patch_shape_history_4429(self, container, padding, tracked_size, children, device, dtype):
        sequence = container(K.ImageSequential(self._sequence(padding, children=children)), K.Resize((3, 4)))
        image = torch.empty(0, 3, 9, 13, device=device, dtype=dtype, requires_grad=True)
        params = sequence.forward_parameters(image.shape)
        assert params[1].data["forward_input_shape"][-2:].tolist() == list(tracked_size)
        received = []
        handle = sequence[1].register_forward_pre_hook(lambda _, args: received.append(args[0].shape))
        try:
            output = sequence(image, params=params)
        finally:
            handle.remove()
        assert received == [torch.Size((0, 3, *tracked_size))]
        assert output.shape == (0, 3, 3, 4)
        assert output.dtype == dtype
        assert output.device == device
        output.sum().backward()
        assert image.grad is not None
        assert image.grad.shape == image.shape

    @pytest.mark.parametrize("layout", ["BTCHW", "BCTHW"])
    @pytest.mark.parametrize("padding,tracked_size", [("same", (9, 13)), ("valid", (8, 12))])
    def test_empty_video_patch_shape_history_4429(self, layout, padding, tracked_size, device, dtype):
        # Nested containers currently support same_on_frame=False only; do not change that contract.
        sequence = K.VideoSequential(self._sequence(padding), K.Resize((3, 4)), data_format=layout, same_on_frame=False)
        shape = (0, 2, 3, 9, 13) if layout == "BTCHW" else (0, 3, 2, 9, 13)
        expected_shape = (0, 2, 3, 3, 4) if layout == "BTCHW" else (0, 3, 2, 3, 4)
        image = torch.empty(*shape, device=device, dtype=dtype, requires_grad=True)
        params = sequence.forward_parameters(image.shape)
        assert params[1].data["forward_input_shape"][-2:].tolist() == list(tracked_size)
        output = sequence(image, params=params)
        assert output.shape == expected_shape
        assert output.dtype == dtype
        assert output.device == device
        output.sum().backward()
        assert image.grad is not None
        assert image.grad.shape == image.shape

    def test_empty_patch_validation_and_unsupported_ops_4429(self, device, dtype):
        sequence = self._sequence("valid")
        with pytest.raises(ValueError, match="non-empty patches"):
            sequence(torch.empty(0, 3, 1, 2, device=device, dtype=dtype))
        with pytest.raises(ValueError, match="non-empty spatial"):
            sequence(torch.empty(0, 3, 0, 13, device=device, dtype=dtype))
        with pytest.raises(ValueError, match="patch count"):
            sequence.restore_from_patches(torch.empty(0, 5, 3, 4, 4, device=device, dtype=dtype))
        image = torch.empty(0, 3, 9, 13, device=device, dtype=dtype)
        output = sequence(image)
        with pytest.raises(NotImplementedError, match="geometric transformations"):
            sequence.inverse(output)
        with pytest.raises(NotImplementedError, match="geometric transformations"):
            sequence.transform_masks(torch.empty(0, 1, 8, 12, device=device, dtype=dtype), sequence._params)

    @pytest.mark.parametrize("padding,output_size", [("same", (9, 13)), ("valid", (8, 12))])
    def test_dynamo_empty_patch_pipeline_4429(self, padding, output_size, device, dtype, torch_optimizer):
        sequence = self._sequence(padding)
        image = torch.empty(0, 3, 9, 13, device=device, dtype=dtype)
        params = sequence.forward_parameters(image.shape)
        actual = torch_optimizer(sequence)(image, params=params)
        assert actual.shape == (0, 3, *output_size)
        assert actual.dtype == dtype
        assert actual.device == device
