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

import kornia.augmentation as K

from testing.augmentation.utils import reproducibility_test
from testing.base import BaseTester


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
        torch.manual_seed(11)
        try:  # skip wrong param settings.
            seq = K.PatchSequential(
                K.color.RgbToBgr(),
                K.ColorJiggle(0.1, 0.1, 0.1, 0.1),
                K.ImageSequential(
                    K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5),
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
        # TODO: improve me and remove the exception.
        except Exception:
            return

        input = torch.randn(*shape, device=device, dtype=dtype)
        out = seq(input)
        assert out.shape[-3:] == input.shape[-3:]

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


class TestConventionPatchSequential(BaseTester):
    """Batch-6 convention pins for `PatchSequential`.

    Literals generated by the body of each pin, executed on this worktree on 2026-09-11 with
    `.venv/bin/python` (torch 2.14.0, python 3.11, cpu, float32). The augmentation is a `p=1.0` flip, so a
    patch is either flipped or untouched and no seed is involved.
    """

    @pytest.fixture(autouse=True)
    def _restore_global_rng(self):
        # These pins seed the global generator (or consume it through a `p=1.0` draw). Restoring its state
        # afterwards keeps them from shifting the draw of the unseeded tests that run after them: on a bare
        # `--dtype=all` run of `tests/augmentation`, which float16 parametrizations of
        # `TestSequential::test_forward` go red depends purely on the RNG position (tracked in #4446).
        # Only the CPU generator is restored - kornia draws its parameters there - so a pin that allocates
        # on an accelerator still advances that device's generator.
        state = torch.random.get_rng_state()
        try:
            yield
        finally:
            torch.random.set_rng_state(state)

    def test_wart_patch_sequential_augments_b_times_c_patch_rows_4421(self, device, dtype):
        # Wart pin (#4421): `PatchSequential.forward` hands the *image* shape to `forward_parameters`, which
        # reads `batch_shape[0] * batch_shape[1]` as the number of patch rows - that is B x C, not
        # B x n_patches. So a (1, 1) grid works only for C = 1 and raises `IndexError` otherwise, and where
        # the count happens to be smaller than the real patch count the trailing patches are silently left
        # unaugmented.
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0, cpu):
        # B=2, C=1, grid (1, 1) -> (2, 1, 8, 8); B=2, C=3, grid (1, 1) -> IndexError("index 2 is out of
        # bounds for dimension 0 with size 2") - on mps the same failure surfaces as
        # `torch.AcceleratorError` (a RuntimeError) with a device-scheduling-dependent index, so the pin
        # accepts either type and does not pin the index; B=2, C=3 on a non-square 8x12 image with a (2, 3)
        # grid -> the 12 patches come back as ['flip'] * 6 + ['same'] * 6, i.e. exactly B*C = 6 patch rows
        # are augmented and the whole second sample is left untouched; the square 8x8 / (2, 2) case gives
        # ['flip'] * 6 + ['same', 'same'], whose 6-of-8 split also discriminates the patch enumeration order.
        # The fix lands in the repair window; do not "correct" this pin here.
        single_channel = K.PatchSequential(K.RandomHorizontalFlip(p=1.0), grid_size=(1, 1), patchwise_apply=False)
        assert single_channel(torch.rand(2, 1, 8, 8, device=device, dtype=dtype)).shape == (2, 1, 8, 8)
        with pytest.raises((IndexError, RuntimeError), match="is out of bounds for dimension 0 with size 2"):
            K.PatchSequential(K.RandomHorizontalFlip(p=1.0), grid_size=(1, 1), patchwise_apply=False)(
                torch.rand(2, 3, 8, 12, device=device, dtype=dtype)
            )
        assert self._patch_states(device, dtype, (2, 3), 8, 12) == ["flip"] * 6 + ["same"] * 6
        assert self._patch_states(device, dtype, (2, 2), 8, 8) == ["flip"] * 6 + ["same", "same"]

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

    @pytest.mark.xfail(strict=True, reason="Tracked in #4421")
    def test_convention_patch_sequential_augments_every_patch_row(self, device, dtype):
        # Strict xfail (#4421): the intended reading is the one `PatchSequential`'s own docstring example
        # promises - the grid splits the image into B x n_patches patch rows and the chain is applied to all
        # of them, for any channel count. Today `forward` sizes the draw as B x C, so on a non-square 8x12
        # image with a (2, 3) grid only the first 6 of the 12 patches are augmented; this XFAILs and turns
        # XPASS when the repair lands, which is the signal to delete the wart pins above.
        # Executed 2026-09-11 (torch 2.14.0, cpu): ['flip'] * 6 + ['same'] * 6.
        assert self._patch_states(device, dtype, (2, 3), 8, 12) == ["flip"] * 12

    def test_wart_patch_sequential_padding_changes_image_and_batch_size_4421(self, device, dtype):
        # Wart pin (#4421, second half): `padding="same"` is documented as padding the image so that every
        # pixel is covered and `padding="valid"` as dropping the redundant border, but "same" SHRINKS the
        # image and "valid" can change the batch dimension, or fail outright, when the grid does not divide
        # the spatial shape.
        # Snippet used to generate expected: this body, executed 2026-09-11 (torch 2.14.0, cpu):
        # same/(3,3) on 8x8 -> (2, 1, 7, 7); same/(3,3) on 6x8 -> (2, 1, 6, 7); valid/(3,3) on 8x8 ->
        # (2, 1, 8, 8); valid/(4,4) on 6x8 -> (3, 1, 6, 6) [batch 2 -> 3]; valid/(3,3) on 6x8 ->
        # RuntimeError("shape '[-1, 3, 3, 1, 1, 2]' is invalid for input of size 48").
        # The fix lands in the repair window; do not "correct" this pin here.
        def run(padding, grid, height, width):
            seq = K.PatchSequential(
                K.RandomHorizontalFlip(p=1.0), grid_size=grid, padding=padding, patchwise_apply=False
            )
            return tuple(seq(torch.rand(2, 1, height, width, device=device, dtype=dtype)).shape)

        assert run("same", (3, 3), 8, 8) == (2, 1, 7, 7)
        assert run("same", (3, 3), 6, 8) == (2, 1, 6, 7)
        assert run("valid", (3, 3), 8, 8) == (2, 1, 8, 8)
        assert run("valid", (4, 4), 6, 8) == (3, 1, 6, 6)
        with pytest.raises(RuntimeError, match="is invalid for input of size"):
            run("valid", (3, 3), 6, 8)
