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

from reproducibility_test import reproducibility_test

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