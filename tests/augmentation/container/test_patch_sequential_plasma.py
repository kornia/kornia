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


@pytest.mark.parametrize(
    "augmentation_cls",
    [
        K.RandomPlasmaBrightness,
        K.RandomPlasmaContrast,
        K.RandomPlasmaShadow,
    ],
)
@pytest.mark.parametrize(
    ("patchwise_apply", "random_apply"),
    [
        (False, False),
        (True, False),
        (True, True),
    ],
)
def test_random_plasma_patch_sequential_4445(augmentation_cls, patchwise_apply, random_apply):
    torch.manual_seed(0)

    input = torch.rand(2, 3, 8, 8)
    modules = [augmentation_cls(p=1.0) for _ in range(4 if patchwise_apply else 1)]
    seq = K.PatchSequential(
        *modules,
        grid_size=(2, 2),
        patchwise_apply=patchwise_apply,
        random_apply=random_apply,
    )

    output = seq(input)

    assert output.shape == input.shape
