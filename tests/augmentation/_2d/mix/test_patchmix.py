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

from kornia.augmentation import PatchMix


@pytest.mark.parametrize(
    "batch_size,channels,height,width,patch_size",
    [
        (4, 3, 32, 32, 8),
        (2, 1, 16, 16, 4),
    ],
)
def test_patchmix_shape_and_type(batch_size, channels, height, width, patch_size):
    aug = PatchMix(alpha=1.0, patch_size=patch_size)
    x = torch.rand(batch_size, channels, height, width)
    params = aug.generate_parameters(x.shape)
    out = aug.apply_transform(x, params, {})
    assert out.shape == x.shape
    assert params["mix_pairs"].shape[0] == batch_size
    assert params["lam"].shape[0] == batch_size
    assert out.dtype == x.dtype


def test_patchmix_different_inputs():
    aug = PatchMix(alpha=1.0, patch_size=8)
    x = torch.rand(4, 3, 32, 32)
    y = torch.rand(4, 3, 32, 32)
    params = aug.generate_parameters(x.shape)
    out1 = aug.apply_transform(x, params, {})
    out2 = aug.apply_transform(y, params, {})
    assert not torch.allclose(out1, out2)


def test_patchmix_grad():
    aug = PatchMix(alpha=1.0, patch_size=8)
    x = torch.rand(2, 3, 16, 16, requires_grad=True)
    params = aug.generate_parameters(x.shape)
    out = aug.apply_transform(x, params, {})
    out.sum().backward()
    assert x.grad is not None


@pytest.mark.parametrize("patch_size", [9, 16])
def test_patchmix_rejects_a_patch_larger_than_the_input(patch_size):
    """A patch wider or taller than the image made the corner range negative.

    The corners were then drawn negative and the negative slice that followed copied an
    arbitrary rectangle: `patch_size=16` on an 8x10 input changed a 7x5 region, and
    `patch_size=9` changed a non-square 8x9 one, neither raising.
    """
    x = torch.rand(4, 1, 8, 10)
    with pytest.raises(ValueError, match=r"Expect `patch_size` to fit the input"):
        PatchMix(patch_size=patch_size, p=1.0)(x)


@pytest.mark.parametrize("patch_size", [1, 4, 8])
def test_patchmix_corners_stay_inside_the_input(patch_size):
    """Every corner a fitting patch draws leaves the patch inside the image."""
    x = torch.rand(4, 1, 8, 10)
    torch.manual_seed(0)
    aug = PatchMix(patch_size=patch_size, p=1.0)
    aug(x)
    coords = aug._params["patch_coords"]
    assert int(coords.min()) >= 0
    assert int(coords[:, 0].max()) + patch_size <= x.shape[-1]
    assert int(coords[:, 1].max()) + patch_size <= x.shape[-2]


def test_patchmix_same_on_batch_shares_the_location_not_the_pairing():
    """`same_on_batch` used to make the whole augmentation an identity.

    The pairing is an argsort of the drawn values; sharing that draw across the batch made
    every value equal, so the argsort was the identity permutation and each image was
    patched with itself. The location is what `same_on_batch` shares.
    """
    torch.manual_seed(0)
    aug = PatchMix(patch_size=4, p=1.0, same_on_batch=True)
    x = torch.rand(4, 1, 8, 8)
    out = aug(x)

    assert not torch.equal(out, x)
    coords = aug._params["patch_coords"].tolist()
    assert len({tuple(row) for row in coords}) == 1

    unchanged = 0
    for seed in range(50):
        torch.manual_seed(seed)
        sample = torch.rand(4, 1, 8, 8)
        unchanged += int(torch.equal(PatchMix(patch_size=4, p=1.0, same_on_batch=True)(sample), sample))
    assert unchanged < 10
