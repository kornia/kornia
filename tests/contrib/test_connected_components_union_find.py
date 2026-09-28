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


def _flood_fill(mask):
    """Independent 8-neighbor CPU oracle with canonical (first pixel) labels."""
    height, width = len(mask), len(mask[0])
    labels = [[0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            if not mask[y][x] or labels[y][x]:
                continue
            label = y * width + x + 1
            labels[y][x] = label
            queue = [(y, x)]
            while queue:
                row, col = queue.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = row + dy, col + dx
                        if 0 <= yy < height and 0 <= xx < width and mask[yy][xx] and not labels[yy][xx]:
                            labels[yy][xx] = label
                            queue.append((yy, xx))
    return labels


def _canonical(labels):
    mapping = {0: 0}
    return [mapping.setdefault(value, index + 1) for index, value in enumerate(labels)]


class TestConnectedComponentsUnionFind(BaseTester):
    @pytest.mark.parametrize("shape", [(1, 1, 513), (2, 1, 513, 1), (2, 3, 1, 17, 19)])
    def test_long_connected_regions(self, device, dtype, shape):
        image = torch.ones(shape, device=device, dtype=dtype)
        labels = kornia.contrib.connected_components_union_find(image)
        assert labels.shape == image.shape
        assert labels.dtype == torch.int64
        batches = labels.reshape(-1, shape[-2] * shape[-1])
        assert all(row.unique().numel() == 1 for row in batches)
        assert batches[:, 0].unique().numel() == batches.shape[0]

    @pytest.mark.parametrize("height,width", [(3, 3), (2, 4), (4, 2)])
    def test_exhaustive_small_masks(self, device, dtype, height, width):
        # Exhaust every occupancy pattern across horizontal/vertical block edges
        # as well as diagonals, empty blocks and odd-sized borders.
        pixels = height * width
        values = torch.arange(2**pixels, device=device)
        image = (values[:, None] >> torch.arange(pixels, device=device)) & 1
        image = image.reshape(-1, 1, height, width).to(dtype)
        labels = kornia.contrib.connected_components_union_find(image)
        for mask, result in zip(
            image.cpu().reshape(-1, height, width).tolist(), labels.cpu().reshape(-1, pixels).tolist()
        ):
            expected = [value for row in _flood_fill(mask) for value in row]
            assert _canonical(result) == expected

    def test_random_masks(self, device, dtype):
        generator = torch.Generator().manual_seed(19)
        image = (torch.rand(24, 1, 17, 23, generator=generator) < 0.35).to(device=device, dtype=dtype)
        labels = kornia.contrib.connected_components_union_find(image)
        for mask, result in zip(image.cpu().reshape(-1, 17, 23).tolist(), labels.cpu().reshape(-1, 391).tolist()):
            assert _canonical(result) == [value for row in _flood_fill(mask) for value in row]

    def test_serpentine(self, device, dtype):
        image = torch.zeros(1, 1, 33, 65, device=device, dtype=dtype)
        image[..., ::2, :] = 1
        image[..., 1::4, -1] = 1
        image[..., 3::4, 0] = 1
        labels = kornia.contrib.connected_components_union_find(image)
        assert labels[image == 1].unique().numel() == 1
        assert (labels[image == 0] == 0).all()

    def test_noncontiguous(self, device, dtype):
        image = torch.eye(31, device=device, dtype=dtype)[None].transpose(-1, -2)
        assert not image.is_contiguous()
        labels = kornia.contrib.connected_components_union_find(image)
        assert labels[image == 1].unique().numel() == 1
        assert (labels[image == 0] == 0).all()

    def test_half_labels_do_not_collide(self, device):
        image = torch.zeros(1, 1, 257, 257, device=device, dtype=torch.float16)
        image[..., ::2, ::2] = 1
        labels = kornia.contrib.connected_components_union_find(image)
        assert labels[image == 1].unique().numel() == 129**2
        assert labels.dtype == torch.int64

    def test_bool_and_determinism(self, device):
        image = torch.tensor([[[True, False, False], [False, False, True]]], device=device)
        expected = torch.tensor([[[1, 0, 0], [0, 0, 2]]], device=device)
        for _ in range(3):
            self.assert_close(kornia.contrib.connected_components_union_find(image), expected)

    @pytest.mark.parametrize("shape", [(0, 1, 3, 5), (1, 0, 5), (2, 1, 3, 0), (1, 3, 5)])
    def test_empty(self, device, dtype, shape):
        image = torch.zeros(shape, device=device, dtype=dtype)
        labels = kornia.contrib.connected_components_union_find(image)
        self.assert_close(labels, torch.zeros(shape, device=device, dtype=torch.int64))

    def test_exact_foreground(self, device, dtype):
        image = torch.tensor([[[0, 0.5, 1, -1, 2]]], device=device, dtype=dtype)
        labels = kornia.contrib.connected_components_union_find(image)
        self.assert_close(labels != 0, image == 1)

    @pytest.mark.parametrize("shape", [(1, 1, 1), (1, 1, 2), (1, 2, 1)])
    def test_single_block_integer(self, device, shape):
        image = torch.ones(shape, device=device, dtype=torch.int32)
        self.assert_close(kornia.contrib.connected_components_union_find(image), image.long())

    def test_no_grad(self, device, dtype):
        image = torch.ones(1, 3, 3, device=device, dtype=dtype, requires_grad=True)
        assert not kornia.contrib.connected_components_union_find(image).requires_grad

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError, match=r"torch\.Tensor"):
            kornia.contrib.connected_components_union_find(None)
        for shape in [(3, 4), (2, 3, 4), (1, 2, 3, 4)]:
            with pytest.raises(ValueError, match="shape"):
                kornia.contrib.connected_components_union_find(torch.zeros(shape, device=device, dtype=dtype))
