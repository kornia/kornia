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


class TestConnectedComponents(BaseTester):
    def test_smoke(self, device, dtype):
        img = torch.rand(1, 1, 3, 4, device=device, dtype=dtype)
        out = kornia.contrib.connected_components(img, num_iterations=10)
        assert out.shape == (1, 1, 3, 4)

    @pytest.mark.parametrize("shape", [(1, 3, 4), (2, 1, 3, 4)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.rand(shape, device=device, dtype=dtype)
        out = kornia.contrib.connected_components(img, num_iterations=10)
        assert out.shape == shape

    def test_exception(self, device, dtype):
        img = torch.rand(1, 1, 3, 4, device=device, dtype=dtype)

        with pytest.raises(TypeError) as errinf:
            assert kornia.contrib.connected_components(img, 1.0)
        assert "Input num_iterations must be a positive integer." in str(errinf)

        with pytest.raises(TypeError) as errinf:
            assert kornia.contrib.connected_components("not a tensor", 0)
        assert "Input imagetype is not a torch.Tensor" in str(errinf)

        with pytest.raises(TypeError) as errinf:
            assert kornia.contrib.connected_components(img, 0)
        assert "Input num_iterations must be a positive integer." in str(errinf)

        img = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        with pytest.raises(ValueError) as errinf:
            kornia.contrib.connected_components(img, 2)
        assert "Input image shape must be (*,1,H,W). Got:" in str(errinf)

    def test_value(self, device, dtype):
        img = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 15.0, 15.0, 0.0, 0.0, 12.0],
                        [0.0, 15.0, 15.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 35.0, 35.0, 0.0],
                        [0.0, 0.0, 0.0, 35.0, 35.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=torch.int64,
        )

        out = kornia.contrib.connected_components(img, num_iterations=10)
        self.assert_close(out, expected)

    def test_float16_precision_no_collision(self, device):
        """Regression test for a silent correctness bug: label IDs used to be
        generated using the input image's own dtype via torch.arange(...,
        dtype=image.dtype). float16 can only represent integers exactly up to
        2**11 = 2048, so on any image larger than that, two genuinely
        disconnected components could receive colliding IDs and end up with
        the same final label, with no error or warning."""
        W, H = 8, 3000  # H * W = 24000 pixels, well past float16's exact-integer limit
        img = torch.zeros(1, 1, H, W, device=device)
        img[0, 0, 2051, 0] = 1.0  # component A
        img[0, 0, 2053, 0] = 1.0  # component B, 2 rows away -> not connected

        out = kornia.contrib.connected_components(img.to(torch.float16), num_iterations=150)

        label_a = out[0, 0, 2051, 0]
        label_b = out[0, 0, 2053, 0]
        assert out.dtype == torch.int64
        assert label_a != label_b, "two disconnected components must not share a label"

    def test_labels_carry_no_gradient(self, device):
        img = torch.ones(2, 1, 4, 5, device=device, dtype=torch.float32, requires_grad=True)
        out = kornia.contrib.connected_components(img)
        assert out.requires_grad is False
        assert out.dtype == torch.int64

    @pytest.mark.parametrize("in_dtype", [torch.float32, torch.float16, torch.bfloat16, torch.bool, torch.uint8])
    def test_int64_labels_keep_distinct_components_apart(self, in_dtype, device):
        # Labels in the input dtype: float16 gave both pixels 2604.0, bool collapsed every label to True.
        m = torch.zeros(1, 1, 3, 1400, device=device)
        m[0, 0, 1, 1202] = 1
        m[0, 0, 1, 1204] = 1
        out = kornia.contrib.connected_components(m.to(in_dtype), num_iterations=5)
        assert out.dtype == torch.int64
        assert out[0, 0, 1, [1202, 1204]].tolist() == [1 * 1400 + 1202 + 1, 1 * 1400 + 1204 + 1]

    def test_uint8_pixel_at_raster_index_255_stays_foreground(self, device):
        # uint8 labels wrapped mod 256, so this pixel got label 0 (background).
        m = torch.zeros(1, 1, 9, 31, dtype=torch.uint8, device=device)
        m[0, 0, 8, 7] = 1
        assert kornia.contrib.connected_components(m, num_iterations=3)[0, 0, 8, 7].item() == 8 * 31 + 7 + 1

    def test_ids_stay_exact_past_2_pow_24_pixels(self, device):
        # float32 ids merged indices 2**24 - 1 and 2**24 + 1 into 16777216.
        if device.type == "mps":
            pytest.skip("MPS has no float64; covered by test_mps_raises_past_2_pow_24_pixels")
        m = torch.zeros(1, 1, 1, 2**24 + 8, device=device)
        m[..., 2**24 - 1] = 1
        m[..., 2**24 + 1] = 1
        out = kornia.contrib.connected_components(m, num_iterations=2)
        assert out[0, 0, 0, [2**24 - 1, 2**24 + 1]].tolist() == [2**24, 2**24 + 2]

    def test_batch_labels_are_unique_and_follow_the_flat_raster_index(self, device):
        # Per-image ids plus an int64 plane offset: label = flattened raster index over the batch + 1.
        g = torch.Generator().manual_seed(0)
        img = (torch.rand(4, 1, 5, 7, generator=g) > 0.5).float().to(device)
        out = kornia.contrib.connected_components(img, num_iterations=3)
        for b in range(4):
            single = kornia.contrib.connected_components(img[b : b + 1], num_iterations=3)
            expected = torch.where(single > 0, single + b * 5 * 7, single)
            assert torch.equal(out[b : b + 1], expected)
        full = kornia.contrib.connected_components(torch.ones(3, 1, 5, 7, device=device), num_iterations=10)
        assert full[:, 0, 0, 0].tolist() == [35, 70, 105]

    def test_batch_past_2_pow_24_pixels_in_total_runs(self, device):
        # The 2**24 bound is per image: pooling never mixes batch planes. 17 * 1024 * 1000 > 2**24 pixels in total.
        m = torch.zeros(17, 1, 1024, 1000, device=device)
        m[16, 0, 5, 3] = 1
        m[0, 0, 7, 9] = 1
        out = kornia.contrib.connected_components(m, num_iterations=1)
        assert out.dtype == torch.int64
        assert out[16, 0, 5, 3].item() == 16 * 1024 * 1000 + 5 * 1000 + 3 + 1
        assert out[0, 0, 7, 9].item() == 7 * 1000 + 9 + 1
        assert int((out > 0).sum()) == 2

    def test_mps_raises_past_2_pow_24_pixels(self, device):
        if device.type != "mps":
            pytest.skip("MPS-only bound")
        with pytest.raises(ValueError, match="2\\*\\*24"):
            kornia.contrib.connected_components(torch.zeros(1, 1, 1, 2**24 + 8, device=device))

    def test_permuted_batch(self, device, dtype):
        # A batch whose leading dims cannot be viewed together crashed on .view.
        g = torch.Generator().manual_seed(0)
        img = (torch.rand(3, 2, 1, 5, 7, generator=g) > 0.5).to(device, dtype).permute(1, 0, 2, 3, 4)
        out = kornia.contrib.connected_components(img)
        assert out.shape == (2, 3, 1, 5, 7)
        assert torch.equal(out, kornia.contrib.connected_components(img.contiguous()))

    @pytest.mark.parametrize("shape", [(1, 1, 0, 5), (2, 1, 5, 0), (0, 1, 5, 7), (3, 0, 1, 5, 7)])
    def test_empty_input(self, shape, device, dtype):
        # H = 0 crashed in reshape; reshape(-1, 1, 0, 5) is itself ambiguous, so the empty check has to come first.
        out = kornia.contrib.connected_components(torch.zeros(shape, device=device, dtype=dtype))
        assert out.shape == shape
        assert out.dtype == torch.int64

    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.contrib.connected_components
        op_jit = torch.jit.script(op)
        self.assert_close(op(img), op_jit(img))

    def test_convention_both_labelings_are_8_connected(self, device, dtype):
        # A diagonal pair is one component. scipy.ndimage.label's default structure is 4-connected (two components);
        # structure=np.ones((3, 3)) matches.
        pair = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype).view(1, 1, 2, 3)
        assert kornia.contrib.connected_components(pair).tolist() == [[[[5, 0, 0], [0, 5, 0]]]]
        assert kornia.contrib.connected_components_union_find(pair).tolist() == [[[[1, 0, 0], [0, 1, 0]]]]

    def test_convention_foreground_is_exactly_one(self, device, dtype):
        image = torch.tensor([1.0, 0.5, 2.0, 1.0, 0.75], device=device, dtype=dtype).view(1, 1, 1, 5)
        assert kornia.contrib.connected_components(image).tolist() == [[[[1, 0, 0, 4, 0]]]]

    def test_convention_the_default_budget_splits_a_line_longer_than_101_pixels(self, device, dtype):
        # A label travels one pixel per iteration; the 1 + largest raster index starts at the right end.
        line = torch.ones(1, 1, 1, 102, device=device, dtype=dtype)
        assert kornia.contrib.connected_components(line[..., :101]).unique().tolist() == [101]
        assert kornia.contrib.connected_components(line).unique().tolist() == [101, 102]
        assert kornia.contrib.connected_components(line, num_iterations=101).unique().tolist() == [102]


def test_compute_padding():
    assert kornia.contrib.compute_padding((6, 6), (2, 2)) == (0, 0, 0, 0)
    assert kornia.contrib.compute_padding((7, 7), (2, 2)) == (0, 1, 0, 1)
    assert kornia.contrib.compute_padding((8, 7), (4, 4)) == (0, 0, 0, 1)
