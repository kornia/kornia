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

import inspect
import warnings

import pytest
import torch

import kornia
from kornia.core._compat import torch_version_lt

from testing.base import BaseTester


class TestCombineTensorPatches(BaseTester):
    def test_smoke(self, device, dtype):
        img = torch.arange(16, device=device, dtype=dtype).view(1, 1, 4, 4)
        m = kornia.contrib.CombineTensorPatches((4, 4), (2, 2), (2, 2))
        patches = kornia.contrib.extract_tensor_patches(img, window_size=(2, 2), stride=(2, 2))
        assert m(patches).shape == (1, 1, 4, 4)
        self.assert_close(img, m(patches))

    def test_error(self, device, dtype):
        patches = kornia.contrib.extract_tensor_patches(
            torch.arange(16, device=device, dtype=dtype).view(1, 1, 4, 4), window_size=(3, 2), stride=(2, 2), padding=1
        )
        with pytest.raises(RuntimeError):
            kornia.contrib.combine_tensor_patches(patches, original_size=(4, 4), window_size=(2, 2), stride=(2, 2))

    def test_rect_odd_dim(self, device, dtype):
        img = torch.arange(12, device=device, dtype=dtype).view(1, 1, 4, 3)
        patches = kornia.contrib.extract_tensor_patches(img, window_size=(2, 2), stride=(2, 2), padding=(0, 2))
        m = kornia.contrib.combine_tensor_patches(
            patches, original_size=(4, 3), window_size=(2, 2), stride=(2, 2), unpadding=(0, 2)
        )
        assert m.shape == (1, 1, 4, 3)
        self.assert_close(img, m)

    def test_pad_triple_error(self, device, dtype):
        patches = kornia.contrib.extract_tensor_patches(
            torch.arange(36, device=device, dtype=dtype).view(1, 1, 6, 6), window_size=(4, 4), stride=(4, 4), padding=1
        )
        with pytest.raises(AssertionError):
            kornia.contrib.combine_tensor_patches(
                patches, original_size=(6, 6), window_size=(4, 4), stride=(4, 4), unpadding=(1, 1, 1)
            )

    def test_rectangle_array(self, device, dtype):
        img = torch.arange(24, device=device, dtype=dtype).view(1, 1, 4, 6)
        patches = kornia.contrib.extract_tensor_patches(img, window_size=(2, 2), stride=(2, 2), padding=1)
        m = kornia.contrib.CombineTensorPatches((4, 6), (2, 2), (2, 2), unpadding=1)
        assert m(patches).shape == (1, 1, 4, 6)
        self.assert_close(img, m(patches))

    def test_padding1(self, device, dtype):
        img = torch.arange(16, device=device, dtype=dtype).view(1, 1, 4, 4)
        padding = kornia.contrib.compute_padding((4, 4), (2, 2))
        patches = kornia.contrib.extract_tensor_patches(img, window_size=(2, 2), stride=(1, 1), padding=padding)
        m = kornia.contrib.CombineTensorPatches((4, 4), (2, 2), stride=(1, 1), unpadding=padding)
        assert m(patches).shape == (1, 1, 4, 4)
        self.assert_close(img, m(patches))

    def test_padding2(self, device, dtype):
        img = torch.arange(64, device=device, dtype=dtype).view(1, 1, 8, 8)
        patches = kornia.contrib.extract_tensor_patches(img, window_size=(2, 2), stride=(2, 2), padding=1)
        m = kornia.contrib.CombineTensorPatches((8, 8), (2, 2), stride=(2, 2), unpadding=1)
        assert m(patches).shape == (1, 1, 8, 8)
        self.assert_close(img, m(patches))

    def test_compute_padding(self, device, dtype):
        img_shape = (8, 13)
        rnge = img_shape[0] * img_shape[1]
        img = torch.arange(rnge, device=device, dtype=dtype).view(1, 1, *img_shape)
        window_size = (3, 3)
        padding = kornia.contrib.compute_padding(img_shape, window_size)
        patches = kornia.contrib.extract_tensor_patches(
            img, window_size=window_size, stride=window_size, padding=padding
        )
        m = kornia.contrib.CombineTensorPatches(img_shape, window_size, stride=window_size, unpadding=padding)
        assert m(patches).shape == (1, 1, *img_shape)
        self.assert_close(img, m(patches))

    def test_stride_greater_than_window_size(self, device, dtype):
        patches = kornia.contrib.extract_tensor_patches(
            torch.arange(16, device=device, dtype=dtype).view(1, 1, 4, 4), window_size=(2, 2), stride=(2, 2), padding=1
        )
        with pytest.raises(AssertionError):
            kornia.contrib.combine_tensor_patches(patches, original_size=(4, 4), window_size=(2, 2), stride=(3, 2))

    def test_expl_autopadding(self, device, dtype):
        img_shape = (8, 13)
        img = torch.arange(img_shape[0] * img_shape[1], device=device, dtype=dtype).view(
            1, 1, img_shape[0], img_shape[1]
        )
        window_size = (3, 3)
        padding = kornia.contrib.compute_padding(img_shape, window_size)
        patches = kornia.contrib.extract_tensor_patches(
            img, window_size=window_size, stride=window_size, padding=padding
        )
        m = kornia.contrib.CombineTensorPatches(img_shape, window_size, stride=window_size, unpadding=padding)
        assert m(patches).shape == (1, 1, *img_shape)

        self.assert_close(img, m(patches))

    def test_impl_autopadding(self, device, dtype):
        img_shape = (11, 16)
        img = torch.arange(img_shape[0] * img_shape[1], device=device, dtype=dtype).view(1, 1, *img_shape)
        window_size = (3, 3)
        patches = kornia.contrib.extract_tensor_patches(
            img, window_size=window_size, stride=window_size, allow_auto_padding=True
        )
        recomb = kornia.contrib.combine_tensor_patches(
            patches, img_shape, window_size=window_size, stride=window_size, allow_auto_unpadding=True
        )
        assert recomb.shape == img.shape
        self.assert_close(img, recomb)

    @pytest.mark.parametrize(
        "shape,window,stride,padding",
        [
            ((4, 6), (8, 8), (2, 2), (2, 2, 1, 1)),
            ((4, 6), (9, 9), (2, 2), (2, 3, 1, 2)),
            ((12, 6), (8, 8), (2, 2), (0, 0, 1, 1)),
            ((6, 12), (8, 8), (2, 2), (1, 1, 0, 0)),
            ((4, 15), (9, 8), (4, 3), (2, 3, 1, 1)),
        ],
    )
    def test_image_smaller_than_window(self, device, dtype, shape, window, stride, padding):
        image = torch.rand(2, 3, *shape, device=device, dtype=dtype)
        assert kornia.contrib.compute_padding(shape, window, stride) == padding
        patches = kornia.contrib.extract_tensor_patches(image, window, stride, allow_auto_padding=True)
        top, bottom, left, right = padding
        padded = torch.nn.functional.pad(image, (left, right, top, bottom))
        expected = torch.stack(
            [
                padded[..., y : y + window[0], x : x + window[1]]
                for y in range(0, padded.shape[-2] - window[0] + 1, stride[0])
                for x in range(0, padded.shape[-1] - window[1] + 1, stride[1])
            ],
            dim=1,
        )
        self.assert_close(patches, expected)
        restored = kornia.contrib.combine_tensor_patches(patches, shape, window, stride, allow_auto_unpadding=True)
        self.assert_close(restored, image)

    def test_small_image_gradcheck(self, device):
        image = torch.rand(1, 1, 2, 3, device=device, dtype=torch.float64, requires_grad=True)

        def roundtrip(x):
            patches = kornia.contrib.extract_tensor_patches(x, (4, 5), 2, allow_auto_padding=True)
            return kornia.contrib.combine_tensor_patches(patches, (2, 3), (4, 5), 2, allow_auto_unpadding=True)

        self.gradcheck(roundtrip, (image,))

    def test_small_image_dynamo(self, device, dtype, torch_optimizer):
        image = torch.rand(2, 3, 4, 6, device=device, dtype=dtype)
        op = torch.nn.Sequential(
            kornia.contrib.ExtractTensorPatches((8, 8), stride=2, allow_auto_padding=True),
            kornia.contrib.CombineTensorPatches((4, 6), (8, 8), stride=2, allow_auto_unpadding=True),
        )
        self.assert_close(torch_optimizer(op)(image), op(image))

    def test_gradcheck(self, device):
        patches = kornia.contrib.extract_tensor_patches(
            torch.arange(16.0, device=device, dtype=torch.float64).view(1, 1, 4, 4), window_size=(2, 2), stride=(2, 2)
        )
        self.gradcheck(kornia.contrib.combine_tensor_patches, (patches, (4, 4), (2, 2), (2, 2)))

    @pytest.mark.parametrize("batch_size", [0, 1, 2])
    @pytest.mark.parametrize(
        "image_size,window,stride,unpadding,auto_unpadding,pad,grid",
        [
            ((8, 12), (4, 4), (4, 4), 0, False, (0, 0, 0, 0), (2, 3)),
            ((7, 11), (3, 5), (2, 3), 0, False, (0, 0, 0, 0), (3, 3)),
            ((7, 11), (3, 5), (2, 3), (1, 1, 2, 1), False, (2, 1, 1, 1), (4, 4)),
            ((9, 13), (4, 5), (4, 5), 0, True, (1, 1, 1, 2), (3, 3)),
        ],
    )
    def test_patch_reconstruction_and_empty_batch_4429(
        self, batch_size, image_size, window, stride, unpadding, auto_unpadding, pad, grid, device, dtype
    ):
        image = (torch.arange(batch_size * 3 * image_size[0] * image_size[1], device=device) % 251).to(dtype)
        image = image.reshape(batch_size, 3, *image_size)
        padded = torch.nn.functional.pad(image, pad)
        # Construct patches without the extraction API so paired layout mistakes cannot cancel out.
        patches = (
            torch.stack(
                [
                    padded[
                        ...,
                        row * stride[0] : row * stride[0] + window[0],
                        col * stride[1] : col * stride[1] + window[1],
                    ]
                    for row in range(grid[0])
                    for col in range(grid[1])
                ],
                dim=1,
            )
            .detach()
            .requires_grad_()
        )
        module = kornia.contrib.CombineTensorPatches(image_size, window, stride, unpadding, auto_unpadding)
        actual = module(patches)
        functional = kornia.contrib.combine_tensor_patches(
            patches, image_size, window, stride, auto_unpadding, unpadding
        )
        accumulated = torch.zeros_like(padded)
        counts = torch.zeros_like(padded)
        for row in range(grid[0]):
            for col in range(grid[1]):
                rows = slice(row * stride[0], row * stride[0] + window[0])
                cols = slice(col * stride[1], col * stride[1] + window[1])
                accumulated[..., rows, cols] += patches[:, row * grid[1] + col]
                counts[..., rows, cols] += 1
        expected = torch.nn.functional.pad(accumulated / (counts + 1e-8), tuple(-value for value in pad))
        assert actual.shape == (batch_size, 3, *image_size)
        assert actual.dtype == dtype
        assert actual.device == device
        self.assert_close(actual, expected)
        self.assert_close(functional, expected)
        self.assert_close(actual, image)
        weights = (torch.arange(actual.numel(), device=device) % 7 + 1).to(dtype).reshape(actual.shape)
        actual_grad = torch.autograd.grad((actual * weights).sum(), patches)[0]
        expected_grad = torch.autograd.grad((expected * weights).sum(), patches)[0]
        assert actual_grad.shape == patches.shape
        self.assert_close(actual_grad, expected_grad)

    def test_invalid_empty_patch_metadata_4429(self, device, dtype):
        patches = torch.empty(0, 6, 3, 4, 4, device=device, dtype=dtype)
        with pytest.raises(AssertionError, match=r"Stride=.*Window size"):
            kornia.contrib.combine_tensor_patches(patches, (8, 12), (4, 4), (5, 4))
        with pytest.raises(RuntimeError, match="sliding blocks"):
            kornia.contrib.combine_tensor_patches(patches[:, :5], (8, 12), (4, 4), (4, 4))

    @pytest.mark.parametrize("batch_size", [0, 2])
    def test_dynamo_patch_reconstruction_4429(self, batch_size, device, dtype, torch_optimizer):
        patches = (torch.arange(batch_size * 6 * 3 * 4 * 4, device=device) % 251).to(dtype)
        patches = patches.reshape(batch_size, 6, 3, 4, 4)
        module = kornia.contrib.CombineTensorPatches((8, 12), (4, 4), (4, 4))
        actual = torch_optimizer(module)(patches)
        assert actual.shape == (batch_size, 3, 8, 12)
        self.assert_close(actual, module(patches))

    def test_all_zero_unpadding_means_no_unpadding(self, device, dtype):
        img = torch.arange(1, 36, device=device, dtype=dtype).view(1, 1, 5, 7)
        patches = kornia.contrib.extract_tensor_patches(img, 2, 2, allow_auto_padding=True)
        out = kornia.contrib.combine_tensor_patches(patches, (5, 7), 2, 2, allow_auto_unpadding=True, unpadding=(0, 0))
        self.assert_close(out, img)

    @pytest.mark.parametrize("stride", [2, 1])
    def test_float64_round_trip_is_exact(self, stride, device):
        # An eps added to every divisor returned 999.99999 for 1000.0.
        if device.type == "mps":
            pytest.skip("MPS has no float64")
        g = torch.Generator().manual_seed(0)
        img = (torch.rand(1, 2, 4, 6, generator=g, dtype=torch.float64) * 1000).to(device)
        out = kornia.contrib.combine_tensor_patches(
            kornia.contrib.extract_tensor_patches(img, 2, stride), (4, 6), 2, stride
        )
        assert torch.equal(out, img)  # overlap counts are 1, 2 or 4: the mean of identical copies is exact

    @pytest.mark.parametrize("stride", [2, 1])
    @pytest.mark.parametrize("int_dtype", [torch.uint8, torch.int32, torch.int64])
    def test_integer_round_trip_is_exact(self, stride, int_dtype, device):
        # Integers folded through float32 returned 16777216 for 16777217 (float64 would still round 2**53 + 1).
        # Integer patches are summed in int64, exactly (on MPS from torch 2.9; older torch keeps the float32 fold).
        # At stride 1 a pixel is covered up to 4 times, so 4 * (2**60 + 1) still fits int64.
        if int_dtype != torch.uint8 and device.type == "mps" and torch_version_lt(2, 9, 0):
            pytest.skip("MPS index_add_ has no int64 source before torch 2.9: the float32 fold is exact only to 2**24")
        big = {torch.uint8: 255, torch.int32: 2**31 - 1, torch.int64: 2**60 + 1}[int_dtype]
        img = torch.full((1, 2, 4, 6), 7, dtype=int_dtype, device=device)
        img[0, 0, 1, 2] = big
        img[0, 1, 2, 3] = big - 2
        out = kornia.contrib.combine_tensor_patches(
            kornia.contrib.extract_tensor_patches(img, 2, stride), (4, 6), 2, stride
        )
        assert out.dtype == int_dtype
        assert torch.equal(out, img)

    def test_integer_overlap_mean_truncates_toward_zero(self, device):
        # Overlapping, non-identical integer patches. Pixel (0, 1) is covered by patches 0 and 1: (5 + 2) / 2 = 3.5
        # -> 3, and (-5 - 2) / 2 = -3.5 -> -3 (toward zero, as a float mean cast to an integer dtype does).
        patches = torch.zeros(1, 6, 1, 2, 2, dtype=torch.int32, device=device)  # 3 x 4 image, 2 x 2 windows, stride 1
        patches[0, 0, 0, 0, 1] = 5
        patches[0, 1, 0, 0, 0] = 2
        out = kornia.contrib.combine_tensor_patches(patches, (3, 4), 2, 1)
        assert out[0, 0, 0, 1].item() == 3
        out = kornia.contrib.combine_tensor_patches(-patches, (3, 4), 2, 1)
        assert out[0, 0, 0, 1].item() == -3

    def test_bool_overlap_keeps_main_result(self, device):
        # A pixel is True when any covering patch is True, as a float overlap mean cast back to bool gives. Literal
        # produced by:
        #   p = torch.zeros(1, 6, 1, 2, 2, dtype=torch.bool); p[0, 0] = True; p[0, 1] = True
        #   combine_tensor_patches(p, (3, 4), 2, 1).tolist()
        patches = torch.zeros(1, 6, 1, 2, 2, dtype=torch.bool, device=device)
        patches[0, 0] = True
        patches[0, 1] = True
        out = kornia.contrib.combine_tensor_patches(patches, (3, 4), 2, 1)
        assert out.dtype == torch.bool
        assert out.tolist() == [[[[True, True, True, False], [True, True, True, False], [False, False, False, False]]]]

    @pytest.mark.parametrize("int_dtype", [torch.int64, torch.bool])
    def test_empty_batch_integer_patches(self, int_dtype, device):
        patches = kornia.contrib.extract_tensor_patches(torch.zeros(0, 3, 8, 12, dtype=int_dtype, device=device), 4, 4)
        out = kornia.contrib.combine_tensor_patches(patches, (8, 12), 4, 4)
        assert out.shape == (0, 3, 8, 12)
        assert out.dtype == int_dtype

    def test_integer_unpadding_crops_the_right_sides(self, device):
        # Non-symmetric unpadding, (top, bottom, left, right) = (0, 2, 1, 3): a top/bottom or left/right swap
        # in the crop changes the result.
        img = torch.arange(1, 1 + 2 * 5 * 7, device=device, dtype=torch.int64).view(1, 2, 5, 7)
        patches = kornia.contrib.extract_tensor_patches(img, 3, 2, padding=(0, 2, 1, 3))
        out = kornia.contrib.combine_tensor_patches(patches, (5, 7), 3, 2, unpadding=(0, 2, 1, 3))
        assert torch.equal(out, img)
        out_f = kornia.contrib.combine_tensor_patches(patches.float(), (5, 7), 3, 2, unpadding=(0, 2, 1, 3))
        assert torch.equal(out, out_f.long())

    @pytest.mark.parametrize("patch_dtype", [torch.int64, torch.int32, torch.bool, torch.float32])
    def test_negative_unpadding_restores_a_zero_border(self, patch_dtype, device):
        # padding=-1 crops a 4 x 6 image to its 2 x 4 interior; unpadding=-1 pads the recombined interior back to
        # 4 x 6 with zeros (False for bool), on the integer path as on the float one. Slicing gave a 1 x 1 tensor.
        img = (torch.arange(1, 25, device=device).view(1, 1, 4, 6) % 3 + 1).to(patch_dtype)
        patches = kornia.contrib.extract_tensor_patches(img, 2, 2, padding=-1)
        out = kornia.contrib.combine_tensor_patches(patches, (4, 6), 2, 2, unpadding=-1)
        expected = torch.zeros_like(img)
        expected[..., 1:-1, 1:-1] = img[..., 1:-1, 1:-1]
        assert out.dtype == patch_dtype
        assert torch.equal(out, expected)

    def test_int64_overlap_mean_is_exact_at_the_int64_limits(self, device):
        # Overlapping copies of 2**63 - 1 or -2**63 overflowed the int64 sum (they came back as -1 and 0). The mean is
        # built from per-element quotients and remainders by the overlap count, so no partial sum leaves int64.
        if device.type == "mps" and torch_version_lt(2, 9, 0):
            pytest.skip("MPS index_add_ has no int64 source before torch 2.9: the float32 fold is exact only to 2**24")
        for value in (2**63 - 1, -(2**63)):
            img = torch.full((1, 2, 3, 5), value, dtype=torch.int64, device=device)
            out = kornia.contrib.combine_tensor_patches(kornia.contrib.extract_tensor_patches(img, 2, 1), (3, 5), 2, 1)
            assert torch.equal(out, img)
        # Two patches over one pixel with opposite extremes: (2**63 - 1 - 2**63) / 2 = -0.5 truncates to 0.
        patches = torch.zeros(1, 8, 1, 2, 2, dtype=torch.int64, device=device)  # 3 x 5 image, 2 x 2 windows, stride 1
        patches[0, 0, 0, 0, 1] = 2**63 - 1
        patches[0, 1, 0, 0, 0] = -(2**63)
        assert kornia.contrib.combine_tensor_patches(patches, (3, 5), 2, 1)[0, 0, 0, 1].item() == 0
        # Random values over the whole int64 range, against the exact truncated mean in Python integers.
        g = torch.Generator().manual_seed(0)
        patches = torch.randint(-(2**63), 2**63 - 1, (2, 8, 2, 2, 2), generator=g, dtype=torch.int64)
        target = kornia.contrib.extract_tensor_patches(torch.arange(15).view(1, 1, 3, 5), 2, 1)[0, :, 0].flatten()
        expected = []
        for b in range(2):
            for c in range(2):
                covering = [[] for _ in range(15)]
                for value, pixel in zip(patches[b, :, c].flatten().tolist(), target.tolist()):
                    covering[pixel].append(value)
                means = [abs(sum(v)) // len(v) * (1 if sum(v) >= 0 else -1) for v in covering]
                expected.append(means)
        out = kornia.contrib.combine_tensor_patches(patches.to(device), (3, 5), 2, 1)
        assert out.flatten(2).flatten(0, 1).tolist() == expected

    def test_uint64_overlap_mean_is_exact_above_2pow63(self, device):
        # A plain int64 cast wraps 2**63 to -2**63, which returned 3 * 2**62 for (2**63 + 0) / 2 instead of 2**62.
        if device.type == "mps" and torch_version_lt(2, 9, 0):
            pytest.skip("MPS index_add_ has no int64 source before torch 2.9: the float32 fold is exact only to 2**24")
        # The values are set in int64 and viewed as uint64 (v - 2**64 for v >= 2**63): torch 2.5.1 cannot set a uint64
        # element from a Python int >= 2**63 ("Overflow when unpacking long").
        signed = torch.zeros(1, 6, 1, 2, 2, dtype=torch.int64, device=device)  # 3 x 4 image, 2 x 2 windows, stride 1
        signed[0, 1, 0, 0, 0] = -(2**63)  # 2**63
        out = kornia.contrib.combine_tensor_patches(signed.view(torch.uint64), (3, 4), 2, 1)
        assert out.dtype == torch.uint64
        assert out[0, 0, 0, 1].item() == 2**62
        # Near the top of the range: (2**64 - 1 + 2**64 - 3) / 2 = 2**64 - 2, beyond a float64 fold's precision.
        signed[0, 0, 0, 0, 1] = -1  # 2**64 - 1
        signed[0, 1, 0, 0, 0] = -3  # 2**64 - 3
        assert (
            kornia.contrib.combine_tensor_patches(signed.view(torch.uint64), (3, 4), 2, 1)[0, 0, 0, 1].item()
            == 2**64 - 2
        )
        # Patches that do not fit (6 patches of a 4 x 6 image recombined to 5 x 7, the warned path) leave row 4 and
        # column 6 uncovered: they stay 0, not 2**63.
        img = torch.full((1, 1, 4, 6), 2**64 - 1, dtype=torch.uint64, device=device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # the existing "does not fit" warning
            out = kornia.contrib.combine_tensor_patches(kornia.contrib.extract_tensor_patches(img, 2, 2), (5, 7), 2, 2)
        expected = torch.full((1, 1, 5, 7), 2**64 - 1, dtype=torch.uint64)
        expected[..., 4, :] = 0
        expected[..., :, 6] = 0
        assert torch.equal(out.cpu(), expected)  # MPS has no uint64 eq kernel
        # Random values over the whole uint64 range, against the exact floor mean in Python integers.
        g = torch.Generator().manual_seed(0)
        patches = torch.randint(-(2**63), 2**63 - 1, (2, 8, 2, 2, 2), generator=g, dtype=torch.int64).view(torch.uint64)
        target = kornia.contrib.extract_tensor_patches(torch.arange(15).view(1, 1, 3, 5), 2, 1)[0, :, 0].flatten()
        expected_means = []
        for b in range(2):
            for c in range(2):
                covering = [[] for _ in range(15)]
                for value, pixel in zip(patches[b, :, c].flatten().tolist(), target.tolist()):
                    covering[pixel].append(value)
                expected_means.append([sum(v) // len(v) for v in covering])
        out = kornia.contrib.combine_tensor_patches(patches.to(device), (3, 5), 2, 1)
        assert out.flatten(2).flatten(0, 1).tolist() == expected_means

    @pytest.mark.parametrize("int_dtype", [torch.int64, torch.bool])
    def test_integer_patch_count_mismatch_raises(self, int_dtype, device):
        patches = torch.zeros(1, 5, 1, 2, 2, dtype=int_dtype, device=device)  # a 4 x 6 image at stride 2 needs 6
        with pytest.raises(RuntimeError, match="window size"):
            kornia.contrib.combine_tensor_patches(patches, (4, 6), 2, 2)
        patches = torch.zeros(1, 6, 1, 3, 3, dtype=int_dtype, device=device)  # window size 3, not 2
        with pytest.raises(RuntimeError, match="window size"):
            kornia.contrib.combine_tensor_patches(patches, (4, 6), 2, 2)

    def test_complex_round_trip(self, device):
        # Complex patches are not integers: main cast them to float and discarded the imaginary part. They are folded
        # as two real tensors, since MPS has no complex col2im kernel.
        if device.type == "mps" and torch_version_lt(2, 9, 0):
            pytest.skip("complex MPS tensors abort the process in unfold on torch 2.5.1")
        g = torch.Generator().manual_seed(0)
        img = torch.randn(1, 2, 4, 6, generator=g, dtype=torch.complex64).to(device)
        out = kornia.contrib.combine_tensor_patches(kornia.contrib.extract_tensor_patches(img, 2, 1), (4, 6), 2, 1)
        assert out.dtype == torch.complex64
        self.assert_close(out, img)

    def test_eps_is_deprecated_and_has_no_effect(self, device, dtype):
        # An eps added to every divisor changed every pixel; clamp_min(eps) would turn a singly covered 7 into 3.5 at
        # eps=2, and eps=0 or eps=-1 gave NaN or inf. Patches that do not fit (6 patches of a 4 x 6 image recombined
        # to 5 x 7, the warned path) leave row 4 and column 6 uncovered: they stay 0 for every eps.
        img = torch.arange(1, 25, device=device, dtype=dtype).view(1, 1, 4, 6)
        patches = kornia.contrib.extract_tensor_patches(img, 2, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # the existing "does not fit" warning
            expected = kornia.contrib.combine_tensor_patches(patches, (5, 7), 2, 2)
            assert (expected[..., 4, :] == 0).all()
            assert (expected[..., :, 6] == 0).all()
            self.assert_close(expected[..., :4, :6], img)
            for eps in (1e-8, 2.0, 0.0, -1.0):
                with pytest.warns(DeprecationWarning, match="eps"):
                    out = kornia.contrib.combine_tensor_patches(patches, (5, 7), 2, 2, eps=eps)
                assert torch.equal(out, expected)

    @pytest.mark.parametrize("half", [torch.float16, torch.bfloat16])
    def test_half_precision_heavy_overlap_round_trip_is_exact(self, half, device):
        # 289 overlapping 0..255 values summed in float16 overflowed (51 inf); bfloat16 was off by 64.
        g = torch.Generator().manual_seed(0)
        img = (torch.rand(1, 3, 40, 52, generator=g) * 255).round().to(device, half)
        out = kornia.contrib.combine_tensor_patches(kornia.contrib.extract_tensor_patches(img, 17, 1), (40, 52), 17, 1)
        assert out.dtype == half
        assert torch.equal(out, img)

    def test_convention_overlaps_are_averaged_where_fold_sums_them(self, device, dtype):
        patches = torch.ones(1, 2, 1, 2, 2, device=device, dtype=dtype)
        patches[0, 1] = 3.0
        out = kornia.contrib.combine_tensor_patches(patches, (2, 3), 2, 1)
        self.assert_close(out[0, 0], torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device=device, dtype=dtype))
        # torch.nn.functional.fold sums the overlap: [[1, 4, 3], [1, 4, 3]]
        folded = torch.nn.functional.fold(patches.permute(0, 2, 3, 4, 1).flatten(1, 3), (2, 3), 2, stride=1)
        assert folded[0, 0, 0].tolist() == [1.0, 4.0, 3.0]
        # Patches that do not fit (the warned path) leave the last row and column uncovered: they are 0.
        img = torch.arange(1, 25, device=device, dtype=dtype).view(1, 1, 4, 6)
        with pytest.warns(UserWarning, match="will not fit"):
            out = kornia.contrib.combine_tensor_patches(kornia.contrib.extract_tensor_patches(img, 2, 2), (5, 7), 2, 2)
        assert (out[0, 0, -1] == 0).all()
        assert (out[0, 0, :, -1] == 0).all()
        self.assert_close(out[..., :4, :6], img)

    def test_convention_sibling_defaults_differ(self):
        assert kornia.contrib.ExtractTensorPatches(2).stride == 1
        assert inspect.signature(kornia.contrib.extract_tensor_patches).parameters["stride"].default == 1
        assert kornia.contrib.CombineTensorPatches((4, 6), (2, 3)).stride == (2, 3)
        assert kornia.contrib.compute_padding((5, 7), 3) == kornia.contrib.compute_padding((5, 7), 3, 3)
        function = inspect.signature(kornia.contrib.combine_tensor_patches).parameters
        assert function["stride"].default is inspect.Parameter.empty
        # The function and the module take allow_auto_unpadding and unpadding in opposite positional order.
        assert list(function)[4:6] == ["allow_auto_unpadding", "unpadding"]
        module = inspect.signature(kornia.contrib.CombineTensorPatches).parameters
        assert list(module)[3:5] == ["unpadding", "allow_auto_unpadding"]
