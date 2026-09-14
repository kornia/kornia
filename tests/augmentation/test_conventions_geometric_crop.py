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

from testing.base import BaseTester


class TestGeometricCropConventions(BaseTester):
    """Batch-6 convention pins for crop, resize, and pad augmentations."""

    def test_convention_center_crop_size_mode_and_inverse(self, device, dtype):
        x = torch.arange(48, device=device, dtype=dtype).reshape(1, 1, 6, 8)
        expected = x[..., 1:5, 2:6]

        slice_crop = K.CenterCrop(4, cropping_mode="slice")
        self.assert_close(slice_crop(x), expected)
        with pytest.raises(NotImplementedError, match="only applicable for resample"):
            slice_crop.inverse(expected)

        resample_crop = K.CenterCrop((4, 4), cropping_mode="resample")
        resample_output = resample_crop(x)
        assert resample_output.shape == expected.shape
        assert resample_crop.inverse(resample_output).shape == x.shape

        nearest = K.CenterCrop((4, 4), resample="nearest", align_corners=False, cropping_mode="resample")
        self.assert_close(nearest(x), expected)

    def test_wart_center_crop_slice_aliases_input_4413(self, device, dtype):
        # #4413: default slice mode returns its direct tensor slice. The write lands at the crop's centre offset.
        x = torch.arange(48, device=device, dtype=dtype).reshape(1, 1, 6, 8)
        output = K.CenterCrop(4, cropping_mode="slice")(x)
        assert output.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
        output[0, 0, 0, 0] = -99
        assert x[0, 0, 1, 2] == -99

    def test_convention_random_crop_padding_modes_and_inverse(self, device, dtype):
        x = torch.arange(9, device=device, dtype=dtype).reshape(1, 1, 3, 3)
        crop = K.RandomCrop((3, 3), padding=(1, 2, 3, 4), fill=9, p=1.0, cropping_mode="slice")
        assert crop.compute_padding((1, 1, 3, 3)) == [1, 3, 2, 4]
        torch.manual_seed(0)
        padded = crop(x)
        assert padded.shape == x.shape
        assert (padded == 9).any()

        needed = K.RandomCrop((4, 5), pad_if_needed=True, fill=9, p=1.0)
        assert needed.compute_padding((1, 1, 2, 3)) == [2, 2, 2, 2]
        prepared = needed.precrop_padding(torch.ones(1, 1, 2, 3, device=device, dtype=dtype))
        assert prepared.shape == (1, 1, 6, 7)
        assert (prepared[..., 0, :] == 9).all()

        resample = K.RandomCrop((3, 3), p=1.0, cropping_mode="resample")
        output = resample(x)
        assert resample.inverse(output).shape == x.shape
        slice_crop = K.RandomCrop((3, 3), p=1.0, cropping_mode="slice")
        slice_output = slice_crop(x)
        with pytest.raises(NotImplementedError, match="only applicable for resample"):
            slice_crop.inverse(slice_output)

        torch.manual_seed(0)
        nearest = K.RandomCrop((3, 3), resample="nearest", align_corners=False, p=1.0, cropping_mode="resample")
        nearest_output = nearest(x)
        left, top = nearest._params["src"][0, 0].to(dtype=torch.long).tolist()
        self.assert_close(nearest_output, x[..., top : top + 3, left : left + 3])

    def test_wart_random_crop_oversized_without_padding_upscales_4414(self, device, dtype):
        # #4414: a 10x10 crop from a 6x8 input without pad_if_needed is the bilinear 10x10 resize.
        x = torch.arange(48, device=device, dtype=dtype).reshape(1, 1, 6, 8) / 48
        output = K.RandomCrop((10, 10), pad_if_needed=False, p=1.0)(x)
        expected_row = torch.tensor(
            [0.0, 0.014583, 0.03125, 0.047917, 0.064583, 0.08125, 0.097917, 0.114583, 0.13125, 0.145833],
            device=device,
            dtype=dtype,
        )
        self.assert_close(output[0, 0, 0], expected_row, low_tolerance=True)

    def test_convention_random_resized_crop_size_modes_and_inverse(self, device, dtype):
        x = torch.arange(48, device=device, dtype=dtype).reshape(1, 1, 6, 8)
        kwargs = {"size": (3, 4), "scale": (1.0, 1.0), "ratio": (0.75, 0.75), "p": 1.0}
        for mode in ("slice", "resample"):
            crop = K.RandomResizedCrop(cropping_mode=mode, **kwargs)
            output = crop(x)
            assert output.shape == (1, 1, 3, 4)
            if mode == "resample":
                assert crop.inverse(output).shape == x.shape
            else:
                with pytest.raises(NotImplementedError, match="only applicable for resample"):
                    crop.inverse(output)

        nearest = K.RandomResizedCrop(
            size=(6, 8),
            scale=(1.0, 1.0),
            ratio=(0.75, 0.75),
            resample="nearest",
            align_corners=True,
            cropping_mode="resample",
            p=1.0,
        )
        params = nearest.forward_parameters(x.shape)
        params["src"] = torch.tensor([[[0, 0], [7, 0], [7, 5], [0, 5]]], device=device, dtype=dtype)
        params["dst"] = torch.tensor([[[0, 0], [7, 0], [7, 5], [0, 5]]], device=device, dtype=dtype)
        self.assert_close(nearest(x, params=params), x)

    def test_wart_crop_siblings_disagree_on_integer_size_4417(self):
        # #4417: CenterCrop accepts an int while its random siblings reject it through implementation details.
        assert K.CenterCrop(4).size == (4, 4)
        with pytest.raises(AssertionError):
            K.RandomCrop(4)(torch.ones(1, 1, 6, 8))  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            K.RandomResizedCrop(4)  # type: ignore[arg-type]

    def test_convention_resize_side_policies_and_inverse(self, device, dtype):
        x = torch.arange(70, device=device, dtype=dtype).reshape(1, 1, 7, 10)
        exact = K.Resize((3, 5), p=1.0)
        assert exact(x).shape == (1, 1, 3, 5)
        assert exact.inverse(exact(x)).shape == x.shape
        assert K.Resize(4, side="short", p=1.0)(x).shape == (1, 1, 4, 5)
        assert K.Resize(4, side="long", p=1.0)(x).shape == (1, 1, 2, 4)
        assert K.LongestMaxSize(4)(x).shape == (1, 1, 2, 4)
        assert K.SmallestMaxSize(4)(x).shape == (1, 1, 4, 5)

    def test_wart_pad_to_smaller_target_crops_and_inverse_is_noop_4410(self, device, dtype):
        # #4410: negative right/bottom padding crops, and the identity matrix leaves inverse unable to restore it.
        x = torch.arange(48, device=device, dtype=dtype).reshape(1, 1, 6, 8)
        crop = K.PadTo((4, 5), pad_value=9)
        output = crop(x)
        self.assert_close(output, x[..., :4, :5])
        self.assert_close(crop.inverse(output), output)
        self.assert_close(crop.transform_matrix, torch.eye(3, device=device, dtype=dtype)[None])

    def test_convention_pad_to_right_bottom_padding_round_trips(self, device, dtype):
        x = torch.arange(48, device=device, dtype=dtype).reshape(1, 1, 6, 8)
        pad = K.PadTo((10, 12), pad_value=9)
        output = pad(x)
        assert output.shape == (1, 1, 10, 12)
        self.assert_close(output[..., :6, :8], x)
        self.assert_close(output[..., 6:, :], torch.full((1, 1, 4, 12), 9, device=device, dtype=dtype))
        self.assert_close(pad.inverse(output), x)
