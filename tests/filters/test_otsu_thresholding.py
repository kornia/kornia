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

from kornia.filters.otsu_thresholding import OtsuThreshold, otsu_threshold

from testing.base import BaseTester, assert_close


class TestOtsuThreshold(BaseTester):
    def test_smoke(self, device, dtype):
        img = torch.rand(1, 3, 5, 5, device=device, dtype=dtype)
        op = OtsuThreshold()
        thresh_result, _thresh_value = op(img, nbins=4)
        assert thresh_result.shape == img.shape

    @pytest.mark.parametrize("input_shape", [(3, 3), (1, 3, 3), (1, 1, 3, 3), (2, 1, 1, 3, 3)])
    def test_transform_input_shapes(self, input_shape, device, dtype):
        img = torch.rand(input_shape, device=device, dtype=dtype)
        op = OtsuThreshold()
        flat, orig_shape = op.transform_input(img)
        assert orig_shape == img.shape
        assert flat.ndim == 2

    def test_otsu_threshold_consistency(self, device, dtype):
        torch.manual_seed(0)
        img = torch.rand(1, 4, 6, 1, device=device, dtype=dtype)
        out_func_tensor, _out_func_value = otsu_threshold(img, nbins=3, return_mask=False)
        out_class_tensor, _out_class_value = OtsuThreshold()(img, nbins=3)
        assert_close(out_func_tensor, out_class_tensor)

    def test_invalid_dim(self, device, dtype):
        img = torch.rand(1, 1, 1, 1, 3, 3, device=device, dtype=dtype)
        op = OtsuThreshold()
        with pytest.raises(ValueError, match="Unsupported tensor dimensionality"):
            op.transform_input(img)

    def test_gradcheck(self, device, dtype):
        img = torch.rand(1, 1, 5, 5, device=device, dtype=dtype, requires_grad=True)
        self.gradcheck(otsu_threshold, (img, 3, True, False))

    def test_differentiable_tensor_otsu(self, device, dtype):
        differentiable_input = torch.rand(1, 1, 5, 5, device=device, dtype=dtype, requires_grad=True)

        input = differentiable_input.clone().detach().requires_grad_(False)

        op = OtsuThreshold()
        diff_thresh_result, _diff_thresh_value = op(input, slow_and_differentiable=True)
        thresh_result, _thresh_value = op(input)
        self.assert_close(diff_thresh_result, thresh_result)

    def test_threshold_result(self, device, dtype):
        input = torch.tensor(
            [[10, 10, 10, 10], [10, 10, 10, 10], [200, 200, 200, 200], [200, 200, 200, 200]], device=device, dtype=dtype
        )

        expected = torch.tensor(
            [[0, 0, 0, 0], [0, 0, 0, 0], [200, 200, 200, 200], [200, 200, 200, 200]], device=device, dtype=dtype
        )

        op = OtsuThreshold()
        thresh_result, _thresh_value = op(input)
        self.assert_close(thresh_result, expected)

    def test_gradual_threshold(self, device, dtype):
        input = torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]], device=device, dtype=dtype)

        expected = torch.tensor([[0, 0, 0], [0, 50, 60], [70, 80, 90]], device=device, dtype=dtype)

        op = OtsuThreshold()
        thresh_result, _thresh_value = op(input)
        self.assert_close(thresh_result, expected)

    def test_uniform_result(self, device, dtype):
        input = torch.tensor(
            [[10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10]], device=device, dtype=dtype
        )

        expected = torch.tensor(
            [[10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10]], device=device, dtype=dtype
        )

        op = OtsuThreshold()
        thresh_result, _thresh_value = op(input)
        self.assert_close(thresh_result, expected)


def test_mask(device, dtype):
    input = torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]], device=device, dtype=dtype)

    expected = torch.tensor([[0, 0, 0], [0, 1, 1], [1, 1, 1]], device=device, dtype=torch.bool)

    thresh_result, _thresh_value = otsu_threshold(input, return_mask=True)
    assert_close(thresh_result, expected)


@pytest.mark.parametrize("slow_and_differentiable", [False, True])
@pytest.mark.parametrize(
    "values",
    [(-1.0, 0.0), (-2.0, -1.0), (0.0, 1.0), (1.0, 2.0)],
)
def test_mask_is_the_comparison_5173(values, slow_and_differentiable, device, dtype):
    # #5173: the mask used to be `result > 0`, so a foreground pixel of value 0 or below, above the threshold but not
    # above 0, came out False. The mask is `x > threshold`, whatever the sign of the data.
    low, high = values
    x = torch.tensor([[low, low, high, high]], device=device, dtype=dtype)
    mask, threshold = otsu_threshold(x, slow_and_differentiable=slow_and_differentiable, return_mask=True)
    assert mask.dtype == torch.bool
    assert_close(mask, x > threshold)
    assert mask.tolist() == [[False, False, True, True]]


def test_mask_per_channel_threshold_5173(device, dtype):
    # Each channel of a (B, C, H, W) input has its own threshold; the mask follows its channel's comparison and keeps
    # the input shape. Channel 0 is non-positive data, channel 1 positive.
    x = torch.tensor([[[[-1.0, -1.0], [0.0, 0.0]], [[1.0, 1.0], [2.0, 2.0]]]], device=device, dtype=dtype)
    mask, threshold = otsu_threshold(x, return_mask=True)
    assert mask.shape == x.shape
    assert threshold.shape == (2,)
    assert_close(mask, x > threshold.reshape(1, 2, 1, 1))
    assert mask[0, 0].tolist() == [[False, False], [True, True]]
    assert mask[0, 1].tolist() == [[False, False], [True, True]]


@pytest.mark.parametrize("slow_and_differentiable", [False, True])
def test_mask_agrees_with_the_thresholded_image_5173(slow_and_differentiable, device):
    # Integer input: the threshold is truncated to the input dtype, so on the fast path one pixel (0) equals it, and a
    # `>=` in the mask would keep that pixel while the thresholded image drops it.
    x = torch.tensor([[-30, -20, -10], [0, 10, 20], [30, 40, 50]], device=device)
    image, threshold = otsu_threshold(x, slow_and_differentiable=slow_and_differentiable)
    mask, _ = otsu_threshold(x, slow_and_differentiable=slow_and_differentiable, return_mask=True)
    assert torch.equal(mask, x > threshold)
    assert torch.equal(image, torch.where(mask, x, torch.zeros_like(x)))
    assert slow_and_differentiable or (x == threshold).any()  # the fixture does contain a pixel equal to the threshold


@pytest.mark.parametrize("shape", [(1, 3, 5, 5), (2, 1, 10, 10)])
def test_otsu_threshold_basic(shape, device, dtype):
    img = torch.rand(shape, device=device, dtype=dtype)
    thresh_result, _thresh_value = otsu_threshold(img)
    assert thresh_result.shape == img.shape
