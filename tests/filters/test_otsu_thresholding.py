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

from cv2 import threshold
import pytest
import torch

from kornia.filters.otsu_thresholding import ThreshOtsu, otsu_threshold

from testing.base import BaseTester, assert_close


class TestThreshOtsu(BaseTester):
    def test_smoke(self, device, dtype):
        img = torch.rand(1, 3, 5, 5, device=device, dtype=dtype)
        op = ThreshOtsu()
        thresh_result, thresh_value = op(img, nbins=4)
        assert thresh_result.shape == img.shape

    @pytest.mark.parametrize("input_shape", [(3, 3), (1, 3, 3), (1, 1, 3, 3), (2, 1, 1, 3, 3)])
    def test_transform_input_shapes(self, input_shape, device, dtype):
        img = torch.rand(input_shape, device=device, dtype=dtype)
        op = ThreshOtsu()
        flat, orig_shape = op.transform_input(img)
        assert orig_shape == img.shape
        assert flat.ndim == 2

    def test_otsu_threshold_consistency(self, device, dtype):
        torch.manual_seed(0)
        img = torch.rand(1, 4, 6, 1, device=device, dtype=dtype)
        out_func_tensor, out_func_value = otsu_threshold(
            img, nbins=3, return_mask=False)
        out_class_tensor, out_class_value = ThreshOtsu()(img, nbins=3)
        assert_close(out_func_tensor, out_class_tensor)

    def test_invalid_dim(self, device, dtype):
        img = torch.rand(1, 1, 1, 1, 3, 3, device=device, dtype=dtype)
        op = ThreshOtsu()
        with pytest.raises(ValueError, match="Unsupported tensor dimensionality"):
            op.transform_input(img)

    def test_gradcheck(self, device):
        img = torch.rand(1, 1, 5, 5, device=device,
                         dtype=torch.float64, requires_grad=True)
        self.gradcheck(otsu_threshold, (img, 3, False))

    def test_threshold_result(self, device, dtype):
        input = torch.tensor(
            [[10, 10, 10, 10],
             [10, 10, 10, 10],
             [200, 200, 200, 200],
             [200, 200, 200, 200]],
            device=device,
            dtype=dtype
        )

        expected = torch.tensor(
            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [200, 200, 200, 200],
             [200, 200, 200, 200]],
            device=device,
            dtype=dtype
        )

        op = ThreshOtsu()
        thresh_result, thresh_value = op(input)
        self.assert_close(thresh_result, expected)

    def test_gradual_threshold(self, device, dtype):
        input = torch.tensor(
            [[10, 20, 30],
             [40, 50, 60],
             [70, 80, 90]],
            device=device,
            dtype=dtype
        )

        expected = torch.tensor(
            [[0, 0, 0],
             [0, 50, 60],
             [70, 80, 90]],
            device=device,
            dtype=dtype
        )

        op = ThreshOtsu()
        thresh_result, thresh_value = op(input)
        self.assert_close(thresh_result, expected)

    def test_uniform_result(self, device, dtype):
        input = torch.tensor(
            [[10, 10, 10, 10],
             [10, 10, 10, 10],
             [10, 10, 10, 10],
             [10, 10, 10, 10]],
            device=device,
            dtype=dtype
        )

        expected = torch.tensor(
            [[10, 10, 10, 10],
             [10, 10, 10, 10],
             [10, 10, 10, 10],
             [10, 10, 10, 10]],
            device=device,
            dtype=dtype
        )

        op = ThreshOtsu()
        thresh_result, thresh_value = op(input)
        self.assert_close(thresh_result, expected)


def test_mask(device, dtype):
    input = torch.tensor(
        [[10, 20, 30],
         [40, 50, 60],
         [70, 80, 90]],
        device=device,
        dtype=dtype
    )

    expected = torch.tensor(
        [[0, 0, 0],
         [0, 1, 1],
         [1, 1, 1]],
        device=device,
        dtype=torch.bool
    )

    thresh_result, thresh_value = otsu_threshold(
        input, return_mask=True)
    assert_close(thresh_result, expected)


@pytest.mark.parametrize("shape", [(1, 3, 5, 5), (2, 1, 10, 10)])
def test_otsu_threshold_basic(shape, device, dtype):
    img = torch.rand(shape, device=device, dtype=dtype)
    thresh_result, thresh_value = otsu_threshold(img)
    assert thresh_result.shape == img.shape
