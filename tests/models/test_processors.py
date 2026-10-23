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

from kornia.models.processors import ResizePostProcessor, ResizePreProcessor

from testing.base import BaseTester

# Image sizes that a half-precision image dtype cannot hold. bfloat16 keeps 8 significant bits, so 667 rounds to 668,
# 1001 to 1000 and 333 to 332; float16 keeps 11, so 4097 rounds to 4096 and 2051 to 2052.
_HALF_SIZE_CASES = [
    pytest.param(torch.bfloat16, [(667, 1000)], False, id="bfloat16-batch-of-one"),
    pytest.param(torch.bfloat16, [(667, 1000)], True, id="bfloat16-list-of-one"),
    pytest.param(torch.float16, [(4097, 2051)], True, id="float16-list-of-one"),
    pytest.param(torch.bfloat16, [(667, 1000), (1001, 333)], True, id="bfloat16-list-of-two"),
]


class TestResizePreProcessor(BaseTester):
    @pytest.mark.parametrize(("image_dtype", "sizes", "as_list"), _HALF_SIZE_CASES)
    def test_resize_preprocessor_records_int64_sizes(self, device, image_dtype, sizes, as_list):
        images = [torch.rand(3, h, w, device=device).to(image_dtype) for h, w in sizes]

        resized, original_sizes = ResizePreProcessor(8, 8)(images if as_list else torch.stack(images))

        assert resized.shape == (len(sizes), 3, 8, 8)
        assert original_sizes.dtype == torch.int64
        assert original_sizes.device == device
        assert original_sizes.tolist() == [list(size) for size in sizes]

    def test_tensor_batch_matches_the_per_image_resize(self, device, dtype):
        # A tensor batch is resized in one call, which keeps its batch dynamic in an ONNX export.
        images = torch.rand(3, 3, 13, 17, device=device, dtype=dtype)

        resized, original_sizes = ResizePreProcessor(8, 9)(images)
        per_image, per_image_sizes = ResizePreProcessor(8, 9)(list(images))

        self.assert_close(resized, per_image)
        assert original_sizes.tolist() == per_image_sizes.tolist() == [[13, 17]] * 3


class TestResizePostProcessor(BaseTester):
    @pytest.mark.parametrize(
        ("image_dtype", "sizes"), [(torch.bfloat16, [(667, 1000)]), (torch.bfloat16, [(667, 1000), (1001, 333)])]
    )
    def test_resize_postprocessor_returns_original_size(self, device, dtype, image_dtype, sizes):
        images = [torch.rand(3, h, w, device=device).to(image_dtype) for h, w in sizes]
        _, original_sizes = ResizePreProcessor(8, 8)(images)
        maps = torch.rand(len(sizes), 1, 8, 8, device=device, dtype=dtype)

        out = ResizePostProcessor()(maps, original_sizes)

        # a list of (1, C, H, W) maps, each at its image's original size
        assert isinstance(out, list)
        assert [tuple(item.shape) for item in out] == [(1, 1, h, w) for h, w in sizes]
