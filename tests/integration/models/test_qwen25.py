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

"""Integration test for the Qwen2.5-VL vision tower on the released weights.

Set ``QWEN25_VL_WEIGHTS_DIR`` to a local copy of ``Qwen/Qwen2.5-VL-3B-Instruct`` holding
``model.safetensors.index.json`` and the shard with the ``visual.*`` tensors
(``model-00001-of-00002.safetensors``, 3.98 GB). The test is skipped otherwise.
"""

import os

import pytest
import torch

from kornia.models.qwen25 import Qwen25VisionTransformer

from testing.base import assert_close


def test_qwen25_vision_official_weights():
    """Match Hugging Face ``transformers`` v4.51.0 on the released 3B weights.

    The expected values come from ``Qwen2_5_VisionTransformerPretrainedModel`` (eager attention, float32) with
    the same ``visual.*`` tensors, fed the patches its ``Qwen2VLImageProcessor`` makes from this image with
    resizing, rescaling and normalization disabled. Over the whole ``(30, 2048)`` output the two agree to 1.1e-4,
    for values up to 28.2.
    """
    weights_dir = os.environ.get("QWEN25_VL_WEIGHTS_DIR")
    if not weights_dir:
        pytest.skip("QWEN25_VL_WEIGHTS_DIR is not set")
    if not os.path.exists(os.path.join(weights_dir, "model.safetensors.index.json")):
        pytest.skip(f"No model.safetensors.index.json in {weights_dir}")

    model = Qwen25VisionTransformer.from_pretrained(weights_dir).eval()

    # 168 x 140 is 6 x 5 merged cells, so the 4 x 4 windows are uneven along both axes.
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, 168), torch.linspace(-1, 1, 140), indexing="ij")
    images = torch.stack([torch.sin(3 * xs + ys), torch.cos(2 * ys - xs), xs * ys]).unsqueeze(0)

    with torch.no_grad():
        out = model(images)

    expected = torch.tensor(
        [
            [-1.641796, -1.128161, 0.735769, 1.44872, -1.607152, 1.058822],
            [0.157594, -1.174068, 0.210073, -0.051154, -0.719309, 0.046013],
        ]
    )
    assert out.shape == (1, 30, 2048)
    assert_close(out[0, [0, 29], :6], expected, rtol=1e-3, atol=1e-3)
