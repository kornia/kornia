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

import torch

from kornia.feature import LightGlue


def test_lightglue_empty_after_pruning():
    model = LightGlue(features="superpoint", width_confidence=0.99)
    model.eval()

    data = {
        "image0": {
            "keypoints": torch.empty(1, 0, 2),
            "descriptors": torch.empty(1, 0, 256),
            "image_size": torch.tensor([[640, 480]]),
        },
        "image1": {
            "keypoints": torch.empty(1, 0, 2),
            "descriptors": torch.empty(1, 0, 256),
            "image_size": torch.tensor([[640, 480]]),
        },
    }

    with torch.no_grad():
        out = model(data)

    assert out["matches0"].shape == (1, 0)
    assert out["matches1"].shape == (1, 0)
    assert out["matching_scores0"].shape == (1, 0)
    assert out["matching_scores1"].shape == (1, 0)
