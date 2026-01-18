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

import numpy as np
import torch
from PIL import Image

from kornia.models.depth_anything_3.architecture.dinov2 import DinoV2
from kornia.models.depth_anything_3.model import DepthAnything3
from kornia.models.depth_anything_3.specs import Prediction

from testing.base import BaseTester


class TestDa3(BaseTester):
    def test_dinov2(self):
        img = torch.rand(2, 3, 3, 518, 518)
        cfg = {
            "name": "vitb",
            "out_layers": [5, 7, 9, 11],
            "alt_start": 4,
            "qknorm_start": 4,
            "rope_start": 4,
            "cat_token": True,
        }

        dino = DinoV2(**cfg)
        results = dino(img)

        assert len(results) == 2

    def test_da3(self):
        images = []
        for _ in range(2):
            arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(arr, "RGB")
            images.append(img)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE", device=device)
        result = model.inference(images)

        assert type(result) == Prediction
