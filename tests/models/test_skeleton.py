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

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import torch

from kornia.models.paligemma import PaliGemmaConfig, PaliGemmaForConditionalGeneration


def test_skeleton():
    print("1. Loading Config...")
    config = PaliGemmaConfig()
    print("✅ Config Loaded!")

    print("2. Initializing Model (Skeleton)...")
    model = PaliGemmaForConditionalGeneration(config)
    print(f"✅ Model Initialized! Vision Tower: {type(model.vision_tower)}")

    print("3. Testing Forward Pass (Dummy)...")
    # Create a dummy image (Batch=1, Channels=3, Height=224, Width=224)
    pixel_values = torch.randn(1, 3, 224, 224)
    # Create dummy input_ids (Batch=1, SeqLen=10) - technically not used yet but good for checking signature
    input_ids = torch.randint(0, 1000, (1, 10))

    output = model(input_ids, pixel_values)
    print(f"✅ Forward Pass Complete! Output shape: {output.shape}")
    print("   (Should be [1, 256, 2048] -> 1 batch, 256 patches, 2048 hidden dim)")


if __name__ == "__main__":
    test_skeleton()
