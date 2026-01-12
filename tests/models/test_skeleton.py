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

from kornia.models.paligemma import PaliGemma, PaliGemmaConfig


def test_skeleton():
    print("🚀 Starting PaliGemma Skeleton Test...")

    print("1. Loading configuration...")
    config = PaliGemmaConfig()

    config.hidden_size = 128
    config.intermediate_size = 512
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.head_dim = 32
    config.vocab_size = 1000

    print("2. Initializing model...")
    model = PaliGemma(config)
    model.eval()

    print("3. Creating dummy inputs...")

    pixel_values = torch.randn(1, 3, 224, 224)

    input_ids = torch.randint(0, config.vocab_size, (1, 10))

    print("4. Running forward pass...")
    try:
        logits = model(input_ids=input_ids, pixel_values=pixel_values)

        print("\n✅ Forward pass successful!")
        print(f"   Output Shape: {logits.shape}")

        expected_seq_len = 256 + 10
        if logits.shape == (1, expected_seq_len, config.vocab_size):
            print("✅ Shape Check Passed: Image and Text merged correctly!")
        else:
            print(f"⚠️ Shape Warning: Expected [1, {expected_seq_len}, {config.vocab_size}], but got {logits.shape}")

    except Exception as e:
        print(f"\n❌ Error during forward pass: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_skeleton()
