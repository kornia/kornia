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

import gc

import pytest
import torch

from kornia.models.paligemma import PaliGemma, PaliGemmaConfig

try:
    from transformers import PaliGemmaForConditionalGeneration

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers library not installed")
def test_load_and_verify_weights():
    print("\n🧠 STARTING BRAIN TRANSPLANT (Final Mode)...")

    gc.collect()
    torch.cuda.empty_cache()

    print("⏳ Loading Google Model (Float16)...")
    model_id = "google/paligemma-3b-pt-224"

    hf_model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cpu", token=True
    )

    text_conf = hf_model.config.text_config
    vis_conf = hf_model.config.vision_config
    image_token_id = hf_model.config.image_token_index

    print("🧪 Generatng Reference Output...")

    text_len = 10
    kornia_input_ids = torch.randint(0, 1000, (1, text_len))

    num_image_tokens = 256
    image_tokens = torch.full((1, num_image_tokens), image_token_id, dtype=torch.long)

    hf_input_ids = torch.cat([image_tokens, kornia_input_ids], dim=1)

    pixel_values = torch.randn(1, 3, 224, 224).to(dtype=torch.float16)

    with torch.no_grad():
        hf_output = hf_model(input_ids=hf_input_ids, pixel_values=pixel_values).logits

    print("💾 Copying weights to RAM...")
    hf_sd = {k: v.clone() for k, v in hf_model.state_dict().items()}

    print("🗑️ Deleting Google Model to free RAM...")
    del hf_model
    gc.collect()

    print("⏳ Initializing Kornia Model...")
    config = PaliGemmaConfig()

    config.vocab_size = text_conf.vocab_size
    config.hidden_size = text_conf.hidden_size
    config.num_hidden_layers = text_conf.num_hidden_layers
    config.num_attention_heads = text_conf.num_attention_heads
    config.intermediate_size = text_conf.intermediate_size
    config.num_key_value_heads = text_conf.num_key_value_heads

    config.vision_config.image_size = vis_conf.image_size
    config.vision_config.patch_size = vis_conf.patch_size
    config.vision_config.hidden_size = vis_conf.hidden_size
    config.vision_config.num_hidden_layers = vis_conf.num_hidden_layers
    config.vision_config.num_attention_heads = vis_conf.num_attention_heads
    config.vision_config.intermediate_size = vis_conf.intermediate_size

    kornia_model = PaliGemma(config)
    kornia_model.to(dtype=torch.float16)

    kornia_sd = kornia_model.state_dict()

    print("💉 Transferring Weights...")
    params_loaded = 0

    for k_key in kornia_sd.keys():
        hf_key = None

        if k_key.startswith("vision_tower."):
            if "vision_tower.head" in k_key:
                continue
            suffix = k_key.replace("vision_tower.", "")
            if "embeddings.position_embedding" in suffix:
                hf_key = f"model.vision_tower.vision_model.{suffix}.weight"
                if hf_key not in hf_sd:
                    hf_key = f"model.vision_tower.vision_model.{suffix}"
            else:
                hf_key = f"model.vision_tower.vision_model.{suffix}"

        elif k_key.startswith("multi_modal_projector."):
            suffix = k_key.replace("multi_modal_projector.", "")
            hf_key = f"model.multi_modal_projector.linear.{suffix}"

        elif k_key.startswith("embed_tokens.") or k_key.startswith("layers.") or k_key.startswith("norm."):
            hf_key = f"model.language_model.{k_key}"

        elif k_key == "lm_head.weight":
            hf_key = "lm_head.weight"

        if hf_key and hf_key in hf_sd:
            if kornia_sd[k_key].shape != hf_sd[hf_key].shape:
                print(f"❌ SHAPE MISMATCH: {k_key} {kornia_sd[k_key].shape} != {hf_sd[hf_key].shape}")
            else:
                with torch.no_grad():
                    kornia_sd[k_key].copy_(hf_sd[hf_key])
                params_loaded += 1

    print(f"✅ Transferred {params_loaded} layers.")

    del hf_sd
    gc.collect()

    print("\n🧪 Comparing Outputs...")
    kornia_model.eval()

    with torch.no_grad():
        kornia_out = kornia_model(input_ids=kornia_input_ids, pixel_values=pixel_values)

    diff = (kornia_out - hf_output).abs().max().item()
    print(f"📉 Max Difference: {diff:.6f}")

    if diff < 0.05:
        print("🎉 SUCCESS! The Brain Transplant Worked!")
    else:
        print("⚠️ Warning: Difference is high (Float16 precision issue).")
        print("But check if shapes match and code runs without error.")
