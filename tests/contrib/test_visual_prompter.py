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

from types import SimpleNamespace

import pytest
import torch

from kornia.contrib.visual_prompter import VisualPrompter
from kornia.models.sam import Sam, SamConfig

from testing.base import BaseTester


class TestVisualPrompter(BaseTester):
    def test_default_config(self, monkeypatch):
        captured_configs = []

        class FakeSam:
            image_encoder = SimpleNamespace(img_size=1024)

            def to(self, **kwargs):
                return self

        def fake_from_config(config):
            captured_configs.append(config)
            return FakeSam()

        monkeypatch.setattr(Sam, "from_config", staticmethod(fake_from_config))

        prompter = VisualPrompter()

        assert prompter.model.image_encoder.img_size == 1024
        assert len(captured_configs) == 1
        assert captured_configs[0].model_type == "vit_h"
        assert captured_configs[0].pretrained is True

    def test_smoke(self, device, dtype):
        if dtype not in (torch.float32, torch.float16):
            pytest.skip("VisualPrompter (SAM) primarily supports float32 and float16")

        prompter = VisualPrompter(SamConfig("mobile_sam"), device=device, dtype=dtype)
        assert prompter is not None
        assert not prompter.is_image_set

    @pytest.mark.slow
    def test_batching_pipeline(self, device, dtype):
        if dtype not in (torch.float32, torch.float16):
            pytest.skip("VisualPrompter (SAM) primarily supports float32 and float16")
        prompter = VisualPrompter(SamConfig("mobile_sam"), device=device, dtype=dtype)

        batch_size = 2
        image = torch.rand(batch_size, 3, 256, 256).to(device=device, dtype=dtype)

        prompter.set_image(image)
        assert prompter.is_image_set
        assert prompter.image_embeddings.shape[0] == batch_size

        boxes_tensor = torch.tensor(
            [[[10.0, 10.0, 50.0, 50.0]], [[20.0, 20.0, 80.0, 80.0]]],
            device=device,
            dtype=dtype,
        )

        results = prompter.predict(boxes=boxes_tensor)

        assert results.logits.shape == (batch_size, 3, 256, 256)

    @pytest.mark.slow
    def test_predict_without_prompts(self, device, dtype):
        if dtype not in (torch.float32, torch.float16):
            pytest.skip("VisualPrompter (SAM) primarily supports float32 and float16")
        prompter = VisualPrompter(SamConfig("mobile_sam"), device=device, dtype=dtype)
        prompter.set_image(torch.rand(3, 256, 256, device=device, dtype=dtype))

        results = prompter.predict(multimask_output=True)  # no keypoints, boxes or masks

        assert results.logits.shape == (1, 3, 256, 256)
        assert results.scores.shape == (1, 3)

    @pytest.mark.device_agnostic
    def test_weighted_config_holds_an_eval_model(self, monkeypatch):
        monkeypatch.setattr(Sam, "load_checkpoint", lambda self, checkpoint, device=None: None)

        # ``training`` of the model and of every submodule: ``{False}`` is evaluation mode throughout.
        weighted = VisualPrompter(SamConfig("mobile_sam", pretrained=True)).model
        assert {m.training for m in weighted.modules()} == {False}
        random_init = VisualPrompter(SamConfig("mobile_sam")).model
        assert {m.training for m in random_init.modules()} == {True}

    @pytest.mark.slow
    def test_weighted_config_runs_in_half(self, monkeypatch, device, dtype):
        # A weighted config comes back in evaluation mode, and VisualPrompter casts the model after that.
        if dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("guards the cast of an evaluation-mode model to a half dtype")
        monkeypatch.setattr(Sam, "load_checkpoint", lambda self, checkpoint, device=None: None)
        prompter = VisualPrompter(SamConfig("mobile_sam", pretrained=True), device=device, dtype=dtype)

        prompter.set_image(torch.rand(3, 77, 128, device=device, dtype=dtype))

        # SAM's 1024-pixel input over its 16-pixel patches gives a 64 x 64 grid of 256-channel (``prompt_embed_dim``)
        # embeddings.
        assert prompter.image_embeddings.shape == (1, 256, 64, 64)
        assert prompter.image_embeddings.dtype == dtype

    def test_gradcheck(self, device):
        pytest.skip("Gradcheck is not currently applicable for VisualPrompter inference.")

    def test_cardinality(self, device, dtype):
        pytest.skip("Cardinality is covered by the test_batching_pipeline.")

    def test_dynamo(self, device, dtype, torch_optimizer):
        pytest.skip("Dynamo compilation currently broken for VisualPrompter. See #FIXME in source.")
