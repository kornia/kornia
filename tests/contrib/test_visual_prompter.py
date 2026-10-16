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
from torch import nn

from kornia.contrib.visual_prompter import VisualPrompter
from kornia.core.exceptions import BaseError
from kornia.geometry.boxes import Boxes
from kornia.geometry.transform import resize
from kornia.models.sam import Sam, SamConfig
from kornia.models.structures import SegmentationResults

from testing.base import BaseTester, supports_reflect_padding


class _StubImageEncoder(nn.Module):
    """Stand-in for SAM's image encoder: a fixed, seeded ``(B, 256, 64, 64)`` embedding.

    It keeps ``img_size`` (read by ``VisualPrompter.preprocess_image``). A random MobileSAM encoder at 1024 x 1024
    takes 0.7 s in float32 but 16-18 s in float16 or bfloat16 on CPU with two threads, and what these tests check sits
    on the prompt side.
    """

    img_size = 1024

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        # SAM's 1024-pixel input over 16-pixel patches gives a 64 x 64 grid of 256-channel (``prompt_embed_dim``)
        # embeddings.
        self.embedding = torch.randn(1, 256, 64, 64, generator=generator).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding.expand(x.shape[0], -1, -1, -1)


def _skip_float64_on_mps(device: torch.device, dtype: torch.dtype) -> None:
    if device.type == "mps" and dtype == torch.float64:
        pytest.skip("MPS does not support float64")


def _prompter_with_stub_encoder(device: torch.device, dtype: torch.dtype) -> VisualPrompter:
    """Random-init MobileSAM prompter in evaluation mode, with the stub encoder and a seeded 77 x 128 image set."""
    _skip_float64_on_mps(device, dtype)
    prompter = VisualPrompter(SamConfig("mobile_sam"), device=device, dtype=dtype)
    prompter.model.eval()
    prompter.model.image_encoder = _StubImageEncoder(device, dtype)
    generator = torch.Generator().manual_seed(0)
    prompter.set_image(torch.rand(3, 77, 128, generator=generator).to(device=device, dtype=dtype))
    return prompter


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
            pytest.skip("the real MobileSAM encoder runs here; test_predict_in_every_dtype covers the other dtypes")
        prompter = VisualPrompter(SamConfig("mobile_sam"), device=device, dtype=dtype)
        prompter.model.eval()

        # One image at a time: a batch of images goes through ``Sam.forward``.
        with pytest.raises(BaseError, match=r"Sam\.forward"):
            prompter.set_image(torch.rand(2, 3, 256, 256, device=device, dtype=dtype))
        assert not prompter.is_image_set

        prompter.set_image(torch.rand(1, 3, 256, 256, device=device, dtype=dtype))
        assert prompter.is_image_set
        # SAM's 1024-pixel input over its 16-pixel patches gives a 64 x 64 grid of 256-channel embeddings.
        assert prompter.image_embeddings.shape == (1, 256, 64, 64)

        # The batch axis of the prompts is the prompt axis K: two box prompts for the one image.
        boxes_tensor = torch.tensor([[[10.0, 10.0, 50.0, 50.0], [20.0, 20.0, 80.0, 80.0]]], device=device, dtype=dtype)

        results = prompter.predict(boxes=boxes_tensor)

        assert results.logits.shape == (2, 3, 256, 256)
        assert results.scores.shape == (2, 3)

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

    def test_tensor_box_reaches_the_model(self, device, dtype):
        prompter = _prompter_with_stub_encoder(device, dtype)
        box = torch.tensor([[10.0, 20.0, 60.0, 50.0]], device=device, dtype=dtype)

        from_tensor = prompter.preprocess_prompts(boxes=box).boxes
        from_boxes = prompter.preprocess_prompts(boxes=Boxes.from_tensor(box[None], mode="xyxy")).boxes

        assert from_tensor is not None
        self.assert_close(from_tensor, from_boxes)
        assert not torch.equal(prompter.predict(boxes=box).logits, prompter.predict().logits)

    def test_keypoints_without_labels_raise(self, device, dtype):
        prompter = _prompter_with_stub_encoder(device, dtype)

        with pytest.raises(BaseError, match="keypoints_labels"):
            prompter.predict(keypoints=torch.tensor([[[40.0, 30.0]]], device=device, dtype=dtype))

    def test_prompts_follow_the_half_pixel_resize(self, device, dtype):
        prompter = _prompter_with_stub_encoder(device, dtype)
        # The long side 128 -> 1024 gives the scale s = 8 on both axes, and the short side 77 * 8 = 616.
        assert prompter._input_image_size == (616, 1024)

        keypoints = torch.tensor([[[40.0, 30.0]], [[0.0, 0.0]]], device=device, dtype=dtype)
        labels = torch.ones(2, 1, device=device, dtype=dtype)
        points = prompter.preprocess_prompts(keypoints, labels).keypoints

        # Half-pixel map x' = (x + 0.5) * s - 0.5 (``Resize`` at ``align_corners=False``): (40, 30) ->
        # (40.5 * 8 - 0.5, 30.5 * 8 - 0.5) = (323.5, 243.5); pixel 0's centre lands on the centre of output pixels
        # 0..7, at 3.5. The corner-aligned map x * 1023 / 127, y * 615 / 76 gives (322.20, 242.76) and (0, 0).
        expected = torch.tensor([[[323.5, 243.5]], [[3.5, 3.5]]], device=device, dtype=dtype)
        self.assert_close(points, expected)

    @pytest.mark.parametrize("height, width", [(77, 128), (1500, 2000)])
    def test_resize_round_trip_is_aligned(self, device, dtype, height, width):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("a ramp of 2000 columns needs float32 or wider to resolve 1e-3 px")
        _skip_float64_on_mps(device, dtype)
        prompter = VisualPrompter(SamConfig("mobile_sam"), device=device, dtype=dtype)

        # A horizontal ramp whose value is its own column index, centred on 0 to halve its float32 rounding, goes
        # through the forward resize, the padding to 1024, and the resize back of ``SegmentationResults``.
        ramp = (torch.arange(width, device=device, dtype=dtype) - width / 2).expand(1, 1, height, width).contiguous()
        resized = prompter.transforms(ramp, data_keys=["input"])
        input_size = (resized.shape[-2], resized.shape[-1])
        no_normalization = torch.zeros(1, device=device, dtype=dtype), torch.ones(1, device=device, dtype=dtype)
        padded = prompter.preprocess_image(resized, *no_normalization)
        back = SegmentationResults(padded, torch.zeros(1, 1, device=device, dtype=dtype)).original_res_logits(
            input_size, (height, width), (1024, 1024)
        )

        # The three outer columns on each side are left out: at 2000 -> 1024 the antialiasing blur (3 taps,
        # reflected at the border) bends the ramp in column 0, which the two bilinear resizes then spread to
        # columns 1 and 2 (column 0 moves by 0.57 px, column 2 by 0.02 px). A corner-aligned forward resize stretches
        # the ramp about its centre, by up to 0.43 px at 77 x 128 and 0.48 px at 1500 x 2000, most at the borders.
        row = height // 2
        self.assert_close(back[0, 0, row, 3:-3], ramp[0, 0, row, 3:-3], rtol=0.0, atol=1e-3)

    def test_downscale_is_antialiased(self, device, dtype):
        _skip_float64_on_mps(device, dtype)
        # The antialiasing Gaussian blur pads by reflection; torch 2.5.1 has no float16 CPU reflection_pad2d.
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        prompter = VisualPrompter(SamConfig("mobile_sam"), device=device, dtype=dtype)
        generator = torch.Generator().manual_seed(0)
        image = torch.rand(1, 3, 1500, 2000, generator=generator).to(device=device, dtype=dtype)

        # The long side 2000 -> 1024 scales by 0.512, so the short side is 1500 * 0.512 = 768.
        expected = resize(image, (768, 1024), align_corners=False, antialias=True)
        self.assert_close(prompter.transforms(image, data_keys=["input"]), expected)

    def test_predict_in_every_dtype(self, device, dtype):
        prompter = _prompter_with_stub_encoder(device, dtype)
        keypoints = torch.tensor([[[40.0, 30.0]]], device=device, dtype=dtype)
        labels = torch.ones(1, 1, device=device, dtype=dtype)
        box = torch.tensor([[10.0, 20.0, 60.0, 50.0]], device=device, dtype=dtype)

        for results in (prompter.predict(), prompter.predict(keypoints, labels), prompter.predict(boxes=box)):
            assert results.logits.dtype == dtype
            assert results.logits.shape == (1, 3, 256, 256)

    def test_set_image_rejects_a_batch(self, device, dtype):
        _skip_float64_on_mps(device, dtype)
        prompter = VisualPrompter(SamConfig("mobile_sam"), device=device, dtype=dtype)
        prompter.model.image_encoder = _StubImageEncoder(device, dtype)

        with pytest.raises(BaseError, match=r"Sam\.forward"):
            prompter.set_image(torch.rand(2, 3, 77, 128, device=device, dtype=dtype))

        prompter.set_image(torch.rand(1, 3, 77, 128, device=device, dtype=dtype))
        assert prompter.is_image_set
        assert prompter.image_embeddings.shape == (1, 256, 64, 64)

    def test_box_prompt_shapes(self, device, dtype):
        prompter = _prompter_with_stub_encoder(device, dtype)

        # (1, K, 4) holds the K boxes of the one image, as K = 3 prompts of 3 masks each.
        three = torch.tensor(
            [[[10.0, 20.0, 60.0, 50.0], [5.0, 5.0, 120.0, 70.0], [30.0, 10.0, 90.0, 40.0]]], device=device, dtype=dtype
        )
        for boxes in (three, Boxes.from_tensor(three, mode="xyxy")):
            assert prompter.predict(boxes=boxes).logits.shape == (3, 3, 256, 256)

        # A leading size of 2 would be boxes for two images.
        two = torch.tensor([[[10.0, 20.0, 60.0, 50.0]], [[5.0, 5.0, 120.0, 70.0]]], device=device, dtype=dtype)
        for boxes in (two, Boxes.from_tensor(two, mode="xyxy")):
            with pytest.raises(BaseError, match=r"Sam\.forward"):
                prompter.predict(boxes=boxes)

    def test_gradcheck(self, device):
        pytest.skip("Gradcheck is not currently applicable for VisualPrompter inference.")

    def test_cardinality(self, device, dtype):
        pytest.skip("Cardinality is covered by the test_batching_pipeline.")

    def test_dynamo(self, device, dtype, torch_optimizer):
        pytest.skip("Dynamo compilation currently broken for VisualPrompter. See #FIXME in source.")
