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
import torch.nn.functional as F
from torch import nn

import kornia.augmentation as K
from kornia.augmentation.container.augment import AugmentationSequential
from kornia.contrib.visual_prompter import VisualPrompter
from kornia.core.exceptions import BaseError
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints
from kornia.models.sam import Sam, SamConfig
from kornia.models.structures import SegmentationResults

from testing.base import BaseTester


class _StubImageEncoder(nn.Module):
    """Stand-in for SAM's image encoder: a fixed, seeded ``(B, 256, 64, 64)`` embedding.

    It keeps ``img_size`` (read by ``VisualPrompter.preprocess_image``) and records its input, the resized and padded
    image. A random MobileSAM encoder at 1024 x 1024 takes 0.7 s in float32 but 16-18 s in float16 or bfloat16 on CPU
    with two threads, and what these tests check sits before and after it.
    """

    img_size = 1024

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        # SAM's 1024-pixel input over 16-pixel patches gives a 64 x 64 grid of 256-channel (``prompt_embed_dim``)
        # embeddings.
        self.embedding = torch.randn(1, 256, 64, 64, generator=generator).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x
        return self.embedding.expand(x.shape[0], -1, -1, -1)


def _prompt_dtype(dtype: torch.dtype) -> torch.dtype:
    """The dtype of mapped prompts: at least float32, whatever the model dtype."""
    return torch.promote_types(dtype, torch.float32)


def _skip_float64_on_mps(device: torch.device, dtype: torch.dtype) -> None:
    if device.type == "mps" and dtype == torch.float64:
        pytest.skip("MPS does not support float64")


def _prompter_with_stub(device: torch.device, dtype: torch.dtype) -> tuple[VisualPrompter, _StubImageEncoder]:
    """Random-init MobileSAM prompter in evaluation mode, with the stub encoder and no image set."""
    _skip_float64_on_mps(device, dtype)
    prompter = VisualPrompter(SamConfig("mobile_sam"), device=device, dtype=dtype)
    prompter.model.eval()
    stub = _StubImageEncoder(device, dtype)
    prompter.model.image_encoder = stub
    return prompter, stub


def _prompter_with_stub_encoder(device: torch.device, dtype: torch.dtype) -> VisualPrompter:
    """Random-init MobileSAM prompter in evaluation mode, with the stub encoder and a seeded 77 x 128 image set."""
    prompter, _ = _prompter_with_stub(device, dtype)
    generator = torch.Generator().manual_seed(0)
    prompter.set_image(torch.rand(3, 77, 128, generator=generator).to(device=device, dtype=dtype))
    return prompter


class TestVisualPrompter(BaseTester):
    @pytest.mark.parametrize(
        "config, type_name",
        [
            pytest.param("vit_b", "str", id="model-name"),
            pytest.param({"model_type": "vit_b"}, "dict", id="config-dict"),
            pytest.param(nn.Identity(), "Identity", id="model-instance"),
        ],
    )
    def test_unsupported_config(self, config: object, type_name: str) -> None:
        """Unsupported configuration types report the accepted alternatives."""
        with pytest.raises(NotImplementedError, match=rf"config.*{type_name}.*SamConfig.*None"):
            VisualPrompter(config)

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
        _skip_float64_on_mps(device, dtype)
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
            pytest.skip("the real MobileSAM encoder runs here; test_predict_in_every_dtype covers the other dtypes")
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

        keypoints = torch.tensor([[[40.0, 30.0]]], device=device, dtype=dtype)
        for unlabelled in (keypoints, Keypoints.from_tensor(keypoints)):
            with pytest.raises(BaseError, match="keypoints_labels"):
                prompter.predict(keypoints=unlabelled)

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
        expected = torch.tensor([[[323.5, 243.5]], [[3.5, 3.5]]], device=device, dtype=_prompt_dtype(dtype))
        self.assert_close(points, expected)

    def test_box_edges_scale_with_the_image(self, device, dtype):
        prompter = _prompter_with_stub_encoder(device, dtype)
        # kornia's xyxy corners are pixel edges (width = x2 - x1); an edge maps to x * s, as both corners do in the
        # reference's ``apply_boxes``. At s = 8: the whole 77 x 128 image -> the whole 616 x 1024 resized image, and
        # (10, 20, 60, 50) -> (80, 160, 480, 400). The half-pixel map of a pixel centre would give (3.5, 3.5, 1027.5,
        # 619.5) for the whole image, past its far edges.
        cases = (
            ([0.0, 0.0, 128.0, 77.0], [0.0, 0.0, 1024.0, 616.0]),
            ([10.0, 20.0, 60.0, 50.0], [80.0, 160.0, 480.0, 400.0]),
        )
        for box, mapped in cases:
            box = torch.tensor([box], device=device, dtype=dtype)
            expected = torch.tensor([mapped], device=device, dtype=_prompt_dtype(dtype))
            for boxes in (box, Boxes.from_tensor(box[None], mode="xyxy")):
                self.assert_close(prompter.preprocess_prompts(boxes=boxes).boxes, expected)

    # Bound of each case: at 77 x 128 the bilinear upscale reproduces a ramp exactly. At 1500 x 2000 the
    # antialiasing filter of SAM's reference (torch's, the same as PIL's bilinear reduce) starts and ends its taps on
    # whole source pixels, which moves a ramp in a sawtooth of up to 0.022 px, in float64 as well.
    @pytest.mark.parametrize("height, width, bound", [(77, 128, 1e-3), (1500, 2000, 0.025)])
    def test_resize_round_trip_is_aligned(self, device, dtype, height, width, bound):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("a ramp of 2000 columns needs float32 or wider to resolve the bound")
        prompter, stub = _prompter_with_stub(device, dtype)

        # A horizontal ramp whose value is its own column index, centred on 0 to halve its float32 rounding, goes
        # through set_image (without normalization) and back through the resize of ``SegmentationResults``.
        ramp = (torch.arange(width, device=device, dtype=dtype) - width / 2).expand(3, height, width).contiguous()
        prompter.set_image(ramp, torch.zeros(3, device=device, dtype=dtype), torch.ones(3, device=device, dtype=dtype))
        back = SegmentationResults(stub.last_input[:, :1], torch.zeros(1, 1, device=device, dtype=dtype))
        back = back.original_res_logits(
            prompter._input_image_size, prompter._original_image_size, prompter._input_encoder_size
        )

        # The three outer columns on each side are left out: at 2000 -> 1024 the filter's taps are cut at the image
        # border, which bends the ramp in column 0 (0.69 px), and the resize back spreads it to columns 1 and 2
        # (0.15 and 0.04 px). A corner-aligned forward resize stretches the ramp about its centre, by up to 0.43 px at
        # 77 x 128 and 0.48 px at 1500 x 2000, most at the borders.
        row = height // 2
        self.assert_close(back[0, 0, row, 3:-3], ramp[0, row, 3:-3], rtol=0.0, atol=bound)

    def test_downscale_is_antialiased(self, device, dtype):
        prompter, stub = _prompter_with_stub(device, dtype)
        generator = torch.Generator().manual_seed(0)
        image = torch.rand(3, 1500, 2000, generator=generator).to(device=device, dtype=dtype)
        no_normalization = torch.zeros(3, device=device, dtype=dtype), torch.ones(3, device=device, dtype=dtype)

        prompter.set_image(image, *no_normalization)

        # The long side 2000 -> 1024 scales by 0.512, so the short side is 1500 * 0.512 = 768. SAM's reference,
        # ``ResizeLongestSide.apply_image_torch`` (segment-anything@3518c86), resizes with this interpolate call. It
        # runs here on the CPU (torch 2.5.1 has no MPS kernel for it) and in float32 (none for the half dtypes on CPU).
        compute_dtype = torch.promote_types(dtype, torch.float32)
        expected = F.interpolate(
            image[None].cpu().to(compute_dtype), (768, 1024), mode="bilinear", align_corners=False, antialias=True
        ).to(device=device, dtype=dtype)
        assert prompter._input_image_size == (768, 1024)
        self.assert_close(stub.last_input[..., :768, :1024], expected)

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

    @pytest.mark.device_agnostic
    def test_set_image_rejects_an_integer_image(self):
        prompter, _ = _prompter_with_stub(torch.device("cpu"), torch.float32)
        # The image is float in [0, 1]; a uint8 image in [0, 255] would be normalised as if it were.
        generator = torch.Generator().manual_seed(0)
        image = torch.randint(0, 256, (3, 77, 128), generator=generator, dtype=torch.uint8)

        with pytest.raises(BaseError, match="floating-point"):
            prompter.set_image(image)
        assert not prompter.is_image_set

    @pytest.mark.device_agnostic
    def test_transforms_must_be_one_resize(self):
        prompter, _ = _prompter_with_stub(torch.device("cpu"), torch.float32)
        resize = K.Resize(1024, side="long", align_corners=False, antialias=True)
        # A first stage that is not a Resize has no output size; extra stages would move the image away from the
        # prompts, which follow the Resize scales only.
        for stages in ((K.RandomHorizontalFlip(),), (resize, K.RandomHorizontalFlip(p=1.0))):
            prompter.transforms = AugmentationSequential(*stages, same_on_batch=True)
            with pytest.raises(BaseError, match="exactly one stage"):
                prompter.set_image(torch.rand(3, 77, 128))

        # A Resize past the encoder's 1024 input would be cropped by the padding while the prompts are not.
        prompter.transforms = AugmentationSequential(
            K.Resize(2048, side="long", align_corners=False, antialias=True), same_on_batch=True
        )
        with pytest.raises(BaseError, match="larger than the image encoder's input size 1024"):
            prompter.set_image(torch.rand(3, 77, 128))

        prompter.transforms = AugmentationSequential(resize, same_on_batch=True)
        prompter.set_image(torch.rand(3, 77, 128))
        assert prompter._input_image_size == (616, 1024)

    def test_half_prompts_map_in_float32(self, device, dtype):
        prompter, _ = _prompter_with_stub(device, dtype)
        prompter.set_image(torch.rand(3, 1380, 2076, device=device, dtype=dtype))
        # The long side 2076 -> 1024; the short side is int(1380 * 1024 / 2076) = 680.
        assert prompter._input_image_size == (680, 1024)
        scale = torch.tensor([1024 / 2076, 680 / 1380], dtype=torch.float64)
        # Keypoints that bfloat16 and float16 hold exactly, and a box with the same corners.
        xy = torch.tensor([[[1504.0, 1000.0]], [[2000.0, 1296.0]], [[600.0, 400.0]]], dtype=torch.float64)
        box = torch.tensor([[600.0, 400.0, 1504.0, 1000.0]], dtype=torch.float64)

        labels = torch.ones(3, 1, device=device, dtype=dtype)
        points = prompter.preprocess_prompts(xy.to(device=device, dtype=dtype), labels).keypoints
        boxes = prompter.preprocess_prompts(boxes=box.to(device=device, dtype=dtype)).boxes

        assert points.dtype == boxes.dtype == _prompt_dtype(dtype)
        # Within float32 rounding (an ulp is 6e-5 at 1000) of the closed forms: (x + 0.5) * s - 0.5 for a keypoint,
        # x * s for a box edge. Mapped in bfloat16 the keypoints were 2.26 px off, in float16 0.26 px; rounding even
        # the exact map back to bfloat16 would cost 1.74 px.
        expected_points = (xy + 0.5) * scale - 0.5
        expected_box = (box.reshape(1, 2, 2) * scale).reshape(1, 4)
        self.assert_close(points.cpu().double(), expected_points, rtol=0.0, atol=2e-4)
        self.assert_close(boxes.cpu().double(), expected_box, rtol=0.0, atol=2e-4)
        # The whole image covers the whole resized image after a downscale too. The box is given in float32, which
        # holds 2076 and 1380 (bfloat16 does not); prompts of any float dtype are accepted.
        whole = torch.tensor([[0.0, 0.0, 2076.0, 1380.0]], device=device)
        self.assert_close(
            prompter.preprocess_prompts(boxes=whole).boxes.cpu().double(),
            torch.tensor([[0.0, 0.0, 1024.0, 680.0]], dtype=torch.float64),
            rtol=0.0,
            atol=2e-4,
        )

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

        # The values of a tensor box are checked as ``Boxes.from_tensor`` checks them: x2 > x1 and y2 > y1.
        with pytest.raises(ValueError, match="negative widths"):
            prompter.predict(boxes=torch.tensor([[60.0, 20.0, 10.0, 50.0]], device=device, dtype=dtype))

    # A sub-pixel box, and one whose -1 vertex offset bfloat16 cannot represent at 256 (spacing 2): the ``Boxes``
    # vertex round trip gives [9.5, 19.5, 11, 21] for the first and, in bfloat16, [256, 256, 256, 256] for the second.
    @pytest.mark.parametrize("box", [[10.0, 20.0, 10.5, 20.5], [256.0, 256.0, 258.0, 258.0]], ids=["subpixel", "256"])
    def test_tensor_box_corners_are_taken_as_given(self, device, dtype, box):
        prompter = _prompter_with_stub_encoder(device, dtype)
        boxes = torch.tensor([box], device=device, dtype=dtype)

        mapped = prompter.preprocess_prompts(boxes=boxes).boxes

        # The edge map at s = 8 (77 x 128 -> 616 x 1024), x * 8 on each given corner, in the float32 prompt dtype.
        expected = torch.tensor([box], dtype=torch.float64) * 8
        self.assert_close(mapped, expected.to(device=device, dtype=_prompt_dtype(dtype)))

    def test_box_prompts_stay_separate(self, device, dtype, cudnn_tf32_follows_option):
        prompter = _prompter_with_stub_encoder(device, dtype)
        boxes = torch.tensor(
            [[10.0, 20.0, 60.0, 50.0], [5.0, 5.0, 120.0, 70.0], [30.0, 10.0, 90.0, 40.0]], device=device, dtype=dtype
        )
        # Each box's own edges scaled by s = 8 (77 x 128 -> 616 x 1024).
        expected = boxes.to(_prompt_dtype(dtype)) * 8

        # The four documented forms of K = 3 boxes: (K, 4), (1, K, 4), and Boxes of (K, 4, 2) and (1, K, 4, 2).
        forms = (boxes, boxes[None], Boxes.from_tensor(boxes, mode="xyxy"), Boxes.from_tensor(boxes[None], mode="xyxy"))
        for form in forms:
            self.assert_close(prompter.preprocess_prompts(boxes=form).boxes, expected)

        # Each prompt's masks are the ones of that box alone. `cudnn_tf32_follows_option` keeps the mask decoder's two
        # ConvTranspose2d out of cuDNN's default TF32 on CUDA, which moves a float32 result by 1.8e-4 between batch
        # sizes (2.7e-7 in real float32), past the float32 tolerance.
        together = prompter.predict(boxes=boxes, output_original_size=False)
        for k in range(3):
            alone = prompter.predict(boxes=boxes[k : k + 1], output_original_size=False)
            self.assert_close(together.logits[k], alone.logits[0])
            self.assert_close(together.scores[k], alone.scores[0])

    def test_gradcheck(self, device):
        pytest.skip("Gradcheck is not currently applicable for VisualPrompter inference.")

    def test_cardinality(self, device, dtype):
        pytest.skip("Cardinality is covered by the test_batching_pipeline.")

    def test_dynamo(self, device, dtype, torch_optimizer):
        pytest.skip("Dynamo compilation currently broken for VisualPrompter. See #FIXME in source.")
