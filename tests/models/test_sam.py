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
import torch.nn.functional as F
from torch import nn

from kornia.core.exceptions import BaseError
from kornia.models.sam import Sam, SamConfig
from kornia.models.sam.architecture.prompt_encoder import PositionEmbeddingRandom, PromptEncoder

from testing.base import BaseTester


def _pad_rb(x, size):
    """Pads right bottom."""
    pad_h = size - x.shape[-2]
    pad_w = size - x.shape[-1]
    return F.pad(x, (0, pad_w, 0, pad_h))


class TestSam(BaseTester):
    @pytest.mark.slow
    @pytest.mark.parametrize("model_type", ["vit_b", "mobile_sam"])
    def test_smoke(self, device, model_type):
        model = Sam.from_config(SamConfig(model_type)).to(device)
        assert isinstance(model, Sam)

        img_size = model.image_encoder.img_size
        data = torch.randn(1, 3, img_size, img_size, device=device)
        keypoints = torch.randint(0, img_size, (1, 2, 2), device=device, dtype=torch.float)
        labels = torch.randint(0, 1, (1, 2), device=device, dtype=torch.float)

        model(data, [{"points": (keypoints, labels)}], False)

    @pytest.mark.slow
    @pytest.mark.parametrize("batch_size", [1, 3])
    @pytest.mark.parametrize("N", [2, 5])
    @pytest.mark.parametrize("multimask_output", [True, False])
    def test_cardinality(self, device, batch_size, N, multimask_output):
        # SAM: don't supports float64
        dtype = torch.float32
        data = torch.rand(1, 3, 77, 128, device=device, dtype=dtype)
        model = Sam.from_config(SamConfig("vit_b"))
        model = model.to(device=device, dtype=dtype)
        data = _pad_rb(data, model.image_encoder.img_size)
        keypoints = torch.randint(0, min(data.shape[-2:]), (batch_size, N, 2), device=device).to(dtype=dtype)
        labels = torch.randint(0, 1, (batch_size, N), device=device).to(dtype=dtype)

        out = model(data, [{"points": (keypoints, labels)}], multimask_output)

        C = 3 if multimask_output else 1
        assert len(out) == data.size(0)
        assert out[0].logits.shape == (batch_size, C, 256, 256)

    def test_exception(self):
        model = Sam.from_config(SamConfig("mobile_sam"))

        from kornia.core.exceptions import ShapeError

        data = torch.rand(3, 1, 2)
        with pytest.raises(ShapeError) as errinfo:
            model(data, [], False)
        assert "Shape dimension mismatch" in str(errinfo.value) or "Expected shape" in str(errinfo.value)

        data = torch.rand(2, 3, 1, 2)
        with pytest.raises(Exception) as errinfo:
            model(data, [{}], False)
        assert "The number of images (`B`) should match with the length of prompts!" in str(errinfo)

    @pytest.mark.slow
    @pytest.mark.parametrize("model_type", ["vit_b", "vit_l", "vit_h", "mobile_sam"])
    def test_config(self, device, model_type):
        model = Sam.from_config(SamConfig(model_type))
        model = model.to(device=device)

        assert isinstance(model, Sam)
        assert next(model.parameters()).device == device

    @pytest.mark.skip(reason="Unsupported at moment -- the code is not tested for training and had `torch.no_grad`")
    def test_gradcheck(self, device): ...

    @pytest.mark.skip(reason="Unnecessary test")
    def test_module(self): ...

    @pytest.mark.skip(reason="Needs to be reviewed.")
    def test_dynamo(self, device, torch_optimizer):
        dtype = torch.float32
        img = torch.rand(1, 3, 128, 75, device=device, dtype=dtype)

        op = Sam.from_config(SamConfig("vit_b"))
        op = op.to(device=device, dtype=dtype)

        op_optimized = torch_optimizer(op)

        img = _pad_rb(img, op.image_encoder.img_size)

        expected = op(img, [{}], False)
        actual = op_optimized(img, [{}], False)

        self.assert_close(expected[0].logits, actual[0].logits)
        self.assert_close(expected[0].scores, actual[0].scores)

    @pytest.mark.slow
    @pytest.mark.parametrize("model_type", ["vit_b", "mobile_sam"])
    def test_pretrained(self, model_type):
        Sam.from_config(SamConfig(model_type, pretrained=True))

    @pytest.mark.device_agnostic
    def test_from_config_mode(self, monkeypatch):
        loaded = []
        monkeypatch.setattr(Sam, "load_checkpoint", lambda self, checkpoint, device=None: loaded.append(checkpoint))

        # ``training`` of the model and of every submodule: ``{False}`` is evaluation mode throughout.
        weighted = Sam.from_config(SamConfig("mobile_sam", pretrained=True))
        assert {m.training for m in weighted.modules()} == {False}
        from_checkpoint = Sam.from_config(SamConfig("mobile_sam", checkpoint="x.pt"))
        assert {m.training for m in from_checkpoint.modules()} == {False}
        assert len(loaded) == 2
        assert loaded[1] == "x.pt"

        assert {m.training for m in Sam.from_config(SamConfig("mobile_sam")).modules()} == {True}
        assert {m.training for m in Sam.from_name("mobile_sam").modules()} == {True}
        assert len(loaded) == 2

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("global_attn_indexes", [(1,), [1], 1])
    def test_from_config_builds_a_custom_encoder(self, global_attn_indexes):
        config = SamConfig(
            encoder_embed_dim=16, encoder_depth=2, encoder_num_heads=1, encoder_global_attn_indexes=global_attn_indexes
        )
        encoder = Sam.from_config(config).image_encoder

        # Block 1 attends globally (window size 0); block 0 keeps SAM's 14 x 14 windows (``_build_sam``).
        assert [block.window_size for block in encoder.blocks] == [14, 0]
        # 684,272 parameters, mostly the neck's 3 x 3 convolution, 256 * 256 * 9 = 589,824; ViT-B's encoder has
        # 89.7M. The whole model is not bounded: ``_build_sam`` always attaches the full mask decoder.
        assert sum(p.numel() for p in encoder.parameters()) < 1e6

    @pytest.mark.device_agnostic
    def test_pretrained_needs_a_model_type(self):
        config = SamConfig(
            encoder_embed_dim=16, encoder_depth=1, encoder_num_heads=1, encoder_global_attn_indexes=0, pretrained=True
        )
        with pytest.raises(BaseError, match="needs a model_type"):
            Sam.from_config(config)

    def test_positional_encoding_keeps_float32_precision(self, device, dtype):
        if device.type == "mps" and dtype == torch.float64:
            pytest.skip("MPS does not support float64")
        # SAM's prompt encoder: 256-dimensional embeddings, so 128 random frequencies drawn from N(0, 1).
        generator = torch.Generator().manual_seed(0)
        pe = PositionEmbeddingRandom(128).to(device)
        pe.positional_encoding_gaussian_matrix.copy_(torch.randn(2, 128, generator=generator))
        pe = pe.to(dtype)
        # The reference: the same, possibly rounded, frequencies, encoded in float32.
        reference = PositionEmbeddingRandom(128).to(device)
        reference.positional_encoding_gaussian_matrix.copy_(pe.positional_encoding_gaussian_matrix.float())
        # Keypoints in the 1024 x 1024 input frame, exact in bfloat16 (spacing 4 between 512 and 1024): the image
        # centre, a corner pixel and a point near the far corner.
        coords = torch.tensor([[[512.0, 512.0], [4.0, 4.0], [1020.0, 1016.0]]], device=device)

        dense, points = pe((64, 64)), pe.forward_with_coords(coords.to(dtype), (1024, 1024))

        assert dense.dtype == points.dtype == dtype
        # Within the rounding of the float32 encoding. Computed in bfloat16 throughout, this seed's dense encoding is
        # off by 0.12 (0.013 in float16).
        self.assert_close(dense, reference((64, 64)).to(dtype))
        self.assert_close(points, reference.forward_with_coords(coords, (1024, 1024)).to(dtype))

    def test_positional_encoding_of_a_float64_model_is_float64(self, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64")
        generator = torch.Generator().manual_seed(0)
        pe = PositionEmbeddingRandom(128).to(device=device, dtype=torch.float64)
        pe.positional_encoding_gaussian_matrix.copy_(torch.randn(2, 128, generator=generator, dtype=torch.float64))

        def closed_form(dtype: torch.dtype) -> torch.Tensor:
            # SAM's dense encoding of a 64 x 64 grid at pixel centres, (2 c - 1) M, then sin and cos of 2 pi times it.
            centres = (torch.arange(64, device=device, dtype=dtype) + 0.5) / 64
            grid = torch.stack(torch.meshgrid(centres, centres, indexing="xy"), dim=-1)
            phase = 2 * torch.pi * ((2 * grid - 1) @ pe.positional_encoding_gaussian_matrix.to(dtype))
            return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1).permute(2, 0, 1).to(torch.float64)

        dense = pe((64, 64))

        assert dense.dtype == torch.float64
        self.assert_close(dense, closed_form(torch.float64), rtol=0.0, atol=1e-12)
        # The same encoding computed in float32 is off by about 1e-6, so the comparison above can tell them apart.
        assert (closed_form(torch.float32) - dense).abs().max() > 1e-8

    def test_prompt_encoder_shifts_to_pixel_centres_in_float32(self, device, dtype):
        if device.type == "mps" and dtype == torch.float64:
            pytest.skip("MPS does not support float64")
        # SAM's prompt encoder for a 1024 x 1024 input and a 64 x 64 embedding (``_build_sam``).
        encoder = PromptEncoder(256, (64, 64), (1024, 1024), mask_in_chans=16).to(device=device, dtype=dtype)
        seen = []
        forward_with_coords = encoder.pe_layer.forward_with_coords

        def spy(coords: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
            seen.append(coords)
            return forward_with_coords(coords, image_size)

        encoder.pe_layer.forward_with_coords = spy
        # Coordinates that bfloat16 holds (its spacing is 2 between 256 and 512 and 4 between 512 and 1024), so a
        # +0.5 shift computed in bfloat16 is rounded away.
        points = torch.tensor([[[1000.0, 600.0]]], device=device, dtype=dtype)
        boxes = torch.tensor([[260.0, 400.0, 1020.0, 800.0]], device=device, dtype=dtype)

        encoder(points=(points, torch.ones(1, 1, device=device, dtype=dtype)), boxes=boxes, masks=None)

        point_coords, box_coords = seen
        compute_dtype = torch.promote_types(dtype, torch.float32)
        assert point_coords.dtype == box_coords.dtype == compute_dtype
        expected_points = torch.tensor([[[1000.5, 600.5]]], device=device, dtype=compute_dtype)
        expected_boxes = torch.tensor([[[260.5, 400.5], [1020.5, 800.5]]], device=device, dtype=compute_dtype)
        self.assert_close(point_coords, expected_points, rtol=0.0, atol=0.0)
        self.assert_close(box_coords, expected_boxes, rtol=0.0, atol=0.0)

    def test_positional_encoding_ignores_autocast(self, device):
        # Autocast runs a matmul in its half dtype whatever its operands are; the encoding turns it off.
        autocast_dtype = torch.bfloat16 if device.type == "cpu" else torch.float16
        generator = torch.Generator().manual_seed(0)
        pe = PositionEmbeddingRandom(128).to(device)
        pe.positional_encoding_gaussian_matrix.copy_(torch.randn(2, 128, generator=generator))
        coords = torch.tensor([[[512.0, 512.0], [4.0, 4.0], [1020.0, 1016.0]]], device=device)
        dense, points = pe((64, 64)), pe.forward_with_coords(coords, (1024, 1024))

        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            dense_autocast, points_autocast = pe((64, 64)), pe.forward_with_coords(coords, (1024, 1024))

        # Under bfloat16 autocast, a matmul in bfloat16 moves the dense encoding by up to 0.14.
        assert torch.equal(dense_autocast, dense)
        assert torch.equal(points_autocast, points)

    @pytest.mark.slow
    def test_mobile_sam_pretrained_is_batch_independent(self):
        encoder = Sam.from_config(SamConfig("mobile_sam", pretrained=True)).image_encoder
        running_means = [m.running_mean.clone() for m in encoder.modules() if isinstance(m, nn.BatchNorm2d)]
        # The image encoder takes only its square 1024 input; VisualPrompter pads every image to it.
        generator = torch.Generator().manual_seed(0)
        x1 = torch.rand(1, 3, 1024, 1024, generator=generator)
        x2 = torch.rand(1, 3, 1024, 1024, generator=generator)

        with torch.no_grad():
            alone = encoder(x1)[0]
            batched = encoder(torch.cat([x1, x2]))[0]

        # In evaluation mode sample 0 does not see sample 1; only the kernels' summation order may change with the
        # batch size, so the float32 tolerance is widened to 1e-4.
        self.assert_close(alone, batched, rtol=1e-4, atol=1e-4)
        after = [m.running_mean for m in encoder.modules() if isinstance(m, nn.BatchNorm2d)]
        assert all(torch.equal(a, b) for a, b in zip(after, running_means))
