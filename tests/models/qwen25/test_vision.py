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

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
import torch

from kornia.models.qwen25 import Qwen25VisionConfig, Qwen25VisionTransformer

from testing.base import BaseTester


def _tiny_config(**kwargs) -> Qwen25VisionConfig:
    # A 56-pixel window is 2 x 2 merged cells, so a 112 x 84 image (4 x 3 cells) has four windows of
    # unequal size, including the narrow column the reference pads.
    values = {
        "depth": 2,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_heads": 2,
        "out_hidden_size": 16,
        "window_size": 56,
        "fullatt_block_indexes": (1,),
    }
    values.update(kwargs)
    return Qwen25VisionConfig(**values)


def _seeded(config: Qwen25VisionConfig, image_size: tuple[int, int] = (112, 84)):
    """Return a float64 CPU model with every parameter drawn from N(0, 0.2^2), and one input image."""
    model = Qwen25VisionTransformer(config).double().eval()
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=generator, dtype=torch.float64) * 0.2)
    images = torch.randn(1, 3, *image_size, generator=generator, dtype=torch.float64)
    return model, images


def _save_safetensors(tensors: dict[str, torch.Tensor], path: Path) -> None:
    """Write float32 tensors as a ``.safetensors`` file, straight from the format specification."""
    header, blob = {}, b""
    for name, tensor in tensors.items():
        payload = tensor.to(torch.float32).contiguous().numpy().tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [len(blob), len(blob) + len(payload)],
        }
        blob += payload
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + blob)


class TestQwen25VisionTransformer(BaseTester):
    def test_repeated_frame_projection_workspace(self, device, dtype):
        if device.type != "cpu" or dtype not in (torch.float32, torch.float64):
            pytest.skip("The repeated-frame optimization targets CPU float32/float64.")
        config = _tiny_config(
            depth=0,
            hidden_size=4,
            num_heads=1,
            intermediate_size=4,
            out_hidden_size=4,
            in_channels=1,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=1,
            window_size=2,
        )
        model = Qwen25VisionTransformer(config).to(device, dtype).eval()
        images = torch.rand(1, 1, 16, 16, device=device, dtype=dtype)
        with torch.no_grad():
            model(images)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
            ) as profile:
                model(images)
        copy_bytes = sum(e.cpu_memory_usage for e in profile.events() if e.name == "aten::contiguous")
        assert copy_bytes < images.numel() * images.element_size()

    def test_repeated_frame_projection_keeps_autocast_rounding(self):
        model = Qwen25VisionTransformer(_tiny_config(depth=0, temporal_patch_size=2))
        projection = model.patch_embed.proj
        with torch.no_grad():
            projection.weight[:, :, 0].fill_(1.0)
            projection.weight[:, :, 1].fill_(-0.9999)
        images = torch.ones(1, 3, 56, 56)
        captured = []
        hook = projection.register_forward_hook(lambda module, args, output: captured.append(output))
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                model(images)
        finally:
            hook.remove()
        # Autocast rounds both temporal taps to +/-1 before their products, so they cancel exactly.
        self.assert_close(captured[0], torch.zeros_like(captured[0]), atol=0.0, rtol=0.0)

    def test_repeated_frame_projection_values_gradients_and_hook(self, device, dtype):
        config = _tiny_config(
            depth=0,
            hidden_size=4,
            num_heads=1,
            intermediate_size=4,
            out_hidden_size=4,
            in_channels=1,
            patch_size=2,
            temporal_patch_size=3,
            spatial_merge_size=1,
            window_size=2,
        )
        model = Qwen25VisionTransformer(config).to(device, dtype)
        projection = model.patch_embed.proj
        with torch.no_grad():
            projection.weight.copy_(torch.arange(48, device=device, dtype=dtype).reshape_as(projection.weight) / 16)
        images = (torch.arange(16, device=device, dtype=dtype).reshape(1, 1, 4, 4) / 16).requires_grad_()
        projected = []
        hook = projection.register_forward_hook(lambda module, args, output: projected.append(output))
        try:
            model(images)
        finally:
            hook.remove()
        assert len(projected) == 1

        # Independent sums over each 2x2 patch: the three temporal taps at channel c and offset j
        # sum to (36*c + 12 + 3*j)/16; pixels are integer multiples of 1/16.
        expected = torch.tensor(
            [
                [[0.75, 1.265625], [2.8125, 3.328125]],
                [[2.15625, 3.796875], [8.71875, 10.359375]],
                [[3.5625, 6.328125], [14.625, 17.390625]],
                [[4.96875, 8.859375], [20.53125, 24.421875]],
            ],
            device=device,
            dtype=dtype,
        ).reshape(1, 4, 1, 2, 2)
        self.assert_close(projected[0], expected)
        image_grad, weight_grad = torch.autograd.grad(projected[0].sum(), (images, projection.weight))
        expected_image_grad = (
            torch.tensor([[16.5, 17.25], [18.0, 18.75]], device=device, dtype=dtype).repeat(2, 2).reshape_as(images)
        )
        expected_weight_grad = torch.tensor([[1.25, 1.5], [2.25, 2.5]], device=device, dtype=dtype).expand_as(
            projection.weight
        )
        self.assert_close(image_grad, expected_image_grad)
        self.assert_close(weight_grad, expected_weight_grad)

    @pytest.mark.parametrize("batch_size", [0, 2])
    def test_smoke(self, batch_size, device, dtype):
        model = Qwen25VisionTransformer(_tiny_config()).to(device, dtype).eval()
        images = torch.rand(batch_size, 3, 112, 84, device=device, dtype=dtype)

        with torch.no_grad():
            out = model(images)

        assert out.shape == (batch_size, 12, 16)
        assert out.dtype == dtype
        assert out.device.type == device.type

    @pytest.mark.parametrize(("height", "width", "tokens"), [(28, 28, 1), (56, 84, 6)])
    def test_cardinality(self, height, width, tokens, device, dtype):
        model = Qwen25VisionTransformer(_tiny_config()).to(device, dtype).eval()

        with torch.no_grad():
            out = model(torch.rand(1, 3, height, width, device=device, dtype=dtype))

        assert out.shape == (1, tokens, 16)

    @pytest.mark.parametrize("shape", [(1, 3, 30, 28), (1, 3, 28, 42), (1, 1, 28, 28), (3, 28, 28)])
    def test_exception_input_shape(self, shape, device, dtype):
        model = Qwen25VisionTransformer(_tiny_config()).to(device, dtype)

        with pytest.raises(ValueError):
            model(torch.rand(*shape, device=device, dtype=dtype))

    @pytest.mark.parametrize(
        "kwargs",
        [{"hidden_size": 30, "num_heads": 4}, {"hidden_size": 36, "num_heads": 2}, {"window_size": 42}],
    )
    def test_exception_config(self, kwargs):
        with pytest.raises(ValueError):
            _tiny_config(**kwargs)

    def test_reference_values(self, device, dtype):
        """Match Hugging Face ``transformers`` v4.51.0 on a tiny random configuration.

        One token from each of the four windows is pinned, so a token assigned to the wrong window shows up. The
        expected values were produced by copying the seeded weights below into the reference and feeding it
        the patches of its own image processor (``max abs diff`` 1.6e-7 over the whole float64 output, the
        reference running its normalizations and rotary embedding in float32)::

            from transformers import Qwen2VLImageProcessor
            from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

            model, images = _seeded(_tiny_config())
            cfg = Qwen2_5_VLVisionConfig(depth=2, hidden_size=32, intermediate_size=48, num_heads=2,
                                         out_hidden_size=16, window_size=56, fullatt_block_indexes=[1],
                                         hidden_act="silu")
            ref = Qwen2_5_VisionTransformerPretrainedModel._from_config(cfg, attn_implementation="eager")
            ref = ref.double().eval()
            ref.load_state_dict(model.state_dict(), strict=True)
            proc = Qwen2VLImageProcessor(do_resize=False, do_rescale=False, do_normalize=False)
            enc = proc(images=[images[0].numpy()], return_tensors="pt", input_data_format="channels_first")
            expected = ref(enc["pixel_values"].double(), enc["image_grid_thw"])[[0, 5, 6, 11], :6]
        """
        model, images = _seeded(_tiny_config())
        model = model.to(device, dtype)
        images = images.to(device, dtype)
        expected = torch.tensor(
            [
                [0.09549037857805515, -0.4318030974107194, -0.2147635769408925, 0.04750012480055123,
                 -0.18701005751370206, 0.8672491593678862],
                [0.5476752592262668, -0.34546495229155505, -1.1540137094169247, 0.6783189091656767,
                 0.42628693902281506, 1.4598784556874398],
                [-1.1210075910373243, 0.4016404669306015, 0.6027981222959102, -0.15521229785229282,
                 1.025392446425122, 0.9670840563199808],
                [-0.4691275876673318, -0.4147935590654865, -0.00886787003644618, -0.6345986879315241,
                 0.8549848277851364, 0.8655891917360579],
            ],
            device=device,
            dtype=dtype,
        )  # fmt: skip

        with torch.no_grad():
            out = model(images)

        self.assert_close(out[0, [0, 5, 6, 11], :6], expected, low_tolerance=dtype in (torch.float16, torch.bfloat16))

    def test_patch_swap_is_not_a_token_permutation(self, device, dtype):
        """Swapping two merged cells must change the tokens, not only reorder them, since positions are encoded.

        Every block attends globally here: windowed attention alone already breaks the permutation, so only the
        rotary embedding can make this test pass.
        """
        model = Qwen25VisionTransformer(_tiny_config(fullatt_block_indexes=(0, 1))).to(device, dtype).eval()
        images = torch.rand(1, 3, 112, 84, device=device, dtype=dtype)
        swapped = images.clone()
        swapped[..., 0:28, 0:28] = images[..., 84:112, 56:84]
        swapped[..., 84:112, 56:84] = images[..., 0:28, 0:28]

        with torch.no_grad():
            out = model(images)[0].float()
            out_swapped = model(swapped)[0].float()

        # Under a pure permutation every swapped token would have an exact twin among the original tokens: with the
        # rotary embedding removed the largest twin distance is below 1e-6, with it above 4e-3 for every dtype.
        nearest = torch.cdist(out_swapped, out).min(dim=1).values
        assert nearest.max() > 1e-4

    def test_windowed_block_is_local(self, device, dtype):
        """With no full-attention block, a change inside one window leaves the tokens of the other windows alone."""
        model = Qwen25VisionTransformer(_tiny_config(fullatt_block_indexes=())).to(device, dtype).eval()
        images = torch.rand(1, 3, 112, 84, device=device, dtype=dtype)
        changed = images.clone()
        changed[..., 0:56, 0:56] += 1.0

        with torch.no_grad():
            out = model(images)[0]
            out_changed = model(changed)[0]

        # The 4 x 3 merged grid in raster order; the first window covers cells (0, 0), (0, 1), (1, 0), (1, 1).
        inside = [0, 1, 3, 4]
        outside = [2, 5, 6, 7, 8, 9, 10, 11]
        self.assert_close(out_changed[outside], out[outside])
        assert not torch.allclose(out_changed[inside], out[inside])

    def test_batch_consistency(self, device, dtype):
        model = Qwen25VisionTransformer(_tiny_config()).to(device, dtype).eval()
        images = torch.rand(2, 3, 56, 84, device=device, dtype=dtype)

        with torch.no_grad():
            batched = model(images)
            single = model(images[1:])

        self.assert_close(batched[1:], single)

    def test_gradcheck(self, device):
        model = Qwen25VisionTransformer(_tiny_config(depth=1, fullatt_block_indexes=())).to(device, torch.float64)
        images = torch.rand(1, 3, 56, 56, device=device, dtype=torch.float64)

        self.gradcheck(model, (images,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        model = Qwen25VisionTransformer(_tiny_config()).to(device, dtype).eval()
        images = torch.rand(1, 3, 112, 84, device=device, dtype=dtype)
        model_optimized = torch_optimizer(model)

        with torch.no_grad():
            expected = model(images)
            actual = model_optimized(images)

        self.assert_close(actual, expected)


class TestQwen25FromPretrained:
    @pytest.fixture
    def checkpoint(self, tmp_path):
        """Write a two-shard checkpoint whose first shard mixes vision and language tensors, like the release."""
        model, _ = _seeded(_tiny_config())
        vision = {f"visual.{name}": tensor for name, tensor in model.state_dict().items()}
        _save_safetensors({**vision, "model.embed_tokens.weight": torch.zeros(4, 8)}, tmp_path / "a.st")
        _save_safetensors({"lm_head.weight": torch.zeros(4, 8)}, tmp_path / "b.st")
        weight_map = dict.fromkeys(vision, "a.st")
        weight_map.update({"model.embed_tokens.weight": "a.st", "lm_head.weight": "b.st"})
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        return tmp_path, model

    def test_directory(self, checkpoint):
        path, model = checkpoint

        loaded = Qwen25VisionTransformer.from_pretrained(path, _tiny_config())

        for name, tensor in model.state_dict().items():
            assert torch.equal(loaded.state_dict()[name], tensor.float()), name
        assert not loaded.training

    def test_single_file(self, checkpoint):
        path, model = checkpoint

        loaded = Qwen25VisionTransformer.from_pretrained(path / "a.st", _tiny_config())

        assert torch.equal(loaded.merger.mlp[2].weight, model.merger.mlp[2].weight.float())

    def test_mismatched_config_raises(self, checkpoint):
        path, _ = checkpoint

        with pytest.raises(RuntimeError):
            Qwen25VisionTransformer.from_pretrained(path, _tiny_config(depth=3))
