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


import pickle

import pytest
import torch

from kornia.models.efficient_vit import EfficientViT, EfficientViTConfig
from kornia.models.efficient_vit import backbone as vit
from kornia.models.efficient_vit.nn.norm import LayerNorm2d, build_norm
from kornia.models.efficient_vit.nn.ops import ConvLayer

from testing.base import BaseTester


class TestEfficientViTLayerNorm2d(BaseTester):
    @pytest.mark.applies_to(dtypes=("float16", "bfloat16"), reason="half-precision channel cancellation")
    def test_convolution_norm_keeps_close_channels(self, device, dtype):
        layer = ConvLayer(2, 2, kernel_size=1, use_bias=False, norm="ln2d", act_func=None).to(device, dtype)
        with torch.no_grad():
            layer.conv.weight.copy_(torch.eye(2, device=device, dtype=dtype).reshape(2, 2, 1, 1))
        delta = torch.finfo(dtype).eps / 2
        image = torch.tensor([0.5, 0.5 + delta], device=device, dtype=dtype).reshape(1, 2, 1, 1)
        # NumPy float64: d = 2**-11 for float16, 2**-8 for bfloat16;
        # expected = array([-d/2, d/2]) / sqrt(d*d/4 + 1e-5).
        magnitude = 0.0769749805426409 if dtype == torch.float16 else 0.525483824574412
        expected = torch.tensor([-magnitude, magnitude], device=device, dtype=dtype).reshape_as(image)
        actual = layer(image)
        assert actual.dtype == dtype
        self.assert_close(actual, expected)

    def test_bias_disabled(self, device, dtype):
        norm = build_norm("ln2d", num_features=2, bias=False).to(device, dtype)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([2.0, 0.5], device=device, dtype=dtype))
        image = torch.tensor([1.0, 3.0], device=device, dtype=dtype).reshape(1, 2, 1, 1)
        # NumPy float64: array([-1., 1.]) / sqrt(1. + 1e-5) * array([2., .5]).
        expected = torch.tensor([-1.999990000075, 0.499997500019], device=device, dtype=dtype).reshape_as(image)
        self.assert_close(norm(image), expected)

    @pytest.mark.applies_to(dtypes=("float16", "bfloat16"), reason="mixed affine/input dtype contract")
    def test_float32_affine_retains_output_dtype(self, device, dtype):
        norm = LayerNorm2d(2).to(device)
        image = torch.tensor([-1.0, 1.0], device=device, dtype=dtype).reshape(1, 2, 1, 1)
        assert norm(image).dtype == torch.float32
        assert LayerNorm2d(2, elementwise_affine=False).to(device)(image).dtype == dtype

    @pytest.mark.parametrize("affine", [True, False])
    def test_gradcheck(self, device, affine):
        norm = LayerNorm2d(4, elementwise_affine=affine, bias=False).to(device, torch.float64)
        image = torch.rand(1, 4, 2, 2, device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(norm, (image,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        norm = LayerNorm2d(4, bias=False).to(device, dtype)
        image = torch.rand(2, 4, 3, 2, device=device, dtype=dtype)
        self.assert_close(torch_optimizer(norm)(image), norm(image))

    def test_jit(self, device, dtype):
        norm = LayerNorm2d(4, bias=False).to(device, dtype)
        image = torch.rand(2, 4, 3, 2, device=device, dtype=dtype)
        self.assert_close(torch.jit.script(norm)(image), norm(image))


class _NotAWeight:
    """An arbitrary class, which ``torch.load(..., weights_only=True)`` refuses to unpickle."""


class TestEfficientViT:
    @staticmethod
    def _fake_checkpoint() -> dict[str, torch.Tensor]:
        """Mimic the hosted checkpoints: backbone weights under a ``backbone.`` prefix plus a ``head.`` classifier."""
        model = vit.efficientvit_backbone_b1()
        state_dict = {
            f"backbone.{key}": val.clone()
            for key, val in model.state_dict().items()
            if "num_batches_tracked" not in key
        }
        state_dict["head.classifier.weight"] = torch.randn(1000, 128)
        state_dict["head.classifier.bias"] = torch.randn(1000)
        return state_dict

    def test_load_pretrained_loads_the_checkpoint_weights(self, monkeypatch):
        # GH#5276: strict=False over the raw checkpoint silently loaded no weight at all
        state_dict = self._fake_checkpoint()
        monkeypatch.setattr(
            "kornia.models.efficient_vit.model.load_state_dict_from_url",
            lambda *args, **kwargs: {"state_dict": state_dict},
        )

        model = EfficientViT.from_config(EfficientViTConfig())

        for key, val in state_dict.items():
            if key.startswith("backbone."):
                assert torch.equal(model.backbone.state_dict()[key[len("backbone.") :]], val)

    def test_load_local_checkpoint_expands_user(self, tmp_path, monkeypatch):
        # a ``~`` path must not reach the URL loader, which resolves a name already in the hub cache to that file
        state_dict = self._fake_checkpoint()
        torch.save(state_dict, tmp_path / "b1-r224.pt")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.setattr(
            "kornia.models.efficient_vit.model.load_state_dict_from_url",
            lambda *args, **kwargs: pytest.fail("local checkpoints must not be downloaded"),
        )

        model = EfficientViT.from_config(EfficientViTConfig(checkpoint="~/b1-r224.pt"))

        for key, val in state_dict.items():
            if key.startswith("backbone."):
                assert torch.equal(model.backbone.state_dict()[key[len("backbone.") :]], val)

    def test_load_local_checkpoint_is_weights_only(self, tmp_path, monkeypatch):
        # a local file is unpickled with weights_only=True, as the URL path and ModelBase.load_checkpoint do
        checkpoint = tmp_path / "b1-local.pt"
        torch.save({"payload": _NotAWeight()}, checkpoint)
        monkeypatch.setattr(
            "kornia.models.efficient_vit.model.load_state_dict_from_url",
            lambda *args, **kwargs: pytest.fail("local checkpoints must not be downloaded"),
        )

        with pytest.raises(pickle.UnpicklingError):
            EfficientViT.from_config(EfficientViTConfig(checkpoint=str(checkpoint)))

    def test_load_failure_preserves_cause(self, monkeypatch):
        cause = RuntimeError("download failed")

        def fail_download(*args, **kwargs):
            raise cause

        monkeypatch.setattr("kornia.models.efficient_vit.model.load_state_dict_from_url", fail_download)

        with pytest.raises(RuntimeError, match="Unable to load the model") as exc_info:
            EfficientViT.from_config(EfficientViTConfig())

        assert exc_info.value.__cause__ is cause

    @pytest.mark.parametrize("drift", ["missing", "unexpected"])
    def test_load_pretrained_is_strict(self, monkeypatch, drift):
        # a checkpoint that lacks a backbone entry, or holds one the backbone has no slot for, raises
        state_dict = self._fake_checkpoint()
        if drift == "missing":
            del state_dict["backbone.input_stem.op_list.0.conv.weight"]
        else:
            state_dict["backbone.input_stem.extra.weight"] = torch.zeros(1)
        monkeypatch.setattr(
            "kornia.models.efficient_vit.model.load_state_dict_from_url",
            lambda *args, **kwargs: {"state_dict": state_dict},
        )

        with pytest.raises(RuntimeError, match=f"{drift.capitalize()} key"):
            EfficientViT.from_config(EfficientViTConfig())

    def test_load_pretrained_accepts_a_backbone_state_dict(self, monkeypatch):
        # a state dict saved from the backbone itself has no "backbone." prefix
        state_dict = {key: val + 1 for key, val in vit.efficientvit_backbone_b1().state_dict().items()}
        monkeypatch.setattr(
            "kornia.models.efficient_vit.model.load_state_dict_from_url", lambda *args, **kwargs: state_dict
        )

        model = EfficientViT.from_config(EfficientViTConfig())

        loaded = model.backbone.state_dict()
        assert loaded.keys() == state_dict.keys()
        for key, val in state_dict.items():
            assert torch.equal(loaded[key], val)

    def test_load_pretrained_returns_eval_mode(self, monkeypatch):
        state_dict = self._fake_checkpoint()
        monkeypatch.setattr(
            "kornia.models.efficient_vit.model.load_state_dict_from_url",
            lambda *args, **kwargs: {"state_dict": state_dict},
        )

        model = EfficientViT.from_config(EfficientViTConfig.from_pretrained("b1", 224))

        # ``training`` of the model and of every submodule: ``{False}`` is evaluation mode throughout.
        assert {m.training for m in model.modules()} == {False}

    @pytest.mark.slow
    def test_load_pretrained_is_batch_independent(self):
        model = EfficientViT.from_config(EfficientViTConfig.from_pretrained("b1", 224))
        running_means = [m.running_mean.clone() for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        generator = torch.Generator().manual_seed(0)
        x1 = torch.rand(1, 3, 224, 256, generator=generator)
        x2 = torch.rand(1, 3, 224, 256, generator=generator)

        with torch.no_grad():
            alone = model(x1)["stage_final"][0]
            batched = model(torch.cat([x1, x2]))["stage_final"][0]

        # In evaluation mode sample 0 does not see sample 1; only the kernels' summation order may change with the
        # batch size, so the float32 tolerance is widened to 1e-4.
        torch.testing.assert_close(alone, batched, rtol=1e-4, atol=1e-4)
        after = [m.running_mean for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        assert all(torch.equal(a, b) for a, b in zip(after, running_means))

    def _test_smoke(self, device, dtype, img_size: int, expected_resolution: int, model_name: str):
        model = getattr(vit, f"efficientvit_backbone_{model_name}")()
        model = model.to(device=device, dtype=dtype)

        image = torch.randn(1, 3, img_size, img_size, device=device, dtype=dtype)

        out = model(image)

        assert "input" in out
        assert out["input"].shape == image.shape

        assert "stage_final" in out
        assert out["stage_final"].shape[-2:] == torch.Size([expected_resolution, expected_resolution])

    # One size per variant, as in test_smoke: no backbone branches on the input size.
    @pytest.mark.parametrize(
        "model_name,img_size,expected_resolution",
        [("b3", 256, 8), ("l0", 224, 7), ("l1", 256, 8), ("l2", 288, 9), ("l3", 320, 10)],
    )
    @pytest.mark.slow
    def test_smoke_slow(self, device, dtype, img_size: int, expected_resolution: int, model_name: str):
        self._test_smoke(device, dtype, img_size, expected_resolution, model_name)

    # The final resolution is the input's over the stride of 32 whatever the width and depth, so each variant runs at
    # one of the sizes, the odd 3 x 3 grid included, rather than every variant at every size.
    @pytest.mark.parametrize("model_name,img_size,expected_resolution", [("b0", 128, 4), ("b1", 96, 3), ("b2", 64, 2)])
    def test_smoke(self, device, dtype, img_size: int, expected_resolution: int, model_name: str):
        self._test_smoke(device, dtype, img_size, expected_resolution, model_name)

    @pytest.mark.applies_to(
        when=lambda p: not (p["device_name"] == "mps" and p["dtype_name"] == "float64"),
        reason="MPS does not support float64",
    )
    @pytest.mark.parametrize("model_name", ["b0", "l0"])
    def test_empty_batch_6266(self, device, dtype, model_name: str):
        """An empty batch goes through the linear attention and keeps every stage's channels and size."""
        model = getattr(vit, f"efficientvit_backbone_{model_name}")().to(device, dtype).eval()
        feats = model(torch.rand(0, 3, 64, 64, device=device, dtype=dtype))
        ref = model(torch.rand(1, 3, 64, 64, device=device, dtype=dtype))
        assert {k: v.shape[1:] for k, v in feats.items()} == {k: v.shape[1:] for k, v in ref.items()}
        assert all(v.shape[0] == 0 for v in feats.values())

    @pytest.mark.slow
    def test_load_pretrained(self, device, dtype):
        model = EfficientViT.from_config(EfficientViTConfig())
        model = model.to(device=device, dtype=dtype)

        image = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)
        feats = model(image)
        assert feats["stage_final"].shape == torch.Size([1, 256, 7, 7])

    def test_config(self):
        # One non-default pair: tests/core/test_weights_prefetch.py enumerates every type/resolution URL.
        config = EfficientViTConfig.from_pretrained("b2", 256)
        assert "b2" in config.checkpoint
        assert "256" in config.checkpoint

    @pytest.mark.applies_to(
        when=lambda p: not (p["device_name"] == "mps" and p["dtype_name"] == "float64"),
        reason="MPS does not support float64",
    )
    def test_forward_returns_a_dict(self, device, dtype):
        model = EfficientViT(backbone=vit.efficientvit_backbone_b0()).to(device, dtype).eval()
        image = torch.rand(1, 3, 64, 96, device=device, dtype=dtype)

        with torch.no_grad():
            out = model(image)

        assert isinstance(out, dict)
        # stride 32: (64 // 32, 96 // 32)
        assert out["stage_final"].shape[-2:] == (2, 3)
