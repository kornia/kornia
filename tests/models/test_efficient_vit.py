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

from kornia.core._compat import torch_version_lt
from kornia.models.efficient_vit import EfficientViT, EfficientViTConfig
from kornia.models.efficient_vit import backbone as vit


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

    def test_load_local_checkpoint(self, tmp_path, monkeypatch):
        state_dict = self._fake_checkpoint()
        checkpoint = tmp_path / "b1-local.pt"
        torch.save(state_dict, checkpoint)
        monkeypatch.setattr(
            "kornia.models.efficient_vit.model.load_state_dict_from_url",
            lambda *args, **kwargs: pytest.fail("local checkpoints must not be downloaded"),
        )

        model = EfficientViT.from_config(EfficientViTConfig(checkpoint=str(checkpoint)))

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

        key = "input_stem.op_list.0.conv.weight"
        assert torch.equal(model.backbone.state_dict()[key], state_dict[f"backbone.{key}"])

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

    def _test_smoke(self, device, dtype, img_size: int, expected_resolution: int, model_name: str):
        model = getattr(vit, f"efficientvit_backbone_{model_name}")()
        model = model.to(device=device, dtype=dtype)

        image = torch.randn(1, 3, img_size, img_size, device=device, dtype=dtype)

        out = model(image)

        assert "input" in out
        assert out["input"].shape == image.shape

        assert "stage_final" in out
        assert out["stage_final"].shape[-2:] == torch.Size([expected_resolution, expected_resolution])

    @pytest.mark.parametrize("model_name", ["b3"])
    @pytest.mark.parametrize("img_size,expected_resolution", [(224, 7), (256, 8), (288, 9)])
    @pytest.mark.slow
    def test_smoke_slow(self, device, dtype, img_size: int, expected_resolution: int, model_name: str):
        self._test_smoke(device, dtype, img_size, expected_resolution, model_name)

    @pytest.mark.parametrize("model_name", ["b0", "b1", "b2"])
    @pytest.mark.parametrize("img_size,expected_resolution", [(64, 2), (96, 3), (128, 4)])
    def test_smoke(self, device, dtype, img_size: int, expected_resolution: int, model_name: str):
        self._test_smoke(device, dtype, img_size, expected_resolution, model_name)

    @pytest.mark.slow
    @pytest.mark.skipif(torch_version_lt(2, 0, 0), reason="requires torch 2.0.0 or higher")
    @pytest.mark.parametrize("model_name", ["l0", "l1", "l2", "l3"])
    @pytest.mark.parametrize("img_size,expected_resolution", [(224, 7), (256, 8), (288, 9), (320, 10), (384, 12)])
    def test_smoke_large(self, device, dtype, img_size: int, expected_resolution: int, model_name: str):
        self._test_smoke(device, dtype, img_size, expected_resolution, model_name)

    @pytest.mark.slow
    def test_load_pretrained(self, device, dtype):
        model = EfficientViT.from_config(EfficientViTConfig())
        model = model.to(device=device, dtype=dtype)

        image = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)
        feats = model(image)
        assert feats["stage_final"].shape == torch.Size([1, 256, 7, 7])

    @pytest.mark.parametrize("model_type", ["b1", "b2", "b3"])
    @pytest.mark.parametrize("resolution", [224, 256, 288])
    def test_config(self, model_type, resolution):
        config = EfficientViTConfig.from_pretrained(model_type, resolution)
        assert model_type in config.checkpoint
        assert str(resolution) in config.checkpoint
