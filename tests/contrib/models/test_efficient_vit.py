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

from pathlib import Path

import pytest
import torch

from kornia.contrib.models.efficient_vit import EfficientViT, EfficientViTConfig
from kornia.contrib.models.efficient_vit import backbone as vit
from kornia.utils._compat import torch_version_lt


class TestEfficientViT:
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
    @pytest.mark.parametrize("img_size,expected_resolution", [(224, 7), (256, 8), (288, 9)])
    def test_smoke(self, device, dtype, img_size: int, expected_resolution: int, model_name: str):
        self._test_smoke(device, dtype, img_size, expected_resolution, model_name)

    @pytest.mark.slow
    @pytest.mark.skipif(torch_version_lt(2, 0, 0), reason="requires torch 2.0.0 or higher")
    @pytest.mark.parametrize("model_name", ["l0", "l1", "l2", "l3"])
    @pytest.mark.parametrize("img_size,expected_resolution", [(224, 7), (256, 8), (288, 9), (320, 10), (384, 12)])
    def test_smoke_large(self, device, dtype, img_size: int, expected_resolution: int, model_name: str):
        self._test_smoke(device, dtype, img_size, expected_resolution, model_name)

    @pytest.mark.skipif(torch_version_lt(2, 0, 0), reason="requires torch 2.0.0 or higher")
    def test_onnx(self, device, dtype, tmp_path: Path):
        model: vit.EfficientViTBackbone = vit.efficientvit_backbone_b0()
        model = model.to(device=device, dtype=dtype)

        image = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)

        model_path = tmp_path / "efficientvit_backbone_b0.onnx"

        torch.onnx.export(model, image, model_path, opset_version=16)

        assert model_path.is_file()

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
