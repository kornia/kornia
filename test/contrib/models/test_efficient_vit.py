from pathlib import Path

import pytest
import torch

from kornia.contrib.models.efficient_vit import EfficientViT, EfficientViTConfig
from kornia.contrib.models.efficient_vit import backbone as vit


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

    @pytest.mark.parametrize("model_name", ["b0", "b1", "b2", "b3"])
    @pytest.mark.parametrize("img_size,expected_resolution", [(224, 7), (256, 8), (288, 9)])
    def test_smoke(self, device, dtype, img_size: int, expected_resolution: int, model_name: str):
        self._test_smoke(device, dtype, img_size, expected_resolution, model_name)

    @pytest.mark.slow
    @pytest.mark.parametrize("model_name", ["l0", "l1", "l2", "l3"])
    @pytest.mark.parametrize("img_size,expected_resolution", [(224, 7), (256, 8), (288, 9), (320, 10), (384, 12)])
    def test_smoke_large(self, device, dtype, img_size: int, expected_resolution: int, model_name: str):
        self._test_smoke(device, dtype, img_size, expected_resolution, model_name)

    def test_onnx(self, device, dtype, tmp_path: Path):
        model: vit.EfficientViTBackbone = vit.efficientvit_backbone_b0()
        model = model.to(device=device, dtype=dtype)

        image = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)

        model_path = tmp_path / "efficientvit_backbone_b0.onnx"

        torch.onnx.export(model, image, model_path, opset_version=16)

        assert model_path.is_file()

    def test_load_pretrained(self, device, dtype):
        model = EfficientViT.from_config(EfficientViTConfig())
        model = model.to(device=device, dtype=dtype)

        image = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)
        feats = model(image)
        assert feats["stage_final"].shape == torch.Size([1, 256, 7, 7])
