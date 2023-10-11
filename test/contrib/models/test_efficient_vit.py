from pathlib import Path

import pytest
import torch

from kornia.contrib.models.efficient_vit import EfficientViTBackbone, efficientvit_backbone_b0


class TestEfficientViT:
    @pytest.mark.parametrize("img_size,expected_resolution", [(224, 7), (256, 8)])
    def test_smoke(self, device, dtype, img_size: int, expected_resolution: int):
        model: EfficientViTBackbone = efficientvit_backbone_b0(device=device, dtype=dtype)
        image = torch.randn(1, 3, img_size, img_size, device=device, dtype=dtype)

        out = model(image)

        assert "input" in out
        assert out["input"].shape == image.shape

        assert "stage_final" in out
        assert out["stage_final"].shape == torch.Size([1, 128, expected_resolution, expected_resolution])

    def test_onnx(self, device, dtype, tmp_path: Path):
        model: EfficientViTBackbone = efficientvit_backbone_b0(device=device, dtype=dtype)
        image = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)

        model_path = tmp_path / "efficientvit_backbone_b0.onnx"

        torch.onnx.export(model, image, model_path, opset_version=16)

        assert model_path.is_file()
