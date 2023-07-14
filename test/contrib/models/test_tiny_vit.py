import pytest
import torch

from kornia.contrib.models.tiny_vit import TinyViT, tiny_vit_5m, tiny_vit_11m, tiny_vit_21m
from kornia.core import Tensor
from kornia.testing import BaseTester


class TestTinyViT(BaseTester):
    @pytest.mark.parametrize("img_size", [224, 256])
    def test_smoke(self, device, dtype, img_size):
        model = TinyViT(img_size=img_size).to(device=device, dtype=dtype)
        inpt = torch.randn(1, 3, img_size, img_size, device=device, dtype=dtype)
        out = model(inpt)
        assert isinstance(out, Tensor)

    @pytest.mark.parametrize('num_classes', [10, 100])
    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_cardinality(self, device, dtype, batch_size, num_classes):
        img_size = 224
        model = TinyViT(img_size=img_size, num_classes=num_classes).to(device=device, dtype=dtype)
        inpt = torch.rand(batch_size, 3, img_size, img_size, device=device, dtype=dtype)
        out = model(inpt)
        assert out.shape == (batch_size, num_classes)

    def test_exception(self):
        ...

    def test_gradcheck(self):
        ...

    def test_module(self):
        ...

    def test_dynamo(self, device, dtype, torch_optimizer):
        img_size = 224
        img = torch.rand(1, 3, img_size, img_size, device=device, dtype=dtype)

        op = TinyViT(img_size=img_size).to(device=device, dtype=dtype)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(img), op_optimized(img))

    @pytest.mark.parametrize("factory", [tiny_vit_5m, tiny_vit_11m, tiny_vit_21m])
    def test_pretrained(self, device, dtype, factory):
        img_size = 224
        factory(img_size=img_size, pretrained=True).to(device=device, dtype=dtype)

    @pytest.mark.parametrize('num_classes', [1000, 8])
    @pytest.mark.parametrize("img_size", [224, 256])
    def test_pretrained_complex(self, device, dtype, img_size, num_classes):
        tiny_vit_5m(img_size=img_size, num_classes=num_classes, pretrained=True).to(device=device, dtype=dtype)

    def test_mobile_sam_backbone(self, device, dtype):
        img_size = 1024
        batch_size = 1
        model = tiny_vit_5m(img_size=img_size, mobile_sam=True).to(device=device, dtype=dtype)
        inpt = torch.randn(batch_size, 3, img_size, img_size, device=device, dtype=dtype)
        out = model(inpt)

        assert out.shape == (batch_size, 256, img_size // 16, img_size // 16)
