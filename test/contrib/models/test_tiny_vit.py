import pytest
import torch

from kornia.contrib.models.tiny_vit import TinyViT
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
        model = TinyViT(num_classes=num_classes).to(device=device, dtype=dtype)
        inpt = torch.rand(batch_size, 3, model.img_size, model.img_size, device=device, dtype=dtype)

        out = model(inpt)
        assert out.shape == (batch_size, num_classes)

    @pytest.mark.skip('not implemented')
    def test_exception(self):
        ...

    @pytest.mark.skip('not implemented')
    def test_gradcheck(self):
        ...

    @pytest.mark.skip('not implemented')
    def test_module(self):
        ...

    @pytest.mark.slow
    def test_dynamo(self, device, dtype, torch_optimizer):
        op = TinyViT().to(device=device, dtype=dtype)
        img = torch.rand(1, 3, op.img_size, op.img_size, device=device, dtype=dtype)

        op_optimized = torch_optimizer(op)
        self.assert_close(op(img), op_optimized(img))

    @pytest.mark.slow
    @pytest.mark.parametrize("pretrained", [False, True])
    @pytest.mark.parametrize("variant", ["5m", "11m", "21m"])
    def test_from_config(self, variant, pretrained):
        model = TinyViT.from_config(variant, pretrained=pretrained)
        assert isinstance(model, TinyViT)

    @pytest.mark.parametrize('num_classes', [1000, 8])
    @pytest.mark.parametrize("img_size", [224, 256])
    def test_pretrained(self, img_size, num_classes):
        model = TinyViT.from_config("5m", img_size=img_size, num_classes=num_classes, pretrained=True)
        assert isinstance(model, TinyViT)

    @pytest.mark.slow
    def test_mobile_sam_backbone(self, device, dtype):
        img_size = 1024
        batch_size = 1
        model = TinyViT.from_config("5m", img_size=img_size, mobile_sam=True).to(device=device, dtype=dtype)
        inpt = torch.randn(batch_size, 3, img_size, img_size, device=device, dtype=dtype)

        out = model(inpt)
        assert out.shape == (batch_size, 256, img_size // 16, img_size // 16)
