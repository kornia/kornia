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

import copy
import sys

import pytest
import torch
from torch import nn

from kornia.models.tiny_vit import Attention, TinyViT

from testing.base import BaseTester


def _attentions(model: nn.Module) -> list[Attention]:
    return [m for m in model.modules() if isinstance(m, Attention)]


def _random_biases(model: TinyViT) -> TinyViT:
    # ``attention_biases`` starts as zeros, so a bias read from a stale copy would still be right. Seeded N(0, 1)
    # values make every relative-position bias table distinct from its initial value.
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for attention in _attentions(model):
            attention.attention_biases.copy_(torch.randn(attention.attention_biases.shape, generator=generator))
    return model


def _modes(model: nn.Module) -> set[bool]:
    # The ``training`` flags of the model and every submodule: ``{False}`` is evaluation mode throughout.
    return {m.training for m in model.modules()}


def _fake_5m_checkpoint(url: str | list[str], *args: object, **kwargs: object) -> dict[str, torch.Tensor]:
    # Stands in for ``load_state_dict_from_url``. The ImageNet-22k checkpoints (``*_22k_distill.pth``) carry a
    # 21841-class head and the ImageNet-1k ones (``*_22kto1k_distill.pth``) a 1000-class head
    # (docs/source/models/tiny_vit.rst).
    candidates = [url] if isinstance(url, str) else url
    num_classes = 21841 if any("_22k_distill" in candidate for candidate in candidates) else 1000
    return TinyViT.from_config("5m", num_classes=num_classes).state_dict()


def _image(batch_size: int = 1, seed: int = 0) -> torch.Tensor:
    # TinyViT takes only its square ``img_size`` (224 by default).
    return torch.rand(batch_size, 3, 224, 224, generator=torch.Generator().manual_seed(seed))


class TestTinyViT(BaseTester):
    # 128 is not a multiple of the 7-pixel attention window at any stage, so it takes the padded
    # window-attention path; 224 divides evenly and takes the unpadded one. Both belong here: the
    # only other coverage of the exact-multiple path is marked slow, which PR CI skips.
    @pytest.mark.parametrize("img_size", [128, 224])
    def test_smoke(self, device, dtype, img_size):
        model = TinyViT(img_size=img_size).to(device=device, dtype=dtype)
        data = torch.randn(1, 3, img_size, img_size, device=device, dtype=dtype)

        out = model(data)
        assert isinstance(out, torch.Tensor)

    @pytest.mark.slow
    @pytest.mark.parametrize("num_classes", [10, 100])
    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_cardinality(self, device, dtype, batch_size, num_classes):
        model = TinyViT(num_classes=num_classes).to(device=device, dtype=dtype)
        data = torch.rand(batch_size, 3, model.img_size, model.img_size, device=device, dtype=dtype)

        out = model(data)
        assert out.shape == (batch_size, num_classes)

    @pytest.mark.skip("not implemented")
    def test_exception(self): ...

    @pytest.mark.skip("not implemented")
    def test_gradcheck(self): ...

    @pytest.mark.skip("not implemented")
    def test_module(self): ...

    @pytest.mark.skipif(sys.version_info.major == 3 and sys.version_info.minor == 8, reason="not working for py3.8")
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

    @pytest.mark.slow
    @pytest.mark.parametrize("num_classes", [1000, 8])
    @pytest.mark.parametrize("img_size", [224, 256])
    def test_pretrained(self, img_size, num_classes):
        model = TinyViT.from_config("5m", img_size=img_size, num_classes=num_classes, pretrained=True)
        assert isinstance(model, TinyViT)

    @pytest.mark.slow
    def test_mobile_sam_backbone(self, device, dtype):
        img_size = 1024
        batch_size = 1
        model = TinyViT.from_config("5m", img_size=img_size, mobile_sam=True).to(device=device, dtype=dtype)
        data = torch.randn(batch_size, 3, img_size, img_size, device=device, dtype=dtype)

        out = model(data)
        assert out.shape == (batch_size, 256, img_size // 16, img_size // 16)

    @pytest.mark.device_agnostic
    def test_eval_bias_follows_load_state_dict(self):
        src = _random_biases(TinyViT.from_config("5m"))
        dst = TinyViT.from_config("5m").eval()
        x = _image()
        with torch.no_grad():
            dst(x)  # an evaluation-mode forward before the load, so a bias kept from it would go stale
        dst.load_state_dict(src.state_dict())
        src.eval()

        with torch.no_grad():
            assert torch.equal(dst(x), src(x))

    @pytest.mark.device_agnostic
    def test_eval_mode_bias_gets_gradients(self):
        model = _random_biases(TinyViT.from_config("5m")).eval()

        model(_image()).sum().backward()

        for attention in _attentions(model):
            assert attention.attention_biases.grad is not None
            assert attention.attention_biases.grad.abs().sum() > 0

    @pytest.mark.device_agnostic
    def test_eval_bias_follows_parameter_update(self):
        model = _random_biases(TinyViT.from_config("5m")).eval()
        x = _image()
        with torch.no_grad():
            y0 = model(x)
            # Doubling N(0, 1) biases changes the gaps between the logits of a row, which the softmax keeps; a
            # constant shift of every bias would cancel in the softmax and could not show a stale bias.
            for attention in _attentions(model):
                attention.attention_biases.mul_(2.0)
            y1 = model(x)

            fresh = TinyViT.from_config("5m")
            fresh.load_state_dict(model.state_dict())
            expected = fresh.eval()(x)

        self.assert_close(y1, expected)
        # The update moves the logits by 0.11 at most; float32 rounding alone moves them by about 4e-7.
        assert (y1 - y0).abs().max() > 1e-2

    def test_eval_then_cast_matches_cast_then_eval(self, device, dtype):
        if device.type == "mps" and dtype == torch.float64:
            pytest.skip("MPS does not support float64")
        evaluated_first = _random_biases(TinyViT.from_config("5m"))
        cast_first = copy.deepcopy(evaluated_first)
        evaluated_first.eval().to(device, dtype)
        cast_first.to(device, dtype).eval()
        x = _image().to(device, dtype)

        with torch.no_grad():
            assert torch.equal(evaluated_first(x), cast_first(x))

    @pytest.mark.device_agnostic
    def test_from_config_mode(self, monkeypatch):
        assert _modes(TinyViT.from_config("5m")) == {True}
        assert _modes(TinyViT.from_config("5m", pretrained=False)) == {True}

        monkeypatch.setattr("kornia.models.tiny_vit.load_state_dict_from_url", _fake_5m_checkpoint)
        assert _modes(TinyViT.from_config("5m", pretrained=True)) == {False}
        assert _modes(TinyViT.from_config("5m", pretrained="in1k", num_classes=1000)) == {False}
        assert _modes(TinyViT.from_config("5m", pretrained="in22k", num_classes=21841)) == {False}

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize(
        ("pretrained", "num_classes"),
        [(True, 8), (True, 1001), ("in22k", 1000)],
        ids=["head-narrower", "head-wider", "in22k-default-classes"],
    )
    def test_from_config_partial_weights_mode(self, monkeypatch, pretrained, num_classes):
        monkeypatch.setattr("kornia.models.tiny_vit.load_state_dict_from_url", _fake_5m_checkpoint)

        # A class count other than the checkpoint's (1000 for in1k, 21841 for in22k) re-initialises the head: a
        # pretrained backbone to fine-tune, so training mode.
        with pytest.warns(UserWarning, match="Resetting classification head"):
            model = TinyViT.from_config("5m", pretrained=pretrained, num_classes=num_classes)
        assert model.head.out_features == num_classes
        assert not model.head.weight.any()
        assert not model.head.bias.any()
        assert _modes(model) == {True}

    @pytest.mark.skipif(sys.version_info.major == 3 and sys.version_info.minor == 8, reason="not working for py3.8")
    def test_dynamo_eval_mode(self, device, dtype, torch_optimizer):
        if device.type == "mps" and dtype == torch.float64:
            pytest.skip("MPS does not support float64")
        # Evaluation mode first, then the cast: the order of a pretrained model moved to its device.
        op = _random_biases(TinyViT.from_config("5m")).eval().to(device, dtype)
        img = _image().to(device, dtype)

        op_optimized = torch_optimizer(op)
        # Inductor keeps fused intermediates in float32 where eager rounds every op to the half dtype. Over the network
        # the two differ on MPS by up to 1.1e-3 in float16 and 1.6e-2 (two bfloat16 steps at the logit scale of 2) in
        # bfloat16, past the default tolerances of 1e-3 and 7.8e-3.
        tol = {torch.float16: 1e-2, torch.bfloat16: 5e-2}.get(dtype)
        with torch.no_grad():
            self.assert_close(op(img), op_optimized(img), rtol=tol, atol=tol)

    @pytest.mark.slow
    def test_pretrained_is_batch_independent(self):
        model = TinyViT.from_config("5m", pretrained=True)
        running_means = [m.running_mean.clone() for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        x1, x2 = _image(seed=0), _image(seed=1)

        with torch.no_grad():
            alone = model(x1)[0]
            batched = model(torch.cat([x1, x2]))[0]

        # In evaluation mode sample 0 does not see sample 1; only the kernels' summation order may change with the
        # batch size, so the float32 tolerance is widened to 1e-4.
        self.assert_close(alone, batched, rtol=1e-4, atol=1e-4)
        after = [m.running_mean for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        assert all(torch.equal(a, b) for a, b in zip(after, running_means))
