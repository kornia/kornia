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

from kornia.models.tiny_vit import Attention, TinyViT, urls

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


def _windows(model: TinyViT) -> list[int]:
    # The window size of each attention stage; stage 0 is convolutional.
    return [layer.blocks[0].window_size for layer in model.layers[1:]]


# TinyViT-21M's layout, from Cream tiny_vit_21m_224/_384/_512 (microsoft/Cream@8dc3882, TinyViT/models/tiny_vit.py).
_21M_LAYOUT = {"embed_dims": [96, 192, 384, 576], "depths": [2, 2, 6, 2], "num_heads": [3, 6, 12, 18]}


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

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize(
        ("img_size", "key", "window", "pretrained_values", "num_classes"),
        [
            # True picks in1k below 384 px; the in22k checkpoint has a 21841-class head and is only reached by name.
            (224, "in1k", 7, (True, "in1k"), 1000),
            (224, "in22k", 7, ("in22k",), 21841),
            (384, "in1k_384", 12, (True, "in1k_384"), 1000),
            (512, "in1k_512", 16, (True, "in1k_512"), 1000),
        ],
    )
    def test_21m_checkpoints_keep_their_windows(
        self, monkeypatch, img_size, key, window, pretrained_values, num_classes
    ):
        # Cream builds tiny_vit_21m_224 with window_sizes [7, 7, 14, 7], tiny_vit_21m_384 with [12, 12, 24, 12] and
        # tiny_vit_21m_512 with [16, 16, 32, 16] (microsoft/Cream@8dc3882, TinyViT/models/tiny_vit.py); each checkpoint
        # holds the attention biases of its windows. The stand-in has that shape and N(0, 1) biases.
        windows = [window, window, 2 * window, window]
        stand_in = TinyViT(img_size=img_size, window_sizes=windows, num_classes=num_classes, **_21M_LAYOUT)
        checkpoint = _random_biases(stand_in).state_dict()
        biases = {k: v.clone() for k, v in checkpoint.items() if "attention_biases" in k}
        fetched: list[str | list[str]] = []

        def fake_loader(url: str | list[str], *args: object, **kwargs: object) -> dict[str, torch.Tensor]:
            fetched.append(url)
            return dict(checkpoint)

        monkeypatch.setattr("kornia.models.tiny_vit.load_state_dict_from_url", fake_loader)

        for pretrained in pretrained_values:
            model = TinyViT.from_config("21m", pretrained=pretrained, img_size=img_size, num_classes=num_classes)
            assert fetched[-1] == urls["21m"][key]
            assert _windows(model) == windows[1:]
            # Loaded as they are: interpolating them to other windows would change their shape.
            state_dict = model.state_dict()
            for name, bias in biases.items():
                assert torch.equal(state_dict[name], bias), name

        # An explicit ``window_sizes`` wins; the biases are then interpolated to it.
        model = TinyViT.from_config(
            "21m", pretrained=key, img_size=img_size, num_classes=num_classes, window_sizes=[8, 8, 16, 8]
        )
        assert _windows(model) == [8, 16, 8]
        # Random init has no checkpoint to follow and keeps the 224-px windows.
        assert _windows(TinyViT.from_config("21m", img_size=img_size)) == [7, 14, 7]

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize(
        ("variant", "embed_dims", "num_heads"),
        # Each stage's dimension stays divisible by its number of heads, which TinyViTBlock checks; the defaults are
        # 5m [2, 4, 5, 10], 11m [2, 4, 8, 14] and 21m [3, 6, 12, 18] heads.
        [
            ("5m", [32, 64, 160, 320], [2, 4, 4, 8]),
            ("11m", [32, 64, 256, 448], [1, 2, 4, 7]),
            ("21m", [48, 96, 384, 576], [2, 4, 8, 12]),
        ],
    )
    def test_builder_arguments_can_be_overridden(self, variant, embed_dims, num_heads):
        model = TinyViT.from_config(
            variant,
            window_sizes=[8, 8, 16, 8],
            drop_path_rate=0.3,
            embed_dims=embed_dims,
            depths=[1, 1, 2, 1],
            num_heads=num_heads,
        )

        assert _windows(model) == [8, 16, 8]
        stage_dims = [model.layers[0].blocks[0].conv3.c.out_channels]
        stage_dims += [layer.blocks[0].attn.norm.normalized_shape[0] for layer in model.layers[1:]]
        assert stage_dims == embed_dims
        assert [len(layer.blocks) for layer in model.layers] == [1, 1, 2, 1]
        # Stage 0 is convolutional and has no heads.
        assert [layer.blocks[0].attn.num_heads for layer in model.layers[1:]] == num_heads[1:]
        # The stochastic depth rate rises linearly to drop_path_rate at the last block.
        assert model.layers[-1].blocks[-1].drop_path1.drop_prob == pytest.approx(0.3)

    @pytest.mark.parametrize(
        ("img_size", "mobile_sam", "size"),
        [
            (224, False, (224, 224)),
            (224, False, (220, 220)),
            (224, False, (217, 224)),
            (64, True, (64, 64)),
            (64, True, (57, 64)),
        ],
    )
    def test_forward_takes_sizes_of_its_token_grid(self, device, dtype, img_size, mobile_sam, size):
        if device.type == "mps" and dtype == torch.float64:
            pytest.skip("MPS does not support float64")
        # The stem and the first patch merging downsample by 8 with ceil rounding, so sides from img_size - 7 to
        # img_size give the img_size // 8 grid the attention stages are built for, and their result is correct.
        model = TinyViT.from_config("5m", img_size=img_size, mobile_sam=mobile_sam).to(device, dtype).eval()

        with torch.no_grad():
            out = model(torch.rand(1, 3, *size, device=device, dtype=dtype))

        assert out.shape == ((1, 256, img_size // 16, img_size // 16) if mobile_sam else (1, 1000))

    @pytest.mark.parametrize(
        ("img_size", "mobile_sam", "size"),
        [
            (224, False, (112, 448)),
            (224, False, (448, 112)),
            (224, False, (216, 216)),
            (224, False, (225, 225)),
            (224, False, (224, 256)),
            (224, False, (256, 224)),
            (64, True, (32, 128)),
        ],
    )
    def test_forward_rejects_other_token_grids(self, device, dtype, img_size, mobile_sam, size):
        if device.type == "mps" and dtype == torch.float64:
            pytest.skip("MPS does not support float64")
        # 112 x 448 and 448 x 112 give 14 x 56 and 56 x 14 grids of the 784 tokens of 28 x 28, which the stages would
        # read as 28 x 28; 32 x 128 gives MobileSAM's 4 x 16 grid of the 64 tokens of 8 x 8. 216 and 225 give 27 x 27
        # and 29 x 29; 224 x 256 and 256 x 224 have one side right (28 x 32 and 32 x 28).
        model = TinyViT.from_config("5m", img_size=img_size, mobile_sam=mobile_sam).to(device, dtype).eval()

        with torch.no_grad(), pytest.raises(ValueError, match=f"{img_size // 8}x{img_size // 8}"):
            model(torch.rand(1, 3, *size, device=device, dtype=dtype))

    @pytest.mark.parametrize("mobile_sam", [False, True])
    def test_dynamo_fullgraph(self, device, dtype, torch_optimizer, mobile_sam):
        if device.type == "mps" and dtype == torch.float64:
            pytest.skip("MPS does not support float64")
        # The token-grid check reads static shapes: it must compile without a graph break.
        op = _random_biases(TinyViT.from_config("5m", img_size=64, mobile_sam=mobile_sam)).eval().to(device, dtype)
        img = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)

        op_optimized = torch_optimizer(op, fullgraph=True)
        # As in test_dynamo_eval_mode: inductor keeps fused half-precision intermediates in float32.
        tol = {torch.float16: 1e-2, torch.bfloat16: 5e-2}.get(dtype)
        with torch.no_grad():
            self.assert_close(op(img), op_optimized(img), rtol=tol, atol=tol)

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
