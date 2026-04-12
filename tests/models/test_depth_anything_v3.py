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

import pytest
import torch
from torch.autograd import gradcheck

from kornia.models.depth_anything_v3.common import MLP, Attention, Block, LayerScale

from testing.base import BaseTester


# === tests for the class Atention ===
class TestAttention(BaseTester):
    def test_smoke(self, device, dtype):
        model = Attention(dim=64, nb_head=8).to(device=device, dtype=dtype)
        x = torch.randn(2, 14, 64, device=device, dtype=dtype)
        out = model(x)
        assert out.shape == (2, 14, 64)

    def test_cardinality(self, device, dtype):
        batch_size, seq_len, dim, nb_head = 2, 14, 64, 8
        model = Attention(dim=dim, nb_head=nb_head).to(device=device, dtype=dtype)
        x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
        out = model(x)
        assert out.shape == (batch_size, seq_len, dim)
        assert out.dtype == dtype
        assert out.device.type == device.type

    def test_exception(self, device, dtype):
        with pytest.raises(ValueError, match="dim must be > 0"):
            Attention(dim=0, nb_head=4)

        with pytest.raises(ValueError, match="nb_head must be > 0"):
            Attention(dim=64, nb_head=0)

        with pytest.raises(ValueError, match="must be divisible"):
            Attention(dim=64, nb_head=3)

    def test_gradcheck(self, device):
        model = Attention(dim=64, nb_head=4).to(device=device, dtype=torch.float64)
        x = torch.randn(2, 14, 64, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(model, (x,), raise_exception=True, fast_mode=True)

    def test_module(self, device, dtype):
        model = Attention(dim=64, nb_head=4).to(device=device, dtype=dtype)
        x = torch.randn(2, 14, 64, device=device, dtype=dtype, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        for name, param in model.named_parameters():
            assert param.grad is not None, f"the parameter : {name} has no gradient"
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad), atol=1e-12), (
                f"the parameter : {name} has no gradient"
            )

    def test_dynamo(self, device, dtype, torch_optimizer):
        model = Attention(dim=64, nb_head=8).to(device=device, dtype=dtype)
        x = torch.randn(2, 14, 64, device=device, dtype=dtype)
        op = torch_optimizer(model)
        actual = op(x)
        expected = model(x)
        self.assert_close(actual, expected)


# === test for the class Layerscale ===
class TestLayerScale(BaseTester):
    def test_smoke(self, device, dtype):
        model = LayerScale(dim=64).to(device=device, dtype=dtype)
        x = torch.randn(2, 14, 64, device=device, dtype=dtype)
        out = model(x)
        assert out.shape == (2, 14, 64)

    def test_cardinality(self, device, dtype):
        batch_size, seq_len, dim = 2, 14, 64
        model = LayerScale(dim=dim).to(device=device, dtype=dtype)
        x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
        out = model(x)
        assert out.shape == (batch_size, seq_len, dim)
        assert out.dtype == dtype
        assert out.device.type == device.type

    def test_exception(self, device, dtype):
        with pytest.raises(ValueError, match="dim must be > 0"):
            LayerScale(dim=-1)

    def test_gradcheck(self, device):
        model = LayerScale(dim=64).to(device=device, dtype=torch.float64)
        x = torch.randn(2, 14, 64, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(model, (x,), raise_exception=True, fast_mode=True)

    def test_module(self, device, dtype):
        dim = 64
        init_val = 1e-5
        model = LayerScale(dim=dim, init_value=init_val).to(device=device, dtype=dtype)

        x = torch.ones(1, 1, dim, device=device, dtype=dtype)
        out = model(x)
        expected = torch.ones(1, 1, dim, device=device, dtype=dtype) * init_val
        self.assert_close(out, expected)

        # check backprop
        x2 = torch.randn(2, 14, dim, device=device, dtype=dtype, requires_grad=True)
        out2 = model(x2)
        out2.sum().backward()
        assert x2.grad is not None
        assert model.gamma.grad is not None
        assert not torch.allclose(model.gamma.grad, torch.zeros_like(model.gamma.grad))

    def test_dynamo(self, device, dtype, torch_optimizer):
        model = LayerScale(dim=64).to(device=device, dtype=dtype)
        x = torch.randn(2, 14, 64, device=device, dtype=dtype)
        op = torch_optimizer(model)
        actual = op(x)
        expected = model(x)
        self.assert_close(actual, expected)


# == test for the class MLP ===
class TestMLP(BaseTester):
    def test_smoke(self, device, dtype):
        model = MLP(dim_in_f=64).to(device=device, dtype=dtype)
        x = torch.randn(2, 14, 64, device=device, dtype=dtype)
        out = model(x)
        assert out.shape == (2, 14, 64)

    def test_cardinality(self, device, dtype):
        batch_size, seq_len, in_features = 2, 14, 64

        # basic case : all dims equal to in_features
        model_default = MLP(dim_in_f=in_features).to(device=device, dtype=dtype)
        x = torch.randn(batch_size, seq_len, in_features, device=device, dtype=dtype)
        out_default = model_default(x)
        assert out_default.shape == (batch_size, seq_len, in_features)
        assert out_default.dtype == dtype
        assert out_default.device.type == device.type

        # test with custom hidden and output dims
        out_features = 32
        model_custom = MLP(dim_in_f=in_features, dim_hidden_f=in_features * 4, dim_out_f=out_features).to(
            device=device, dtype=dtype
        )
        out_custom = model_custom(x)
        assert out_custom.shape == (batch_size, seq_len, out_features)
        assert out_custom.dtype == dtype

    def test_exception(self, device, dtype):
        with pytest.raises(ValueError, match="dimension of the hidden layer must be > 0"):
            MLP(dim_in_f=1, dim_hidden_f=-1, dim_out_f=1)
        with pytest.raises(ValueError, match="dimension of the output must be > 0"):
            MLP(dim_in_f=1, dim_hidden_f=1, dim_out_f=-1)
        with pytest.raises(ValueError, match="dimension of the input must be > 0"):
            MLP(dim_in_f=-1, dim_hidden_f=1, dim_out_f=1)

    def test_gradcheck(self, device):
        model = MLP(dim_in_f=64, dim_hidden_f=128).to(device=device, dtype=torch.float64)
        x = torch.randn(2, 14, 64, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(model, (x,), raise_exception=True, fast_mode=True)

    def test_module(self, device, dtype):
        in_features = 64
        model = MLP(dim_in_f=in_features, dim_hidden_f=in_features * 4).to(device=device, dtype=dtype)
        x = torch.randn(2, 14, in_features, device=device, dtype=dtype, requires_grad=True)
        out = model(x)
        out.sum().backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad))

    def test_dynamo(self, device, dtype, torch_optimizer):
        model = MLP(dim_in_f=64, dim_hidden_f=256).to(device=device, dtype=dtype)
        x = torch.randn(2, 14, 64, device=device, dtype=dtype)
        op = torch_optimizer(model)
        actual = op(x)
        expected = model(x)
        self.assert_close(actual, expected)


# === test for the block class ===
class TestBlock(BaseTester):
    def test_smoke(self, device, dtype):
        model = Block(dim=64, nb_head=4, dim_hidden_f=128).to(device=device, dtype=dtype)
        x = torch.randn(2, 14, 64, device=device, dtype=dtype)
        out = model(x)
        assert out.shape == (2, 14, 64)

    def test_cardinality(self, device, dtype):
        batch_size, seq_len, dim, nb_head = 2, 14, 64, 4
        model = Block(dim=dim, nb_head=nb_head, dim_hidden_f=dim * 2).to(device=device, dtype=dtype)
        x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
        out = model(x)
        assert out.shape == (batch_size, seq_len, dim)
        assert out.dtype == dtype
        assert out.device.type == device.type

    def test_exception(self, device, dtype):
        with pytest.raises(ValueError, match="dim must be > 0"):
            Block(dim=0, nb_head=4, dim_hidden_f=64)

        with pytest.raises(ValueError, match="nb_head must be > 0"):
            Block(dim=64, nb_head=0, dim_hidden_f=64)

        with pytest.raises(ValueError, match="must be divisible"):
            Block(dim=64, nb_head=3, dim_hidden_f=64)

    def test_gradcheck(self, device):
        model = Block(dim=64, nb_head=4, dim_hidden_f=128).to(device=device, dtype=torch.float64)
        x = torch.randn(2, 14, 64, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(model, (x,), raise_exception=True, fast_mode=True)

    def test_module(self, device, dtype):
        dim, nb_head = 64, 4
        model = Block(dim=dim, nb_head=nb_head, dim_hidden_f=dim * 2).to(device=device, dtype=dtype)
        x = torch.randn(2, 14, dim, device=device, dtype=dtype, requires_grad=True)
        out = model(x)
        out.sum().backward()

        assert x.grad is not None

        assert model.attention_layer.w_q.weight.grad is not None
        assert model.mlp_layer.fc1.weight.grad is not None
        assert model.mlp_layer.fc2.weight.grad is not None
        assert model.layerscale1.gamma.grad is not None

    def test_dynamo(self, device, dtype, torch_optimizer):
        model = Block(dim=64, nb_head=4, dim_hidden_f=128).to(device=device, dtype=dtype)
        x = torch.randn(2, 14, 64, device=device, dtype=dtype)
        op = torch_optimizer(model)
        actual = op(x)
        expected = model(x)
        self.assert_close(actual, expected)
