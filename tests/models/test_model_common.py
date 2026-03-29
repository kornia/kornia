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

from __future__ import annotations

import torch

from kornia.models.common import MLP, ConvNormAct, DropPath, LayerNorm2d, window_partition, window_unpartition


class TestConvNormAct:
    def test_odd_kernel_size(self):
        # Odd kernel_size uses symmetric padding (no self.pad attribute added)
        layer = ConvNormAct(3, 16, kernel_size=3)
        assert not hasattr(layer, "pad")
        x = torch.rand(2, 3, 8, 8)
        out = layer(x)
        assert out.shape == (2, 16, 8, 8)

    def test_even_kernel_size_uses_asymmetric_pad(self):
        # Even kernel_size (e.g. 2) triggers the asymmetric padding branch
        layer = ConvNormAct(3, 16, kernel_size=2)
        assert hasattr(layer, "pad")
        x = torch.rand(2, 3, 8, 8)
        out = layer(x)
        # With kernel_size=2 and stride=1, output H and W should be preserved
        assert out.shape == (2, 16, 8, 8)

    def test_act_relu(self):
        layer = ConvNormAct(4, 8, kernel_size=1, act="relu")
        x = torch.rand(1, 4, 4, 4)
        out = layer(x)
        assert (out >= 0).all()

    def test_act_silu(self):
        layer = ConvNormAct(4, 8, kernel_size=1, act="silu")
        x = torch.rand(1, 4, 4, 4)
        out = layer(x)
        assert out.shape == (1, 8, 4, 4)

    def test_act_none_is_identity(self):
        layer = ConvNormAct(4, 8, kernel_size=1, act="none")
        x = torch.rand(1, 4, 4, 4)
        out = layer(x)
        assert out.shape == (1, 8, 4, 4)


class TestMLP:
    def test_forward_without_sigmoid(self):
        mlp = MLP(input_dim=16, hidden_dim=32, output_dim=8, num_layers=3)
        x = torch.randn(2, 16)
        out = mlp(x)
        assert out.shape == (2, 8)
        # Output is not constrained to [0, 1] when no sigmoid is applied
        assert ((out < 0.0) | (out > 1.0)).any().item()

    def test_forward_with_sigmoid_output(self):
        mlp = MLP(input_dim=16, hidden_dim=32, output_dim=8, num_layers=3, sigmoid_output=True)
        x = torch.rand(2, 16)
        out = mlp(x)
        assert out.shape == (2, 8)
        # sigmoid squashes to (0, 1)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_single_layer(self):
        mlp = MLP(input_dim=4, hidden_dim=8, output_dim=6, num_layers=1)
        x = torch.rand(1, 4)
        out = mlp(x)
        assert out.shape == (1, 6)


class TestDropPath:
    def test_inference_mode_passthrough(self):
        layer = DropPath(drop_prob=0.5)
        layer.eval()
        x = torch.ones(4, 8)
        out = layer(x)
        # In eval mode (not training), no drop should happen
        assert torch.equal(out, x)

    def test_zero_drop_prob_passthrough(self):
        layer = DropPath(drop_prob=0.0)
        layer.train()
        x = torch.ones(4, 8)
        out = layer(x)
        assert torch.equal(out, x)

    def test_training_mode_applies_drop(self):
        torch.manual_seed(0)
        layer = DropPath(drop_prob=0.99, scale_by_keep=False)
        layer.train()
        x = torch.ones(100, 8)
        out = layer(x)
        # With very high drop prob, many rows should be zeroed out
        zero_rows = (out.sum(dim=1) == 0).sum().item()
        assert zero_rows > 50, f"Expected many zero rows, got {zero_rows}"

    def test_training_mode_scale_by_keep(self):
        torch.manual_seed(42)
        layer_scaled = DropPath(drop_prob=0.5, scale_by_keep=True)
        layer_unscaled = DropPath(drop_prob=0.5, scale_by_keep=False)
        layer_scaled.train()
        layer_unscaled.train()
        x = torch.ones(1000, 4)
        out_scaled = layer_scaled(x)
        out_unscaled = layer_unscaled(x)
        # The scaled version should have a higher mean for surviving rows
        # (they get divided by keep_prob = 0.5, so surviving rows have value 2.0)
        surviving_scaled = out_scaled[out_scaled.sum(dim=1) != 0].mean()
        assert surviving_scaled > 1.5, "scale_by_keep=True should amplify surviving rows"
        surviving_unscaled = out_unscaled[out_unscaled.sum(dim=1) != 0].mean()
        assert abs(surviving_unscaled.item() - 1.0) < 0.1


class TestLayerNorm2d:
    def test_output_shape(self):
        layer = LayerNorm2d(num_channels=8)
        x = torch.rand(2, 8, 4, 4)
        out = layer(x)
        assert out.shape == x.shape

    def test_normalizes_channels(self):
        layer = LayerNorm2d(num_channels=4)
        # All-same input should produce near-zero output (before weight/bias)
        x = torch.ones(1, 4, 4, 4) * 5.0
        # The layer has learnable weight (ones) and bias (zeros) by default
        out = layer(x)
        # Mean of out along channel dim should be ~0 for uniform input
        assert out.abs().max() < 1e-5


class TestWindowPartition:
    def test_no_padding_needed(self):
        # H=8, W=8, window_size=4 -> no padding
        x = torch.rand(2, 8, 8, 16)
        windows, (Hp, Wp) = window_partition(x, window_size=4)
        assert Hp == 8 and Wp == 8
        # 2 batches * (8/4)*(8/4) = 2*4 = 8 windows
        assert windows.shape == (8, 4, 4, 16)

    def test_padding_needed(self):
        # H=7, W=9, window_size=4 -> padding needed
        x = torch.rand(2, 7, 9, 16)
        windows, (Hp, Wp) = window_partition(x, window_size=4)
        # Hp = 8 (7 padded to next multiple of 4), Wp = 12
        assert Hp == 8
        assert Wp == 12
        # 2 * (8/4)*(12/4) = 2*2*3 = 12 windows
        assert windows.shape == (12, 4, 4, 16)

    def test_roundtrip_without_padding(self):
        x = torch.rand(2, 8, 8, 16)
        windows, pad_hw = window_partition(x, window_size=4)
        reconstructed = window_unpartition(windows, window_size=4, pad_hw=pad_hw, hw=(8, 8))
        assert torch.allclose(x, reconstructed)

    def test_roundtrip_with_padding(self):
        x = torch.rand(2, 7, 9, 16)
        H, W = x.shape[1], x.shape[2]
        windows, pad_hw = window_partition(x, window_size=4)
        reconstructed = window_unpartition(windows, window_size=4, pad_hw=pad_hw, hw=(H, W))
        assert reconstructed.shape == (2, 7, 9, 16)
        assert torch.allclose(x, reconstructed)


class TestWindowUnpartition:
    def test_no_crop_needed(self):
        # Hp==H and Wp==W, so no cropping
        x = torch.rand(2, 4, 4, 8)
        windows, pad_hw = window_partition(x, window_size=4)
        out = window_unpartition(windows, window_size=4, pad_hw=pad_hw, hw=(4, 4))
        assert out.shape == (2, 4, 4, 8)
        assert torch.allclose(x, out)

    def test_crop_needed(self):
        # Create input with padding scenario: pad_hw != hw
        x = torch.rand(2, 6, 6, 8)
        windows, pad_hw = window_partition(x, window_size=4)
        # pad_hw = (8, 8), original hw = (6, 6)
        out = window_unpartition(windows, window_size=4, pad_hw=pad_hw, hw=(6, 6))
        assert out.shape == (2, 6, 6, 8)
        # Reconstructed values in the non-padded region should match original
        assert torch.allclose(x, out)
