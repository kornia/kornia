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

"""Tests for SAM-3 architecture components."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from kornia.models.sam3.architecture import ImageEncoderHiera
from kornia.models.sam3.architecture.common import Attention, MLPBlock

from testing.base import BaseTester


class TestSam3Common(BaseTester):
    """Test common components used in SAM-3."""

    @pytest.mark.parametrize("embed_dim", [64, 256, 512])
    def test_layer_norm_smoke(self, device: str, embed_dim: int) -> None:
        """Test LayerNorm basic functionality with different dimensions."""
        B, N = 2, 64
        ln = nn.LayerNorm(embed_dim).to(device)
        x = torch.randn(B, N, embed_dim, device=device)
        out = ln(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    @pytest.mark.parametrize("embed_dim,mlp_ratio", [(64, 4.0), (256, 4.0), (512, 2.0)])
    def test_mlp_block_smoke(self, device: str, embed_dim: int, mlp_ratio: float) -> None:
        """Test MLPBlock basic functionality with different configurations."""
        B, N = 2, 64
        mlp_dim = int(embed_dim * mlp_ratio)
        mlp = MLPBlock(embed_dim, mlp_dim).to(device)
        x = torch.randn(B, N, embed_dim, device=device)
        out = mlp(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    @pytest.mark.parametrize("dim,heads", [(256, 8), (512, 16), (768, 12)])
    def test_attention_smoke(self, device: str, dim: int, heads: int) -> None:
        """Test Attention basic functionality with different configurations."""
        B, N = 2, 64
        attn = Attention(dim, heads=heads).to(device)
        x = torch.randn(B, N, dim, device=device)
        out = attn(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype


class TestImageEncoderHiera(BaseTester):
    """Test ImageEncoderHiera architecture."""

    @pytest.mark.parametrize("img_size,embed_dim", [(512, 128), (1024, 256)])
    def test_image_encoder_hiera_smoke(self, device: str, img_size: int, embed_dim: int) -> None:
        """Test ImageEncoderHiera basic functionality with different configs."""
        patch_size = 16
        encoder = ImageEncoderHiera(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=2,
            num_heads=4,
        ).to(device)

        B = 1
        x = torch.randn(B, 3, img_size, img_size, device=device)
        features = encoder(x)

        # Expected output shape
        num_patches = (img_size // patch_size) ** 2
        assert features.shape == (B, num_patches, embed_dim)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_image_encoder_hiera_cardinality(self, device: str, batch_size: int) -> None:
        """Test ImageEncoderHiera output cardinality with different batch sizes."""
        encoder = ImageEncoderHiera(
            img_size=512,
            patch_size=16,
            embed_dim=256,
            depth=2,
            num_heads=8,
        ).to(device)

        x = torch.randn(batch_size, 3, 512, 512, device=device)
        features = encoder(x)
        assert features.shape == (batch_size, (512 // 16) ** 2, 256)

    def test_image_encoder_hiera_output_shape(self) -> None:
        """Test output shape computation method."""
        encoder = ImageEncoderHiera(
            img_size=1024,
            patch_size=16,
            embed_dim=256,
        )

        output_shape = encoder.get_output_shape((1, 3, 1024, 1024))
        expected = (1, (1024 // 16) ** 2, 256)
        assert output_shape == expected

    def test_mlp_block_gradcheck(self, device: str) -> None:
        """Test MLPBlock gradient computation."""
        mlp = MLPBlock(64, 256).to(device).double()
        x = torch.randn(2, 4, 64, device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(mlp, (x,), raise_exception=True)

    def test_attention_gradcheck(self, device: str) -> None:
        """Test Attention gradient computation."""
        attn = Attention(256, heads=8).to(device).double()
        x = torch.randn(1, 16, 256, device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(attn, (x,), raise_exception=True)

    def test_image_encoder_hiera_gradcheck(self, device: str) -> None:
        """Test ImageEncoderHiera gradient computation."""
        encoder = ImageEncoderHiera(
            img_size=64,
            patch_size=16,
            embed_dim=64,
            depth=1,
            num_heads=2,
        ).to(device).double()
        x = torch.randn(1, 3, 64, 64, device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(encoder, (x,), raise_exception=True)
