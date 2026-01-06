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
from torch import Tensor

from kornia.models.processors.naflex import NaFlex

from testing.base import BaseTester


class TestNaFlex(BaseTester):
    @pytest.fixture
    def model(self) -> NaFlex:
        """Create a NaFlex model with mock embeddings for testing.

        Returns:
            NaFlex instance with mock patch embedding function and position embedding.
        """

        def mock_patch_embedding(x: Tensor) -> Tensor:
            """Mock patch embedding function simulating Conv2d output.

            Dynamically calculates output size based on patch_size=16.
            """
            B, _, H, W = x.shape
            h_out = H // 16
            w_out = W // 16
            return torch.randn(B, 768, h_out, w_out, dtype=x.dtype, device=x.device)

        position_embedding = torch.randn(196, 768)
        return NaFlex(
            patch_embedding_fcn=mock_patch_embedding,
            position_embedding=position_embedding,
        )

    def test_smoke(self, model: NaFlex, device: torch.device, dtype: torch.dtype) -> None:
        """Test basic forward pass with standard input."""
        input_data = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)
        out = model(input_data)
        assert isinstance(out, Tensor)
        assert out.shape == (1, 196, 768)

    def test_cardinality(self, model: NaFlex, device: torch.device, dtype: torch.dtype) -> None:
        """Test output cardinality with non-square input resolution.

        For 224x320 input with 16x16 patches, expect 14x20=280 patches.
        """
        input_data = torch.randn(1, 3, 224, 320, device=device, dtype=dtype)
        out = model(input_data)
        assert out.shape[0] == 1
        assert out.shape[1] == 280
        assert out.shape[2] == 768

    def test_exception(self, device: torch.device, dtype: torch.dtype) -> None:
        """Test that invalid position embeddings raise appropriate errors."""

        def fake_patch_fcn(x: Tensor) -> Tensor:
            return torch.randn(1, 100, 768, device=device, dtype=dtype)

        bad_pos_embed = torch.randn(200, 768, device=device, dtype=dtype)
        wrapper_bad = NaFlex(fake_patch_fcn, bad_pos_embed)
        input_data = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)

        with pytest.raises(ValueError, match="Original positional embedding is not a square grid"):
            wrapper_bad(input_data)

    def test_interpolation(self, device: torch.device, dtype: torch.dtype) -> None:
        """Test positional embedding interpolation for different input sizes."""

        def mock_patch_embedding_dynamic(x: Tensor) -> Tensor:
            """Dynamic mock for resizing tests."""
            B, _, H, W = x.shape
            h_out = H // 16
            w_out = W // 16
            return torch.randn(B, 768, h_out, w_out, dtype=x.dtype, device=x.device)

        position_embedding = torch.randn(196, 768)
        model = NaFlex(mock_patch_embedding_dynamic, position_embedding)
        input_448 = torch.randn(1, 3, 448, 448, device=device, dtype=dtype)
        out = model(input_448)
        assert out.shape == (1, 784, 768)
