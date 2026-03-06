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
from torch import Tensor, nn

from kornia.models.processors.naflex import NaFlex

from testing.base import BaseTester


def _make_mock_patch_fn(embed_dim: int = 768, patch_size: int = 16, output_4d: bool = True):
    """Create a deterministic mock patch embedding function.

    Args:
        embed_dim: Embedding dimension.
        patch_size: Patch size used to compute the output grid.
        output_4d: If True return (B, C, H, W); otherwise return (B, N, C).

    Returns:
        A callable that maps pixel tensors to patch embeddings.
    """
    conv = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def _fn(x: Tensor) -> Tensor:
        c = conv.to(x.device, x.dtype)
        out = c(x)
        if not output_4d:
            out = out.flatten(2).transpose(1, 2)
        return out

    return _fn


class TestNaFlex(BaseTester):
    @pytest.fixture
    def model(self) -> NaFlex:
        """Create a NaFlex model with a Conv2d-based patch embedding for testing.

        Returns:
            NaFlex instance with 14x14 = 196 position embeddings and embed_dim=768.
        """
        patch_fn = _make_mock_patch_fn(embed_dim=768, patch_size=16, output_4d=True)
        position_embedding = torch.randn(196, 768)
        return NaFlex(patch_embedding_fcn=patch_fn, position_embedding=position_embedding)

    # ------------------------------------------------------------------
    # Smoke tests
    # ------------------------------------------------------------------

    def test_smoke(self, model: NaFlex, device: torch.device, dtype: torch.dtype) -> None:
        """Test basic forward pass with standard 224x224 input."""
        model = model.to(device, dtype)
        x = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)
        out = model(x)
        assert isinstance(out, Tensor)
        assert out.shape == (1, 196, 768)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_smoke_batch_sizes(self, model: NaFlex, device: torch.device, dtype: torch.dtype, batch_size: int) -> None:
        """Test forward pass across different batch sizes."""
        model = model.to(device, dtype)
        x = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)
        out = model(x)
        assert out.shape == (batch_size, 196, 768)

    def test_smoke_3d_patch_output(self, device: torch.device, dtype: torch.dtype) -> None:
        """Test forward pass when patch embedding returns 3D tensor (B, N, C)."""
        patch_fn = _make_mock_patch_fn(embed_dim=768, patch_size=16, output_4d=False)
        pos_embed = torch.randn(196, 768)
        model = NaFlex(patch_fn, pos_embed).to(device, dtype)
        x = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)
        out = model(x)
        assert out.shape == (1, 196, 768)

    # ------------------------------------------------------------------
    # Exception tests
    # ------------------------------------------------------------------

    def test_exception(self, device: torch.device, dtype: torch.dtype) -> None:
        """Test that non-square position embeddings raise ValueError during interpolation."""

        def fake_patch_fcn(x: Tensor) -> Tensor:
            return torch.randn(1, 100, 768, device=device, dtype=dtype)

        bad_pos_embed = torch.randn(200, 768, device=device, dtype=dtype)
        wrapper = NaFlex(fake_patch_fcn, bad_pos_embed)
        x = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)

        with pytest.raises(ValueError, match="Original positional embedding is not a square grid"):
            wrapper(x)

    def test_exception_non_square_primes(self, device: torch.device, dtype: torch.dtype) -> None:
        """Test ValueError for a prime number of position embeddings which cannot be a square grid."""

        def fake_patch_fcn(x: Tensor) -> Tensor:
            return torch.randn(1, 50, 768, device=device, dtype=dtype)

        bad_pos_embed = torch.randn(197, 768, device=device, dtype=dtype)
        wrapper = NaFlex(fake_patch_fcn, bad_pos_embed)
        x = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)

        with pytest.raises(ValueError, match="Original positional embedding is not a square grid"):
            wrapper(x)

    # ------------------------------------------------------------------
    # Cardinality tests
    # ------------------------------------------------------------------

    def test_cardinality(self, model: NaFlex, device: torch.device, dtype: torch.dtype) -> None:
        """Test output shape with non-square 224x320 input (14x20 = 280 patches)."""
        model = model.to(device, dtype)
        x = torch.randn(1, 3, 224, 320, device=device, dtype=dtype)
        out = model(x)
        assert out.shape == (1, 280, 768)

    @pytest.mark.parametrize(
        "input_size,expected_patches",
        [
            ((224, 224), 196),
            ((224, 320), 280),
            ((320, 320), 400),
            ((160, 160), 100),
            ((128, 256), 128),
        ],
    )
    def test_cardinality_various_resolutions(
        self, device: torch.device, dtype: torch.dtype, input_size: tuple[int, int], expected_patches: int
    ) -> None:
        """Test output shapes for several input resolutions requiring interpolation."""
        patch_fn = _make_mock_patch_fn(embed_dim=768, patch_size=16, output_4d=True)
        pos_embed = torch.randn(196, 768)
        model = NaFlex(patch_fn, pos_embed).to(device, dtype)
        h, w = input_size
        x = torch.randn(1, 3, h, w, device=device, dtype=dtype)
        out = model(x)
        assert out.shape == (1, expected_patches, 768)

    @pytest.mark.parametrize("embed_dim", [256, 512, 768])
    def test_cardinality_embed_dims(self, device: torch.device, dtype: torch.dtype, embed_dim: int) -> None:
        """Test that the embedding dimension is preserved in the output."""
        patch_fn = _make_mock_patch_fn(embed_dim=embed_dim, patch_size=16, output_4d=True)
        pos_embed = torch.randn(196, embed_dim)
        model = NaFlex(patch_fn, pos_embed).to(device, dtype)
        x = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)
        out = model(x)
        assert out.shape == (1, 196, embed_dim)

    # ------------------------------------------------------------------
    # Flexible resolution / interpolation tests
    # ------------------------------------------------------------------

    def test_interpolation(self, device: torch.device, dtype: torch.dtype) -> None:
        """Test positional embedding interpolation for a 448x448 input."""
        patch_fn = _make_mock_patch_fn(embed_dim=768, patch_size=16, output_4d=True)
        pos_embed = torch.randn(196, 768)
        model = NaFlex(patch_fn, pos_embed).to(device, dtype)
        x = torch.randn(1, 3, 448, 448, device=device, dtype=dtype)
        out = model(x)
        assert out.shape == (1, 784, 768)

    def test_no_interpolation_matching_grid(self, device: torch.device, dtype: torch.dtype) -> None:
        """Test that no interpolation occurs when input matches the original grid size."""
        patch_fn = _make_mock_patch_fn(embed_dim=768, patch_size=16, output_4d=True)
        pos_embed = torch.randn(196, 768)
        model = NaFlex(patch_fn, pos_embed).to(device, dtype)
        x = torch.randn(2, 3, 224, 224, device=device, dtype=dtype)
        out = model(x)
        # Position embedding should be added without interpolation
        assert out.shape == (2, 196, 768)

    def test_interpolation_non_square_input(self, device: torch.device, dtype: torch.dtype) -> None:
        """Test interpolation when the input produces a non-square patch grid."""
        patch_fn = _make_mock_patch_fn(embed_dim=768, patch_size=16, output_4d=True)
        pos_embed = torch.randn(196, 768)
        model = NaFlex(patch_fn, pos_embed).to(device, dtype)
        x = torch.randn(1, 3, 224, 448, device=device, dtype=dtype)
        out = model(x)
        # 14 x 28 = 392 patches
        assert out.shape == (1, 392, 768)

    def test_interpolation_downscale(self, device: torch.device, dtype: torch.dtype) -> None:
        """Test interpolation when input resolution is smaller than original grid."""
        patch_fn = _make_mock_patch_fn(embed_dim=768, patch_size=16, output_4d=True)
        pos_embed = torch.randn(196, 768)
        model = NaFlex(patch_fn, pos_embed).to(device, dtype)
        x = torch.randn(1, 3, 112, 112, device=device, dtype=dtype)
        out = model(x)
        # 7 x 7 = 49 patches
        assert out.shape == (1, 49, 768)

    # ------------------------------------------------------------------
    # Batch consistency tests
    # ------------------------------------------------------------------

    def test_batch_consistency(self, device: torch.device, dtype: torch.dtype) -> None:
        """Test that batch processing produces the same result as individual processing."""
        patch_fn = _make_mock_patch_fn(embed_dim=768, patch_size=16, output_4d=True)
        pos_embed = torch.randn(196, 768)
        model = NaFlex(patch_fn, pos_embed).to(device, dtype)

        torch.manual_seed(42)
        x = torch.randn(3, 3, 224, 224, device=device, dtype=dtype)

        out_batch = model(x)

        for i in range(3):
            out_single = model(x[i : i + 1])
            self.assert_close(out_batch[i : i + 1], out_single)

    # ------------------------------------------------------------------
    # Position embedding as buffer tests
    # ------------------------------------------------------------------

    def test_position_embedding_registered_as_buffer(self, device: torch.device, dtype: torch.dtype) -> None:
        """Test that position_embedding is a registered buffer and moves with the model."""
        patch_fn = _make_mock_patch_fn(embed_dim=768, patch_size=16, output_4d=True)
        pos_embed = torch.randn(196, 768)
        model = NaFlex(patch_fn, pos_embed)
        assert "position_embedding" in dict(model.named_buffers())
        model = model.to(device, dtype)
        assert model.position_embedding.device == device
        assert model.position_embedding.dtype == dtype

    def test_state_dict_contains_position_embedding(self) -> None:
        """Test that position_embedding appears in the state dict."""
        patch_fn = _make_mock_patch_fn(embed_dim=768, patch_size=16, output_4d=True)
        pos_embed = torch.randn(196, 768)
        model = NaFlex(patch_fn, pos_embed)
        assert "position_embedding" in model.state_dict()

    # ------------------------------------------------------------------
    # Different grid sizes for position embeddings
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("grid_size", [7, 14, 16])
    def test_different_position_grid_sizes(self, device: torch.device, dtype: torch.dtype, grid_size: int) -> None:
        """Test NaFlex with position embeddings of various square grid sizes."""
        num_pos = grid_size * grid_size
        embed_dim = 256
        patch_fn = _make_mock_patch_fn(embed_dim=embed_dim, patch_size=16, output_4d=True)
        pos_embed = torch.randn(num_pos, embed_dim)
        model = NaFlex(patch_fn, pos_embed).to(device, dtype)
        x = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)
        out = model(x)
        assert out.shape == (1, 196, embed_dim)

    # ------------------------------------------------------------------
    # Gradient checks
    # ------------------------------------------------------------------

    def test_gradcheck(self, device: torch.device) -> None:
        """Test gradient correctness with torch.autograd.gradcheck."""
        embed_dim = 32
        patch_size = 16
        conv = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        pos_embed = torch.randn(4, embed_dim)
        model = NaFlex(conv, pos_embed).to(device, torch.float64)
        x = torch.randn(1, 3, 32, 32, device=device, dtype=torch.float64, requires_grad=True)

        def func(pixel_vals: Tensor) -> Tensor:
            return model(pixel_vals)

        self.gradcheck(func, (x,))

    def test_gradcheck_interpolation(self, device: torch.device) -> None:
        """Test gradient correctness when positional interpolation is triggered."""
        embed_dim = 32
        patch_size = 16
        conv = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        # 2x2 = 4 positions, but input will produce a 4x4 grid -> interpolation
        pos_embed = torch.randn(4, embed_dim)
        model = NaFlex(conv, pos_embed).to(device, torch.float64)
        x = torch.randn(1, 3, 64, 64, device=device, dtype=torch.float64, requires_grad=True)

        def func(pixel_vals: Tensor) -> Tensor:
            return model(pixel_vals)

        self.gradcheck(func, (x,))

    # ------------------------------------------------------------------
    # torch.compile (dynamo) compatibility
    # ------------------------------------------------------------------

    def test_dynamo(self, device: torch.device, dtype: torch.dtype, torch_optimizer) -> None:
        """Test NaFlex compatibility with torch.compile."""
        embed_dim = 64
        patch_size = 16
        conv = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        pos_embed = torch.randn(4, embed_dim)
        model = NaFlex(conv, pos_embed).to(device, dtype).eval()
        x = torch.randn(1, 3, 32, 32, device=device, dtype=dtype)

        op = model.forward
        op_optimized = torch_optimizer(op)

        expected = op(x)
        actual = op_optimized(x)
        self.assert_close(actual, expected)

    def test_dynamo_with_interpolation(self, device: torch.device, dtype: torch.dtype, torch_optimizer) -> None:
        """Test torch.compile compatibility when interpolation is required."""
        embed_dim = 64
        patch_size = 16
        conv = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        pos_embed = torch.randn(4, embed_dim)
        model = NaFlex(conv, pos_embed).to(device, dtype).eval()
        x = torch.randn(1, 3, 64, 64, device=device, dtype=dtype)

        op = model.forward
        op_optimized = torch_optimizer(op)

        expected = op(x)
        actual = op_optimized(x)
        self.assert_close(actual, expected)
