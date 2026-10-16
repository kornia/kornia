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
import math

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from kornia.core._compat import torch_version_lt
from kornia.models.processors.naflex import NaFlex

from testing.base import BaseTester

_CHANNELS = 8


def _zero_patches(x: Tensor) -> Tensor:
    """A patch embedding of patch size 16 that returns zeros, so ``NaFlex``'s output is the position table it adds."""
    batch_size, _, height, width = x.shape
    return x.new_zeros(batch_size, _CHANNELS, height // 16, width // 16)


def _zero_tokens(x: Tensor) -> Tensor:
    """The same zeros as a ``(B, N, C)`` token sequence in raster order."""
    return _zero_patches(x).flatten(2).transpose(1, 2)


def _reference_positions(table: Tensor, height: int, width: int) -> Tensor:
    """The position table resized to a ``height x width`` grid, as transformers resizes it.

    ``Siglip2VisionEmbeddings.resize_positional_embeddings`` (transformers v4.51.0, modeling_siglip2.py): the
    ``(side, side, C)`` table as ``(1, C, side, side)``, ``F.interpolate(mode="bilinear", align_corners=False,
    antialias=True)`` in float32 on the CPU, cast back to the table's dtype; returned as ``(height * width, C)``.
    """
    side = math.isqrt(table.shape[0])
    grid = table.reshape(side, side, -1).permute(2, 0, 1).unsqueeze(0).to("cpu", torch.float32)
    resized = F.interpolate(grid, size=(height, width), mode="bilinear", align_corners=False, antialias=True)
    return resized.reshape(-1, height * width).transpose(0, 1).to(table.device, table.dtype)


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
        model = model.to(device)
        input_data = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)
        out = model(input_data)
        assert isinstance(out, Tensor)
        assert out.shape == (1, 196, 768)

    def test_cardinality(self, model: NaFlex, device: torch.device, dtype: torch.dtype) -> None:
        """Test output cardinality with non-square input resolution.

        For 224x320 input with 16x16 patches, expect 14x20=280 patches.
        """
        model = model.to(device)
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
        model = NaFlex(mock_patch_embedding_dynamic, position_embedding).to(device)
        input_448 = torch.randn(1, 3, 448, 448, device=device, dtype=dtype)
        out = model(input_448)
        assert out.shape == (1, 784, 768)

    def test_non_square_grid_with_matching_count(self, device: torch.device, dtype: torch.dtype) -> None:
        # 128 x 512 at patch 16 is an 8 x 32 grid: 256 patches, as many as the 16 x 16 table has rows, but laid out
        # differently, so the table is resized to 8 x 32 rather than added in raster order.
        table = torch.randn(256, _CHANNELS, device=device, dtype=dtype)
        out = NaFlex(_zero_patches, table)(torch.zeros(3, 3, 128, 512, device=device, dtype=dtype))
        self.assert_close(out, _reference_positions(table, 8, 32).expand(3, -1, -1))

    @pytest.mark.parametrize(("height", "width"), [(128, 512), (192, 336)])
    def test_bnc_patch_output_uses_the_image_aspect(
        self, device: torch.device, dtype: torch.dtype, height: int, width: int
    ) -> None:
        # A (B, N, C) patch output carries no grid shape: it is the one of the image's aspect ratio with N cells,
        # 8 x 32 for 128 x 512 and 12 x 21 for 192 x 336 at patch 16.
        table = torch.randn(256, _CHANNELS, device=device, dtype=dtype)
        images = torch.zeros(3, 3, height, width, device=device, dtype=dtype)
        out = NaFlex(_zero_tokens, table)(images)
        self.assert_close(out, NaFlex(_zero_patches, table)(images))
        self.assert_close(out, _reference_positions(table, height // 16, width // 16).expand(3, -1, -1))

    def test_bnc_patch_count_that_does_not_factor_raises(self, device: torch.device, dtype: torch.dtype) -> None:
        # 7 tokens of a square image: round(sqrt(7)) = 3 rows of 7 // 3 = 2 columns hold 6, not 7.
        table = torch.randn(256, _CHANNELS, device=device, dtype=dtype)
        naflex = NaFlex(lambda x: x.new_zeros(x.shape[0], 7, _CHANNELS), table)
        with pytest.raises(ValueError, match="7 patch embeddings"):
            naflex(torch.zeros(3, 3, 64, 64, device=device, dtype=dtype))

    @pytest.mark.parametrize(("height", "width"), [(192, 336), (256, 128)])
    def test_downsampling_is_antialiased(
        self, device: torch.device, dtype: torch.dtype, height: int, width: int
    ) -> None:
        # 12 x 21 downsamples the 16 x 16 table along the rows, 16 x 8 along the columns.
        table = torch.randn(256, _CHANNELS, device=device, dtype=dtype)
        out = NaFlex(_zero_patches, table)(torch.zeros(3, 3, height, width, device=device, dtype=dtype))
        self.assert_close(out, _reference_positions(table, height // 16, width // 16).expand(3, -1, -1))

    def test_non_square_table_with_matching_count_is_added_as_is(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        # A table that is not a square grid cannot be resized, but one with exactly as many rows as the grid has
        # patches is added in raster order, as for a square table on its own grid.
        table = torch.randn(280, _CHANNELS, device=device, dtype=dtype)
        out = NaFlex(_zero_patches, table)(torch.zeros(3, 3, 224, 320, device=device, dtype=dtype))
        self.assert_close(out, table.expand(3, -1, -1))

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("half", [torch.float16, torch.bfloat16])
    def test_half_on_cpu_interpolates_in_float32(self, half: torch.dtype) -> None:
        # CPU has no float16/bfloat16 antialiased bilinear kernel ("compute_index_ranges_weights" not implemented:
        # NotImplementedError on torch 2.14, RuntimeError on 2.5.1), so the table is resized in float32 and cast back.
        table = torch.randn(256, _CHANNELS, dtype=half)
        out = NaFlex(_zero_patches, table)(torch.zeros(3, 3, 192, 336, dtype=half))
        assert out.dtype == half
        self.assert_close(out, _reference_positions(table, 12, 21).expand(3, -1, -1), rtol=0.0, atol=0.0)

    def test_parameter_stays_a_parameter(self, device: torch.device, dtype: torch.dtype) -> None:
        table = torch.nn.Parameter(torch.randn(256, _CHANNELS, device=device, dtype=dtype))
        naflex = NaFlex(_zero_patches, table)
        assert [(name, param is table) for name, param in naflex.named_parameters()] == [("position_embedding", True)]
        assert list(naflex.named_buffers()) == []
        naflex(torch.zeros(3, 3, 192, 336, device=device, dtype=dtype)).sum().backward()
        assert table.grad is not None
        assert table.grad.abs().sum() > 0

        plain = torch.randn(256, _CHANNELS, device=device, dtype=dtype)
        buffered = NaFlex(_zero_patches, plain)
        assert list(buffered.named_parameters()) == []
        assert [(name, buffer is plain) for name, buffer in buffered.named_buffers()] == [("position_embedding", True)]

    @pytest.mark.device_agnostic
    def test_naflex_onnx_matches_eager(self, tmp_path) -> None:
        pytest.importorskip("onnx")
        ort = pytest.importorskip("onnxruntime")
        pytest.importorskip("onnxscript")
        if torch_version_lt(2, 5, 0):
            pytest.skip("the dynamo ONNX exporter needs torch >= 2.5")
        # The legacy TorchScript exporter has no symbolic for aten::_upsample_bilinear2d_aa; the dynamo exporter
        # exports the antialiased resize. 12 x 21 downsamples the 16 x 16 table, so the antialias path runs.
        naflex = NaFlex(torch.nn.Conv2d(3, _CHANNELS, 16, 16), torch.randn(256, _CHANNELS)).eval()
        images = torch.rand(3, 3, 192, 336)
        path = tmp_path / "naflex.onnx"
        torch.onnx.export(naflex, (images,), str(path), dynamo=True, opset_version=18, input_names=["images"])
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        actual = torch.from_numpy(session.run(None, {"images": images.numpy()})[0])
        with torch.no_grad():
            expected = naflex(images)
        self.assert_close(actual, expected, rtol=0.0, atol=1e-5)

    def test_dynamo_fullgraph_naflex(self, device: torch.device, dtype: torch.dtype, torch_optimizer) -> None:
        conv = torch.nn.Conv2d(3, _CHANNELS, 16, 16).to(device, dtype)
        naflex = NaFlex(conv, torch.randn(256, _CHANNELS, device=device, dtype=dtype)).eval()
        images = torch.rand(3, 3, 192, 336, device=device, dtype=dtype)

        naflex_optimized = torch_optimizer(naflex, fullgraph=True)

        with torch.no_grad():
            self.assert_close(naflex_optimized(images), naflex(images))
