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
import io
import math
import warnings

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from kornia.core._compat import torch_version_lt
from kornia.core.exceptions import BaseError
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

    @pytest.mark.parametrize(("height", "width"), [(128, 512), (192, 336), (40, 63), (230, 230)])
    def test_bnc_patch_output_with_patch_size(
        self, device: torch.device, dtype: torch.dtype, height: int, width: int
    ) -> None:
        # A (B, N, C) patch output carries no grid; patch_size=16 gives (H // 16, W // 16): 8 x 32 for 128 x 512 (the
        # 256 tokens a square reading takes for 16 x 16), 12 x 21 for 192 x 336, and 2 x 3 and 14 x 14 for sides that
        # are not multiples of 16 (the conv drops the remainder). The same as the (B, C, h, w) output and the reference.
        table = torch.randn(256, _CHANNELS, device=device, dtype=dtype)
        images = torch.zeros(3, 3, height, width, device=device, dtype=dtype)
        out = NaFlex(_zero_tokens, table, patch_size=16)(images)
        self.assert_close(out, NaFlex(_zero_patches, table)(images))
        self.assert_close(out, _reference_positions(table, height // 16, width // 16).expand(3, -1, -1))

    def test_bnc_patch_output_with_a_rectangular_patch_size(self, device: torch.device, dtype: torch.dtype) -> None:
        # (16, 32) patches of a 256 x 512 image: a 16 x 16 grid (read with the axes swapped it would be 8 x 32, which
        # has as many cells), equal to the conv's (B, C, h, w) output with the table added as is.
        conv = torch.nn.Conv2d(3, _CHANNELS, (16, 32), stride=(16, 32)).to(device, dtype)
        table = torch.randn(256, _CHANNELS, device=device, dtype=dtype)
        images = torch.rand(2, 3, 256, 512, device=device, dtype=dtype)
        with torch.no_grad():
            tokens = conv(images).flatten(2).transpose(1, 2)
            out = NaFlex(lambda x: conv(x).flatten(2).transpose(1, 2), table, patch_size=(16, 32))(images)
            self.assert_close(out, tokens + table)
            self.assert_close(out, NaFlex(conv, table)(images))

    def test_bnc_patch_size_whose_grid_does_not_match_raises(self, device: torch.device, dtype: torch.dtype) -> None:
        # patch_size=16 gives a 64 x 64 image a 4 x 4 grid, 16 cells, but the patch embedding returns 7 tokens.
        table = torch.randn(256, _CHANNELS, device=device, dtype=dtype)
        naflex = NaFlex(lambda x: x.new_zeros(x.shape[0], 7, _CHANNELS), table, patch_size=16)
        with pytest.raises(BaseError, match=r"4 x 4 grid of 16 patches, but the patch embedding returned 7"):
            naflex(torch.zeros(3, 3, 64, 64, device=device, dtype=dtype))

    @pytest.mark.parametrize(("height", "width", "num_tokens"), [(64, 64, 7), (192, 336, 252)])
    def test_bnc_without_patch_size_needs_a_square_count(
        self, device: torch.device, dtype: torch.dtype, height: int, width: int, num_tokens: int
    ) -> None:
        table = torch.randn(256, _CHANNELS, device=device, dtype=dtype)
        naflex = NaFlex(lambda x: x.new_zeros(x.shape[0], num_tokens, _CHANNELS), table)
        with pytest.raises(BaseError, match=f"{num_tokens} patch embeddings are not a square number. Pass patch_size"):
            naflex(torch.zeros(3, 3, height, width, device=device, dtype=dtype))

    @pytest.mark.parametrize(
        ("height", "width", "kernel", "stride", "side"),
        [
            (20, 20, 3, 2, 9),
            (19, 20, 3, 2, 9),
            (20, 36, 3, (2, 4), 9),
            (256, 512, (16, 32), (16, 32), 16),
            (64, 144, (10, 24), (10, 24), 6),
        ],
    )
    def test_bnc_without_patch_size_reads_a_square_grid(
        self, device: torch.device, dtype: torch.dtype, height: int, width: int, kernel, stride, side: int
    ) -> None:
        # Without patch_size, a square number of tokens is read as the square grid: right for these convolutions,
        # whose grids are side x side. A side x side table is added as is, a 4 x 4 one resized by plain bilinear
        # interpolation (no axis shrinks).
        conv = torch.nn.Conv2d(3, _CHANNELS, kernel, stride=stride).to(device, dtype)
        images = torch.rand(2, 3, height, width, device=device, dtype=dtype)
        with torch.no_grad():
            tokens = conv(images).flatten(2).transpose(1, 2)
            assert tokens.shape[1] == side * side
            own = torch.randn(side * side, _CHANNELS, device=device, dtype=dtype)
            self.assert_close(NaFlex(lambda x: conv(x).flatten(2).transpose(1, 2), own)(images), tokens + own)
            small = torch.randn(16, _CHANNELS, device=device, dtype=dtype)
            grid = small.reshape(4, 4, -1).permute(2, 0, 1).unsqueeze(0).to("cpu", torch.float32)
            resized = F.interpolate(grid, size=(side, side), mode="bilinear", align_corners=False)
            resized = resized.reshape(-1, side * side).transpose(0, 1).to(device, dtype)
            out = NaFlex(lambda x: conv(x).flatten(2).transpose(1, 2), small)(images)
            self.assert_close(out, tokens + resized)

    @pytest.mark.parametrize(("height", "width"), [(192, 336), (256, 128)])
    def test_downsampling_is_antialiased(
        self, device: torch.device, dtype: torch.dtype, height: int, width: int
    ) -> None:
        # 12 x 21 downsamples the 16 x 16 table along the rows, 16 x 8 along the columns.
        table = torch.randn(256, _CHANNELS, device=device, dtype=dtype)
        out = NaFlex(_zero_patches, table)(torch.zeros(3, 3, height, width, device=device, dtype=dtype))
        self.assert_close(out, _reference_positions(table, height // 16, width // 16).expand(3, -1, -1))

    @pytest.mark.parametrize(("height", "width"), [(384, 512), (256, 320)])
    def test_upsampling_matches_the_reference(
        self, device: torch.device, dtype: torch.dtype, height: int, width: int
    ) -> None:
        # 24 x 32 upsamples both axes of the 16 x 16 table, 16 x 20 the columns only. NaFlex resizes without
        # antialiasing here; transformers' antialias=True changes nothing when no axis shrinks (2.4e-6 apart in
        # float32), so the antialiased reference still holds.
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
        # The same for a (B, N, C) output, whose grid is then never read: the 6 tokens of a 48 x 47 image (a 3 x 2
        # grid that the aspect reading would transpose) take a 6-row table as is.
        table = torch.randn(6, _CHANNELS, device=device, dtype=dtype)
        out = NaFlex(_zero_tokens, table)(torch.zeros(3, 3, 48, 47, device=device, dtype=dtype))
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

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize(("height", "width"), [(384, 512), (256, 320)])
    def test_naflex_legacy_onnx_export_of_an_upsampled_table(self, height: int, width: int) -> None:
        pytest.importorskip("onnx")
        ort = pytest.importorskip("onnxruntime")
        # A 24 x 32 grid upsamples the 16 x 16 table on both axes and a 16 x 20 grid keeps the rows and upsamples the
        # columns: the plain bilinear resize, which the legacy TorchScript exporter supports (it has no symbolic for
        # the antialiased aten::_upsample_bilinear2d_aa).
        naflex = NaFlex(torch.nn.Conv2d(3, _CHANNELS, 16, 16), torch.randn(256, _CHANNELS)).eval()
        images = torch.rand(3, 3, height, width)
        buffer = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                naflex,
                (images,),
                buffer,  # ty: ignore[invalid-argument-type]  # The legacy exporter accepts BytesIO.
                input_names=["images"],
                opset_version=17,
                dynamo=False,
            )
        session = ort.InferenceSession(buffer.getvalue(), providers=["CPUExecutionProvider"])
        actual = torch.from_numpy(session.run(None, {"images": images.numpy()})[0])
        with torch.no_grad():
            expected = naflex(images)
        self.assert_close(actual, expected, rtol=0.0, atol=1e-5)

    @pytest.mark.device_agnostic
    def test_naflex_onnx_dynamic_size_exported_from_a_smaller_grid(self) -> None:
        pytest.importorskip("onnx")
        ort = pytest.importorskip("onnxruntime")
        pytest.importorskip("onnxscript")
        if torch_version_lt(2, 14, 0):
            # torch.export reports NaFlex's grid-size comparison as a constraint violation of the dynamic H and W;
            # torch.onnx.export of torch 2.14 exports anyway, and that of torch 2.5.1 raises TorchExportError.
            pytest.skip("checked on torch 2.14: the dynamo exporter of torch 2.5.1 refuses this dynamic export")
        # The documented recipe: export with dynamic H and W from an image whose 8 x 8 grid is smaller than the
        # 16 x 16 table along both axes, so the graph resizes with antialiasing at every size. That is right for a
        # smaller grid (4 x 6) and, since antialiasing changes nothing when upsampling, for a larger one (24 x 32).
        naflex = NaFlex(torch.nn.Conv2d(3, _CHANNELS, 16, 16), torch.randn(256, _CHANNELS)).eval()
        size = {2: torch.export.Dim("height", min=16, max=1024), 3: torch.export.Dim("width", min=16, max=1024)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            program = torch.onnx.export(
                naflex, (torch.rand(2, 3, 128, 128),), dynamo=True, opset_version=18, dynamic_shapes=(size,)
            )
        session = ort.InferenceSession(program.model_proto.SerializeToString(), providers=["CPUExecutionProvider"])
        for height, width in ((64, 96), (384, 512)):
            images = torch.rand(2, 3, height, width)
            actual = torch.from_numpy(session.run(None, {session.get_inputs()[0].name: images.numpy()})[0])
            with torch.no_grad():
                expected = naflex(images)
            self.assert_close(actual, expected, rtol=0.0, atol=1e-5)

    @pytest.mark.device_agnostic
    def test_float64_table_is_resized_in_float64(self) -> None:
        # transformers resizes every CPU table in float32; a float64 table keeps float64 here, so it matches the
        # float64 resize to rounding (a float32 resize is about 1e-7 off). 12 x 21 shrinks the rows (antialiased),
        # 24 x 32 shrinks no axis (plain bilinear).
        table = torch.randn(256, _CHANNELS, dtype=torch.float64)
        grid = table.reshape(16, 16, -1).permute(2, 0, 1).unsqueeze(0)
        for (height, width), antialias in (((192, 336), True), ((384, 512), False)):
            h, w = height // 16, width // 16
            out = NaFlex(_zero_patches, table)(torch.zeros(3, 3, height, width, dtype=torch.float64))
            resized = F.interpolate(grid, size=(h, w), mode="bilinear", align_corners=False, antialias=antialias)
            expected = resized.reshape(-1, h * w).transpose(0, 1).expand(3, -1, -1)
            self.assert_close(out, expected, rtol=0.0, atol=1e-12)

    # 192 x 336: a 12 x 21 grid, antialiased (rows shrink); 384 x 512: 24 x 32, plain bilinear.
    @pytest.mark.parametrize(("height", "width"), [(192, 336), (384, 512)])
    def test_dynamo_fullgraph_naflex(
        self, device: torch.device, dtype: torch.dtype, torch_optimizer, height: int, width: int
    ) -> None:
        conv = torch.nn.Conv2d(3, _CHANNELS, 16, 16).to(device, dtype)
        naflex = NaFlex(conv, torch.randn(256, _CHANNELS, device=device, dtype=dtype)).eval()
        images = torch.rand(3, 3, height, width, device=device, dtype=dtype)

        naflex_optimized = torch_optimizer(naflex, fullgraph=True)

        with torch.no_grad():
            self.assert_close(naflex_optimized(images), naflex(images))
