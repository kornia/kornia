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

import kornia

from testing.base import assert_close


def test_create_meshgrid(device, dtype):
    height, width = 4, 6
    normalized_coordinates = False

    # create the meshgrid and verify shape
    grid = kornia.geometry.create_meshgrid(height, width, normalized_coordinates, device=device, dtype=dtype)

    assert grid.device == device
    assert grid.dtype == dtype
    assert grid.shape == (1, height, width, 2)

    # check grid corner values
    assert tuple(grid[0, 0, 0].cpu().numpy()) == (0.0, 0.0)
    assert tuple(grid[0, height - 1, width - 1].cpu().numpy()) == (width - 1, height - 1)


def test_normalize_pixel_grid(device, dtype):
    if device.type == "cuda" and dtype == torch.float16:
        pytest.skip('"inverse_cuda" not implemented for "Half"')

    # generate input data
    height, width = 2, 4

    # create points grid
    grid_norm = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=True, device=device, dtype=dtype)

    assert grid_norm.device == device
    assert grid_norm.dtype == dtype
    grid_norm = torch.unsqueeze(grid_norm, dim=0)

    grid_pix = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=False, device=device, dtype=dtype)

    assert grid_pix.device == device
    assert grid_pix.dtype == dtype
    grid_pix = torch.unsqueeze(grid_pix, dim=0)

    # grid from pixel space to normalized
    norm_trans_pix = kornia.geometry.conversions.normal_transform_pixel(
        height, width, device=device, dtype=dtype
    )  # 1x3x3
    pix_trans_norm = torch.inverse(norm_trans_pix)  # 1x3x3
    # transform grids
    grid_pix_to_norm = kornia.geometry.linalg.transform_points(norm_trans_pix, grid_pix)
    grid_norm_to_pix = kornia.geometry.linalg.transform_points(pix_trans_norm, grid_norm)
    assert_close(grid_pix, grid_norm_to_pix)
    assert_close(grid_norm, grid_pix_to_norm)


def test_create_meshgrid3d(device, dtype):
    depth, height, width = 5, 4, 6
    normalized_coordinates = False

    # create the meshgrid and verify shape
    grid = kornia.geometry.create_meshgrid3d(depth, height, width, normalized_coordinates, device=device, dtype=dtype)

    assert grid.device == device
    assert grid.dtype == dtype
    assert grid.shape == (1, depth, height, width, 3)

    # check grid corner values
    assert tuple(grid[0, 0, 0, 0].cpu().numpy()) == (0.0, 0.0, 0.0)
    assert tuple(grid[0, depth - 1, height - 1, width - 1].cpu().numpy()) == (depth - 1, width - 1, height - 1)


class TestNormalTransformPixelDtype:
    """`normal_transform_pixel` must refuse a dtype that cannot hold its scales.

    The matrix scales by ``2 / (size - 1)``. That is fractional for any dimension
    wider than three pixels, so an integer dtype truncates it to zero and the
    result collapses the whole image onto a single point.
    """

    @pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.uint8, torch.bool])
    def test_raises_on_non_fractional_dtype(self, dtype):
        with pytest.raises(Exception, match="floating point or complex"):
            kornia.geometry.conversions.normal_transform_pixel(4, 5, dtype=dtype)
        with pytest.raises(Exception, match="floating point or complex"):
            kornia.geometry.conversions.normal_transform_pixel3d(2, 4, 5, dtype=dtype)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    def test_representable_dtypes_still_map_corners_to_the_unit_square(self, dtype):
        # The oracle is the documented contract: the transform takes pixel
        # coordinates to [-1, 1]. Corner (0, 0) must land on (-1, -1) and corner
        # (W-1, H-1) on (1, 1). This is what an integer dtype cannot satisfy, and
        # it holds independently of how the matrix is built.
        height, width = 4, 5
        mat = kornia.geometry.conversions.normal_transform_pixel(height, width, dtype=dtype)
        corners = torch.tensor([[[0.0, 0.0], [width - 1.0, height - 1.0]]], dtype=dtype)

        mapped = kornia.geometry.linalg.transform_points(mat, corners)

        expected = torch.tensor([[[-1.0, -1.0], [1.0, 1.0]]], dtype=dtype)
        assert_close(mapped, expected, atol=1e-3, rtol=1e-3)

    def test_float_output_is_unchanged(self):
        # Guards against the check altering the value path it is meant to leave
        # alone. Literals are the pre-change output of `normal_transform_pixel`.
        mat = kornia.geometry.conversions.normal_transform_pixel(4, 5, dtype=torch.float64)
        expected = torch.tensor([[[0.5, 0.0, -1.0], [0.0, 2.0 / 3.0, -1.0], [0.0, 0.0, 1.0]]], dtype=torch.float64)
        assert_close(mat, expected)

    def test_complex_dtype_is_still_accepted(self):
        # Complex can represent the scales exactly, and rejecting it would turn a
        # currently-working call into an error. Only the truncating dtypes raise.
        mat = kornia.geometry.conversions.normal_transform_pixel(4, 5, dtype=torch.complex64)
        assert mat.is_complex()
        assert torch.det(mat.to(torch.complex128)).abs().item() > 0.0

    def test_default_dtype_unaffected(self):
        mat = kornia.geometry.conversions.normal_transform_pixel(4, 5)
        assert mat.dtype == torch.get_default_dtype()
