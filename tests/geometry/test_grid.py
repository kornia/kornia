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

from testing.base import DYNAMO_UNAVAILABLE_REASON, assert_close, dynamo_is_available


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


@pytest.mark.parametrize(("height", "width"), [(1, 4), (4, 1), (1, 1)])
def test_normalized_meshgrid_singleton_axis_is_centered(height, width, device, dtype):
    grid = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=True, device=device, dtype=dtype)

    if height == 1:
        assert_close(grid[..., 1], torch.zeros_like(grid[..., 1]), atol=0.0, rtol=0.0)
    if width == 1:
        assert_close(grid[..., 0], torch.zeros_like(grid[..., 0]), atol=0.0, rtol=0.0)


@pytest.mark.parametrize(("height", "width"), [(1, 4), (4, 1), (1, 1), (4, 6)])
@pytest.mark.parametrize("grid_dtype", [torch.int32, torch.int64])
def test_normalized_meshgrid_integer_dtype_promotes_uniformly(height, width, grid_dtype, device):
    # An integer ``dtype`` makes the non-singleton branch promote to the default float dtype
    # (integer division), so the singleton branch must promote identically or torch.meshgrid
    # rejects the mixed pair. Eager and tracing take different code paths here; this pins that
    # they agree, and that a singleton axis is still centred once promoted.
    grid = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=True, device=device, dtype=grid_dtype)

    assert grid.dtype == torch.get_default_dtype()
    if height == 1:
        assert_close(grid[..., 1], torch.zeros_like(grid[..., 1]), atol=0.0, rtol=0.0)
    if width == 1:
        assert_close(grid[..., 0], torch.zeros_like(grid[..., 0]), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("grid_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(("trace_height", "runtime_height"), [(2, 1), (1, 2)])
def test_normalized_meshgrid_integer_trace_crosses_singleton_boundary(grid_dtype, trace_height, runtime_height, device):
    class IntegerMeshGrid(torch.nn.Module):
        def forward(self, image):
            return kornia.geometry.create_meshgrid(
                image.shape[-2],
                image.shape[-1],
                normalized_coordinates=True,
                device=image.device,
                dtype=grid_dtype,
            )

    example = torch.zeros(1, 1, trace_height, 4, device=device)
    runtime = torch.zeros(1, 1, runtime_height, 4, device=device)
    traced = torch.jit.trace(IntegerMeshGrid(), example)
    assert_close(traced(runtime), IntegerMeshGrid()(runtime), atol=0.0, rtol=0.0)


@pytest.mark.parametrize(("trace_height", "runtime_height"), [(2, 1), (1, 2)])
def test_normalized_meshgrid_trace_crosses_singleton_boundary(trace_height, runtime_height, device, dtype):
    class MeshGrid(torch.nn.Module):
        def forward(self, image):
            return kornia.geometry.create_meshgrid(
                image.shape[-2], image.shape[-1], normalized_coordinates=True, device=image.device, dtype=image.dtype
            )

    example = torch.zeros(1, 1, trace_height, 4, device=device, dtype=dtype)
    runtime = torch.zeros(1, 1, runtime_height, 4, device=device, dtype=dtype)
    traced = torch.jit.trace(MeshGrid(), example)
    assert_close(traced(runtime), MeshGrid()(runtime), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("is_3d", [False, True], ids=["2d", "3d"])
@pytest.mark.parametrize("size", [258, 300, 1000, 2050, 3000])
def test_normalized_meshgrid_trace_matches_eager_at_unrepresentable_sizes(is_3d, size, device, dtype):
    """A size whose predecessor does not fit the coordinate dtype must not round the divisor.

    ``bfloat16`` holds 299 only as 300, so casting ``size - 1`` into the coordinate dtype before
    dividing shifts every normalized coordinate -- up to 0.0078 in bfloat16, 0.00098 in float16.
    Eager divides by a Python ``int``, so the traced graph has to divide against the unrounded
    size too and round only the quotient. Every size the singleton-boundary tests above use is
    representable in all four dtypes, so none of them can catch this.

    The sizes here are the ones whose predecessor is *not* exactly representable in bfloat16
    (all five) or float16 (2050 and 3000); at float32 and float64 they all are, which pins the
    two paths as agreeing there rather than merely not being compared.
    """

    class MeshGrid(torch.nn.Module):
        def forward(self, image):
            if is_3d:
                return kornia.geometry.create_meshgrid3d(
                    image.shape[-3],
                    image.shape[-2],
                    image.shape[-1],
                    normalized_coordinates=True,
                    device=image.device,
                    dtype=image.dtype,
                )
            return kornia.geometry.create_meshgrid(
                image.shape[-2], image.shape[-1], normalized_coordinates=True, device=image.device, dtype=image.dtype
            )

    shape = (1, 1, 2, size, 4) if is_3d else (1, 1, size, 4)
    image = torch.zeros(*shape, device=device, dtype=dtype)
    traced = torch.jit.trace(MeshGrid(), image)
    assert_close(traced(image), MeshGrid()(image), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("normalized_coordinates", [False, True], ids=["pixel", "normalized"])
@pytest.mark.parametrize("is_3d", [False, True], ids=["2d", "3d"])
@pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
def test_meshgrid_export_crosses_singleton_boundary(is_3d, normalized_coordinates):
    class MeshGrid(torch.nn.Module):
        def forward(self, image):
            if is_3d:
                return kornia.geometry.create_meshgrid3d(
                    image.shape[-3],
                    image.shape[-2],
                    image.shape[-1],
                    normalized_coordinates=normalized_coordinates,
                    device=image.device,
                )
            return kornia.geometry.create_meshgrid(
                image.shape[-2],
                image.shape[-1],
                normalized_coordinates=normalized_coordinates,
                device=image.device,
            )

    image_shape = (1, 1, 2, 3, 4) if is_3d else (1, 1, 2, 4)
    example = torch.zeros(image_shape)
    exported = torch.export.export(
        MeshGrid(),
        (example,),
        dynamic_shapes=({2: torch.export.Dim("singleton_axis", min=1, max=8)},),
    ).module()

    for runtime_size in (1, 5):
        runtime = torch.zeros(*image_shape[:2], runtime_size, *image_shape[3:])
        assert_close(exported(runtime), MeshGrid()(runtime), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("is_3d", [False, True], ids=["2d", "3d"])
def test_pixel_meshgrid_default_dtype_matches_compile(is_3d, default_dtype):
    class MeshGrid(torch.nn.Module):
        def forward(self, image):
            if is_3d:
                return kornia.geometry.create_meshgrid3d(
                    image.shape[-3], image.shape[-2], image.shape[-1], False, device=image.device
                )
            return kornia.geometry.create_meshgrid(image.shape[-2], image.shape[-1], False, device=image.device)

    image = torch.zeros((1, 1, 2, 3, 4) if is_3d else (1, 1, 3, 4), dtype=torch.float32)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        expected = MeshGrid()(image)
        actual = torch.compile(MeshGrid(), fullgraph=True)(image)
    finally:
        torch.set_default_dtype(previous_dtype)

    assert actual.dtype == expected.dtype == default_dtype
    assert_close(actual, expected, atol=0.0, rtol=0.0)


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


@pytest.mark.parametrize(("depth", "height", "width", "axis"), [(1, 4, 6, 0), (5, 1, 6, 2), (5, 4, 1, 1)])
def test_normalized_meshgrid3d_singleton_axis_is_centered(depth, height, width, axis, device, dtype):
    grid = kornia.geometry.create_meshgrid3d(
        depth, height, width, normalized_coordinates=True, device=device, dtype=dtype
    )
    assert_close(grid[..., axis], torch.zeros_like(grid[..., axis]), atol=0.0, rtol=0.0)


@pytest.mark.parametrize(("trace_depth", "runtime_depth"), [(2, 1), (1, 2)])
def test_normalized_meshgrid3d_trace_crosses_singleton_boundary(trace_depth, runtime_depth, device, dtype):
    class MeshGrid3d(torch.nn.Module):
        def forward(self, volume):
            return kornia.geometry.create_meshgrid3d(
                volume.shape[-3],
                volume.shape[-2],
                volume.shape[-1],
                normalized_coordinates=True,
                device=volume.device,
                dtype=volume.dtype,
            )

    example = torch.zeros(1, 1, trace_depth, 3, 4, device=device, dtype=dtype)
    runtime = torch.zeros(1, 1, runtime_depth, 3, 4, device=device, dtype=dtype)
    traced = torch.jit.trace(MeshGrid3d(), example)
    assert_close(traced(runtime), MeshGrid3d()(runtime), atol=0.0, rtol=0.0)
