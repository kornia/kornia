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

from kornia.geometry.camera.perspective import project_points
from kornia.geometry.camera.projection_orthographic import (
    dx_project_points_orthographic,
    project_points_orthographic,
    unproject_points_orthographic,
)
from kornia.geometry.camera.projection_z1 import dx_project_points_z1, project_points_z1, unproject_points_z1

from testing.base import BaseTester


class TestProjectionZ1(BaseTester):
    def test_smoke(self, device, dtype):
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        assert project_points_z1(points) is not None

    def _test_cardinality_unproject_batch(self, device, dtype, batch_size):
        batch_tuple = (batch_size,) if batch_size is not None else ()
        points = torch.rand(batch_tuple + (3,), device=device, dtype=dtype)
        assert project_points_z1(points).shape == batch_tuple + (2,)

    def _test_cardinality_project_batch(self, device, dtype, batch_size):
        batch_tuple = (batch_size,) if batch_size is not None else ()
        points = torch.rand(batch_tuple + (2,), device=device, dtype=dtype)
        assert unproject_points_z1(points).shape == batch_tuple + (3,)

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 3])
    def test_cardinality(self, device, dtype, batch_size):
        self._test_cardinality_project_batch(device, dtype, batch_size)
        self._test_cardinality_unproject_batch(device, dtype, batch_size)

    def test_project_points_z1(self, device, dtype):
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        expected = torch.tensor([0.3333333432674408, 0.6666666865348816], device=device, dtype=dtype)
        self.assert_close(project_points_z1(points), expected)

    def test_project_points_z1_batch(self, device, dtype):
        points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device, dtype=dtype)
        expected = torch.tensor(
            [
                [0.3333333432674408, 0.6666666865348816],
                [0.6666666865348816, 0.8333333730697632],
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(project_points_z1(points), expected)

    def test_project_points_z1_invalid(self, device, dtype):
        # NOTE: this is a corner case where the depth is 0.0 and the point is at infinity
        #      the projection is not defined and the function returns inf. The second point
        #      is behind the camera which is not a valid point and the user should handle it.
        points = torch.tensor([[1.0, 2.0, 0.0], [4.0, 5.0, -1.0]], device=device, dtype=dtype)
        expected = torch.tensor([[float("inf"), float("inf")], [-4.0, -5.0]], device=device, dtype=dtype)
        self.assert_close(project_points_z1(points), expected)

    def test_unproject_points_z1(self, device, dtype):
        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        expected = torch.tensor([1.0, 2.0, 1.0], device=device, dtype=dtype)
        self.assert_close(unproject_points_z1(points), expected)

    def test_unproject_points_z1_batch(self, device, dtype):
        points = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=dtype)
        expected = torch.tensor([[1.0, 2.0, 1.0], [3.0, 4.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(unproject_points_z1(points), expected)

    def test_project_unproject(self, device, dtype):
        points = torch.tensor([1.0, 2.0, 2.0], device=device, dtype=dtype)
        extension = torch.tensor([2.0], device=device, dtype=dtype)
        self.assert_close(unproject_points_z1(project_points_z1(points), extension), points)

    def test_unproject_points_z1_extension(self, device, dtype):
        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        extension = torch.tensor([2.0], device=device, dtype=dtype)
        expected = torch.tensor([2.0, 4.0, 2.0], device=device, dtype=dtype)
        self.assert_close(unproject_points_z1(points, extension), expected)

    def test_unproject_points_z1_batch_extension(self, device, dtype):
        points = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=dtype)
        extension = torch.tensor([2.0, 3.0], device=device, dtype=dtype)
        expected = torch.tensor([[2.0, 4.0, 2.0], [9.0, 12.0, 3.0]], device=device, dtype=dtype)
        self.assert_close(unproject_points_z1(points, extension), expected)

    @pytest.mark.parametrize("batch_shape", [(), (0,), (1,), (2,), (1, 2), (2, 3)])
    @pytest.mark.parametrize("column_depth", [False, True])
    def test_unproject_depth_shapes(self, device, dtype, batch_shape, column_depth):
        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype).expand(batch_shape + (2,))
        depth_shape = batch_shape + (1,) if column_depth else batch_shape
        depth = torch.full(depth_shape, 3.0, device=device, dtype=dtype)
        expected = torch.tensor([3.0, 6.0, 3.0], device=device, dtype=dtype).expand(batch_shape + (3,))
        actual = unproject_points_z1(points, depth)
        self.assert_close(actual, expected)
        self.assert_close(project_points_z1(actual), points)
        self.assert_close(torch.jit.script(unproject_points_z1)(points, depth), expected)

    def test_unproject_depth_shape_mismatch(self, device, dtype):
        points = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]], device=device, dtype=dtype)
        depth = torch.tensor([[[3.0], [5.0]]], device=device, dtype=dtype)
        with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
            unproject_points_z1(points, depth)

    @pytest.mark.parametrize("column_depth", [False, True])
    def test_unproject_batched_depth_gradcheck(self, device, column_depth):
        points = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=torch.float64)
        depth = torch.tensor([2.0, 3.0], device=device, dtype=torch.float64)
        if column_depth:
            depth = depth.unsqueeze(-1)
        self.gradcheck(unproject_points_z1, (points, depth))

    def test_dx_proj_x(self, device, dtype):
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        expected = torch.tensor(
            [
                [0.3333333432674408, 0.0, -0.1111111119389534],
                [0.0, 0.3333333432674408, -0.2222222238779068],
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(dx_project_points_z1(points), expected)

    def test_exception(self, device, dtype) -> None:
        from kornia.core.exceptions import ShapeError

        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        extension = torch.tensor([2.0], device=device, dtype=dtype)
        with pytest.raises(ShapeError):
            unproject_points_z1(points, extension)

    def _test_gradcheck_unproject(self, device):
        points = torch.tensor([1.0, 2.0], device=device, dtype=torch.float64)
        extension = torch.tensor([2.0], device=device, dtype=torch.float64)
        self.gradcheck(unproject_points_z1, (points, extension))

    def _test_gradcheck_project(self, device):
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=torch.float64)
        self.gradcheck(project_points_z1, (points,))

    def test_gradcheck(self, device) -> None:
        self._test_gradcheck_project(device)
        self._test_gradcheck_unproject(device)

    def _test_jit_unproject(self, device, dtype) -> None:
        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        extension = torch.tensor([2.0], device=device, dtype=dtype)
        op_script = torch.jit.script(unproject_points_z1)
        actual = op_script(points, extension)
        expected = unproject_points_z1(points, extension)
        self.assert_close(actual, expected)

    def _test_jit_project(self, device, dtype) -> None:
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        op_script = torch.jit.script(project_points_z1)
        actual = op_script(points)
        expected = project_points_z1(points)
        self.assert_close(actual, expected)

    def test_jit(self, device, dtype) -> None:
        self._test_jit_project(device, dtype)
        self._test_jit_unproject(device, dtype)

    def test_convention_unproject_points_z1_accepts_both_extension_shapes(self, device, dtype):
        # The rank-based guard introduced in f4532f39 accepts both depth representations for multidimensional
        # batches. Snippet used to generate expected: points [[[1, 2]], [[3, 4]]] with depths [[3], [5]] and
        # [[[3]], [[5]]] -> [[[3, 6, 3]], [[15, 20, 5]]] for both.
        points = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]], device=device, dtype=dtype)
        expected = torch.tensor([[[3.0, 6.0, 3.0]], [[15.0, 20.0, 5.0]]], device=device, dtype=dtype)
        flat = unproject_points_z1(points, torch.tensor([[3.0], [5.0]], device=device, dtype=dtype))
        column = unproject_points_z1(points, torch.tensor([[[3.0]], [[5.0]]], device=device, dtype=dtype))
        self.assert_close(flat, expected, atol=0.0, rtol=0.0)
        self.assert_close(column, expected, atol=0.0, rtol=0.0)

    def test_wart_project_points_z1_zero_depth_is_component_dependent_4267(self, device, dtype):
        # project_points_z1 divides plainly. Snippet used to generate expected: project_points_z1 applied to the
        # four points below -> [[inf, -inf], [-inf, inf], [nan, inf], [inf, nan]].
        points = torch.tensor(
            [[1.0, -2.0, 0.0], [-1.0, 2.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 0.0]],
            device=device,
            dtype=dtype,
        )
        actual = project_points_z1(points)
        expected_posinf = torch.tensor([[True, False], [False, True], [False, True], [True, False]], device=device)
        expected_neginf = torch.tensor([[False, True], [True, False], [False, False], [False, False]], device=device)
        expected_nan = torch.tensor([[False, False], [False, False], [True, False], [False, True]], device=device)
        assert torch.equal(torch.isposinf(actual), expected_posinf)
        assert torch.equal(torch.isneginf(actual), expected_neginf)
        assert torch.equal(torch.isnan(actual), expected_nan)

    def test_convention_project_points_z1_differs_below_perspective_epsilon(self, device, dtype):
        # Snippet used to generate expected: project_points_z1([[1., 2., 1e-9]]) -> [[1e9, 2e9]], while
        # project_points(..., eye(3)) -> [[1., 2.]] because its homogeneous conversion does not divide when
        # abs(z) <= 1e-8. float16 is skipped because 1e-9 underflows to zero in that dtype.
        if dtype == torch.float16:
            pytest.skip("1e-9 underflows to zero in float16")
        points = torch.tensor([[1.0, 2.0, 1e-9]], device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        expected_z1 = torch.tensor([[1e9, 2e9]], device=device, dtype=dtype)
        expected_perspective = torch.tensor([[1.0, 2.0]], device=device, dtype=dtype)
        self.assert_close(project_points_z1(points), expected_z1)
        self.assert_close(project_points(points, camera_matrix), expected_perspective)

    def test_convention_dx_project_points_z1_matches_autograd(self, device, dtype):
        # Convention pin: dx_project_points_z1 returns the (..., 2, 3) Jacobian
        # of project_points_z1, laid out d(u, v) / d(x, y, z) -- row-major in the OUTPUT index. Checked against
        # torch.autograd.functional.jacobian at an off-axis point (1, 2, 3) where all six entries differ, so a
        # transposed layout or a swapped (u, v) row fails.
        # Snippet used to generate expected: autograd.functional.jacobian(project_points_z1, [1., 2., 3.])
        # executed 2026-09-05 (torch 2.14.0, cpu and mps) -> max abs difference 1.49e-08 in float32, 0.0 in
        # float64/float16/bfloat16.
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        analytic = dx_project_points_z1(points)
        numeric = torch.autograd.functional.jacobian(project_points_z1, points)
        assert analytic.shape == (2, 3)
        self.assert_close(analytic, numeric)


class TestProjectionOrthographic(BaseTester):
    def test_smoke(self, device, dtype):
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        assert project_points_orthographic(points) is not None

    def _test_cardinality_unproject_batch(self, device, dtype, batch_size):
        batch_tuple = (batch_size,) if batch_size is not None else ()
        points = torch.rand(batch_tuple + (3,), device=device, dtype=dtype)
        assert project_points_orthographic(points).shape == batch_tuple + (2,)

    def _test_cardinality_project_batch(self, device, dtype, batch_size):
        batch_tuple = (batch_size,) if batch_size is not None else ()
        points = torch.rand(batch_tuple + (2,), device=device, dtype=dtype)
        extension = torch.rand(batch_tuple, device=device, dtype=dtype)
        assert unproject_points_orthographic(points, extension).shape == batch_tuple + (3,)

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 3])
    def test_cardinality(self, device, dtype, batch_size):
        self._test_cardinality_project_batch(device, dtype, batch_size)
        self._test_cardinality_unproject_batch(device, dtype, batch_size)

    def test_project_points_orthographic(self, device, dtype):
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        expected = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        self.assert_close(project_points_orthographic(points), expected)

    def test_project_points_orthographic_batch(self, device, dtype):
        points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device, dtype=dtype)
        expected = torch.tensor([[1.0, 2.0], [4.0, 5.0]], device=device, dtype=dtype)
        self.assert_close(project_points_orthographic(points), expected)

    def test_unproject_points_orthographic_extension(self, device, dtype):
        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        extension = torch.tensor([2.0], device=device, dtype=dtype)
        expected = torch.tensor([1.0, 2.0, 2.0], device=device, dtype=dtype)
        self.assert_close(unproject_points_orthographic(points, extension), expected)

    def test_unproject_points_orthographic_batch_extension(self, device, dtype):
        points = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=dtype)
        extension = torch.tensor([2.0, 3.0], device=device, dtype=dtype)
        expected = torch.tensor([[1.0, 2.0, 2.0], [3.0, 4.0, 3.0]], device=device, dtype=dtype)
        self.assert_close(unproject_points_orthographic(points, extension), expected)

    def test_project_unproject(self, device, dtype):
        points = torch.tensor([1.0, 2.0, 2.0], device=device, dtype=dtype)
        extension = torch.tensor([2.0], device=device, dtype=dtype)
        self.assert_close(unproject_points_orthographic(project_points_orthographic(points), extension), points)

    def test_dx_proj_x(self, device, dtype):
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        expected = torch.tensor([1.0], device=device, dtype=dtype)
        self.assert_close(dx_project_points_orthographic(points), expected)

    def test_exception(self, device, dtype) -> None:
        from kornia.core.exceptions import ShapeError

        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        extension = torch.tensor([2.0], device=device, dtype=dtype)
        with pytest.raises(ShapeError):
            unproject_points_orthographic(points, extension)

    def test_convention_dx_orthographic_is_the_scalar_du_dx_not_the_full_jacobian(self, device, dtype):
        # Convention pin: dx_project_points_orthographic returns the single
        # partial derivative its docstring math states, du/dx = 1, with shape (..., 1) -- NOT the (2, 3)
        # Jacobian of project_points_orthographic, and NOT the shape its same-named z1 sibling returns. The two
        # ``dx_*`` functions on this surface therefore mean different things, so the pin asserts the shapes
        # apart and checks each against torch.autograd.functional.jacobian of the projection it differentiates.
        # An off-axis point (x != y != z, one negative) is used so a transposed or wrongly indexed Jacobian
        # changes a literal. Snippet used to generate expected: the four calls below on [1., -2., 4.] executed
        # 2026-09-05 (torch 2.14.0, cpu and mps, float16/bfloat16/float32/float64) ->
        # dx_orthographic [1.] shape (1,); jacobian(project_points_orthographic) [[1, 0, 0], [0, 1, 0]] shape
        # (2, 3); dx_z1 == jacobian(project_points_z1) == [[0.25, 0, -0.0625], [0, 0.25, 0.125]] shape (2, 3).
        # Every literal is a dyadic rational and matched exactly in every dtype and device tested.
        points = torch.tensor([1.0, -2.0, 4.0], device=device, dtype=dtype)
        dx_ortho = dx_project_points_orthographic(points)
        assert dx_ortho.shape == (1,)
        self.assert_close(dx_ortho, torch.tensor([1.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        jacobian_ortho = torch.autograd.functional.jacobian(project_points_orthographic, points)
        assert jacobian_ortho.shape == (2, 3)
        self.assert_close(
            jacobian_ortho,
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype),
            atol=0.0,
            rtol=0.0,
        )
        # dx_project_points_z1, by contrast, IS the full Jacobian of its projection.
        dx_z1 = dx_project_points_z1(points)
        assert dx_z1.shape == (2, 3)
        self.assert_close(dx_z1, torch.autograd.functional.jacobian(project_points_z1, points), atol=0.0, rtol=0.0)
        self.assert_close(
            dx_z1,
            torch.tensor([[0.25, 0.0, -0.0625], [0.0, 0.25, 0.125]], device=device, dtype=dtype),
            atol=0.0,
            rtol=0.0,
        )

    def _test_gradcheck_project(self, device):
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=torch.float64)
        self.gradcheck(project_points_orthographic, (points,))

    def _test_gradcheck_unproject(self, device):
        points = torch.tensor([1.0, 2.0], device=device, dtype=torch.float64)
        extension = torch.tensor([2.0], device=device, dtype=torch.float64)
        self.gradcheck(unproject_points_orthographic, (points, extension))

    def test_gradcheck(self, device) -> None:
        self._test_gradcheck_project(device)
        self._test_gradcheck_unproject(device)

    def _test_jit_project(self, device, dtype) -> None:
        points = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        op_script = torch.jit.script(project_points_orthographic)
        actual = op_script(points)
        expected = project_points_orthographic(points)
        self.assert_close(actual, expected)

    def _test_jit_unproject(self, device, dtype) -> None:
        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        extension = torch.tensor([2.0], device=device, dtype=dtype)
        op_script = torch.jit.script(unproject_points_orthographic)
        actual = op_script(points, extension)
        expected = unproject_points_orthographic(points, extension)
        self.assert_close(actual, expected)

    def test_jit(self, device, dtype) -> None:
        self._test_jit_project(device, dtype)
        self._test_jit_unproject(device, dtype)

    def test_convention_orthographic_drops_and_restores_the_z_axis(self, device, dtype):
        # Convention pin: the orthographic projection drops z and keeps (x, y)
        # in order -- (1, 2, 3) -> (1, 2), never (1, 3) or (2, 1) -- and the unprojection appends the extension as
        # the z component, so (1, 2) with extension 3 restores (1, 2, 3) exactly. Distinct x, y and z so every
        # axis permutation changes the literal.
        # Snippet used to generate expected: both calls executed 2026-09-05 (torch 2.14.0, cpu and mps, every
        # dtype) -> [1., 2.] and [1., 2., 3.]; exact, no divide is involved.
        points_3d = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        projected = project_points_orthographic(points_3d)
        self.assert_close(projected, torch.tensor([1.0, 2.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        restored = unproject_points_orthographic(projected, torch.tensor([3.0], device=device, dtype=dtype))
        self.assert_close(restored, points_3d, atol=0.0, rtol=0.0)

    def test_convention_unproject_points_orthographic_accepts_both_extension_shapes(self, device, dtype):
        # Convention pin: unproject_points_orthographic compares the
        # extension's RANK with the points' rank -- the right predicate -- so a (N,) and a (N, 1) extension are
        # both accepted for N > 1 and give the same answer. Its sibling unproject_points_z1 uses the same
        # rank-based guard after f4532f39.
        # Snippet used to generate expected: points [[1, 2], [3, 4]] with extensions [5, 6] and [[5], [6]]
        # executed 2026-09-05 (torch 2.14.0, cpu and mps, every dtype) -> [[1., 2., 5.], [3., 4., 6.]] for both.
        points = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=dtype)
        expected = torch.tensor([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]], device=device, dtype=dtype)
        flat = unproject_points_orthographic(points, torch.tensor([5.0, 6.0], device=device, dtype=dtype))
        column = unproject_points_orthographic(points, torch.tensor([[5.0], [6.0]], device=device, dtype=dtype))
        self.assert_close(flat, expected, atol=0.0, rtol=0.0)
        self.assert_close(column, expected, atol=0.0, rtol=0.0)
