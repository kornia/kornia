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

from testing.base import BaseTester, script_uncached


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
        # NOTE: corner cases with no valid projection. At depth 0.0 the divide is skipped and (x, y) comes back as
        #      a finite placeholder (#4267). The second point is behind the camera, which is not a valid point
        #      and the user should handle it (#4555).
        points = torch.tensor([[1.0, 2.0, 0.0], [4.0, 5.0, -1.0]], device=device, dtype=dtype)
        expected = torch.tensor([[1.0, 2.0], [-4.0, -5.0]], device=device, dtype=dtype)
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
    def test_unproject_depth_shapes_4282(self, device, dtype, batch_shape, column_depth):
        points = torch.tensor([1.0, 2.0], device=device, dtype=dtype).expand(batch_shape + (2,))
        depth_shape = batch_shape + (1,) if column_depth else batch_shape
        depth = torch.full(depth_shape, 3.0, device=device, dtype=dtype)
        expected = torch.tensor([3.0, 6.0, 3.0], device=device, dtype=dtype).expand(batch_shape + (3,))
        actual = unproject_points_z1(points, depth)
        self.assert_close(actual, expected)
        self.assert_close(project_points_z1(actual), points)
        self.assert_close(torch.jit.script(unproject_points_z1)(points, depth), expected)

    @pytest.mark.parametrize("depth_shape", [(3,), (2, 3, 3)])
    def test_unproject_rejects_a_depth_per_component(self, device, dtype, depth_shape):
        from kornia.core.exceptions import ShapeError

        # a depth whose trailing dimension is 3 would broadcast against (u, v, 1) and scale each component separately
        points = torch.ones(2, 3, 2, device=device, dtype=dtype)
        depth = torch.full(depth_shape, 2.0, device=device, dtype=dtype)
        with pytest.raises(ShapeError):
            unproject_points_z1(points, depth)

    @pytest.mark.parametrize("depth_form", ["none", "flat", "column"])
    def test_unproject_scripted_empty_after_nonempty_keeps_dtype(self, device, dtype, depth_form):
        # TorchScript's CUDA fuser used to compile ``cat([points * depth, depth])`` into a group that, once it had
        # specialized on non-empty calls, returned an int32 tensor for an empty batch.
        scripted = script_uncached(unproject_points_z1)
        for batch in (2, 2, 2, 0, 3, 0, 1, 0):
            points = torch.rand(batch, 2, device=device, dtype=dtype)
            depth = None
            if depth_form != "none":
                depth = torch.rand((batch, 1) if depth_form == "column" else (batch,), device=device, dtype=dtype)
            actual = scripted(points, depth)
            assert actual.dtype == dtype
            assert actual.shape == (batch, 3)
            self.assert_close(actual, unproject_points_z1(points, depth))

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

    @pytest.mark.parametrize("z", [0.0, 1e-9, -1e-9, 1e-8, -1e-8])
    def test_convention_project_points_z1_skips_the_divide_at_small_depth_4267(self, device, dtype, z):
        # The #4267 rule: where abs(z) <= 1e-8 the divide is skipped and (x, y) comes back unchanged, in every
        # component (a zero numerator included), as project_points does with K = I. The Jacobian there is that of
        # (x, y): [[1, 0, 0], [0, 1, 0]]. (In float16 these depths underflow to 0.)
        points = torch.tensor(
            [[1.0, -2.0, z], [-1.0, 2.0, z], [0.0, 2.0, z], [1.0, 0.0, z]],
            device=device,
            dtype=dtype,
        )
        self.assert_close(project_points_z1(points), points[..., :2], atol=0.0, rtol=0.0)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        self.assert_close(project_points(points, camera_matrix), points[..., :2], atol=0.0, rtol=0.0)
        expected_jacobian = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)
        self.assert_close(dx_project_points_z1(points), expected_jacobian.expand(4, 2, 3), atol=0.0, rtol=0.0)

    def test_convention_project_points_z1_divides_exactly_elsewhere_4267(self, device, dtype):
        # Outside the threshold the divide is exact (no epsilon enters it), behind the camera too, and a nan
        # depth takes the divide, so the projection and its Jacobian stay nan.
        if dtype == torch.float16:
            pytest.skip("2e-8 underflows to zero in float16")
        points = torch.tensor([[1.0, 2.0, 2e-8], [1.0, 2.0, -4.0]], device=device, dtype=dtype)
        expected = torch.tensor([[1.0 / 2e-8, 2.0 / 2e-8], [-0.25, -0.5]], device=device, dtype=dtype)
        self.assert_close(project_points_z1(points), expected)
        nan_depth = torch.tensor([[1.0, 2.0, float("nan")]], device=device, dtype=dtype)
        assert torch.isnan(project_points_z1(nan_depth)).all()
        assert torch.isnan(dx_project_points_z1(nan_depth)[..., 2]).all()

    @pytest.mark.parametrize("z", [0.0, 5e-9, -5e-9, 1.0, -4.0])
    def test_convention_dx_project_points_z1_matches_autograd_on_both_branches_4267(self, device, dtype, z):
        # On both sides of the masked divide the analytic Jacobian matches autograd through project_points_z1.
        points = torch.tensor([1.0, 2.0, z], device=device, dtype=dtype)
        numeric = torch.autograd.functional.jacobian(project_points_z1, points)
        self.assert_close(dx_project_points_z1(points), numeric)

    def test_gradcheck_project_points_z1_on_both_branches_4267(self, device):
        # The skipped branch uses a safe-argument torch.where, so neither branch leaks inf or nan into the
        # gradient. Finite differences of 1e-10 keep each z inside its branch.
        points = torch.tensor(
            [[1.0, 2.0, 0.0], [1.0, 2.0, 5e-9], [1.0, 2.0, -5e-9], [1.0, 2.0, 3e-8], [1.0, 2.0, 0.5]],
            device=device,
            dtype=torch.float64,
        )
        self.gradcheck(project_points_z1, (points,), eps=1e-10)

    def test_convention_dx_project_points_z1_matches_autograd(self, device, dtype):
        # The (..., 2, 3) Jacobian d(u, v) / d(x, y, z), rows indexed by the output; checked against autograd at
        # an off-axis point where all six entries differ.
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
        # dx_project_points_orthographic returns the scalar du/dx = 1 with shape (..., 1), not the (2, 3)
        # Jacobian that its z1 sibling dx_project_points_z1 returns. Every literal is dyadic, so exact.
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
        # Projection drops z and keeps (x, y) in order; unprojection appends the extension as z.
        points_3d = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        projected = project_points_orthographic(points_3d)
        self.assert_close(projected, torch.tensor([1.0, 2.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        restored = unproject_points_orthographic(projected, torch.tensor([3.0], device=device, dtype=dtype))
        self.assert_close(restored, points_3d, atol=0.0, rtol=0.0)

    def test_convention_unproject_points_orthographic_accepts_both_extension_shapes(self, device, dtype):
        # A (N,) and a (N, 1) extension are both accepted for N > 1 and give the same answer.
        points = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=dtype)
        expected = torch.tensor([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]], device=device, dtype=dtype)
        flat = unproject_points_orthographic(points, torch.tensor([5.0, 6.0], device=device, dtype=dtype))
        column = unproject_points_orthographic(points, torch.tensor([[5.0], [6.0]], device=device, dtype=dtype))
        self.assert_close(flat, expected, atol=0.0, rtol=0.0)
        self.assert_close(column, expected, atol=0.0, rtol=0.0)
