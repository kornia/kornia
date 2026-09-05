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

    def test_wart_unproject_points_z1_rejects_the_documented_extension_shape_4282(self, device, dtype):
        # Wart pin for kornia#4282 (audit labels 5a-z1-11, 5a-z1-12, Y5-18): the guard is written
        # ``elif extension.shape[0] > 1: extension = extension[..., None]``, which unsqueezes on the BATCH size
        # rather than comparing ranks, so it is exactly backwards -- the documented ``(..., 1)`` extension raises
        # for N > 1 while an undocumented ``(N,)`` extension works. The sibling unproject_points_orthographic
        # compares ranks and accepts both (pinned in TestProjectionOrthographic below).
        # Snippet used to generate expected: points [[1, 2], [3, 4]] with extension [3, 5] executed 2026-09-05
        # (torch 2.14.0, cpu and mps, every dtype) -> [[3., 6., 3.], [15., 20., 5.]]; the (2, 1) extension raises
        # RuntimeError("Sizes of tensors must match except in dimension 2").
        # Pins the CURRENT behavior; NOT a contract; delete when #4282 is repaired.
        points = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=dtype)
        undocumented = unproject_points_z1(points, torch.tensor([3.0, 5.0], device=device, dtype=dtype))
        expected = torch.tensor([[3.0, 6.0, 3.0], [15.0, 20.0, 5.0]], device=device, dtype=dtype)
        self.assert_close(undocumented, expected, atol=0.0, rtol=0.0)
        with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
            unproject_points_z1(points, torch.tensor([[3.0], [5.0]], device=device, dtype=dtype))

    def test_wart_project_points_z1_returns_inf_at_z_zero_4267(self, device, dtype):
        # Wart pin for kornia#4267 (audit labels 5a-z1-02, 5a-z1-07, Y4-05): project_points_z1 divides plainly, so
        # the camera-plane point (1, 2, 0) returns inf in every dtype. Its docstring states a ``z > 0``
        # precondition that is never validated. This is one of four different answers the namespace gives at the
        # same singular input -- project_points returns [[104, 203]], PinholeCamera.project [[100, 200]] and
        # cam2pixel a finite 1e14.
        # Snippet used to generate expected: project_points_z1([[1., 2., 0.]]) executed 2026-09-05 (torch 2.14.0,
        # cpu and mps, every dtype) -> [[inf, inf]]. NOT a contract; delete when #4267 is repaired.
        out = project_points_z1(torch.tensor([[1.0, 2.0, 0.0]], device=device, dtype=dtype))
        assert bool(torch.isinf(out).all())
        assert bool((out > 0).all())

    def test_convention_dx_project_points_z1_matches_autograd(self, device, dtype):
        # Convention pin (audit labels 5a-z1-15, 5a-z1-16): dx_project_points_z1 returns the (..., 2, 3) Jacobian
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
        # Convention pin (audit labels 5a-or-01, 5a-or-03): the orthographic projection drops z and keeps (x, y)
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
        # Convention pin (audit labels 5a-or-04, 5a-or-05, Y5-19): unproject_points_orthographic compares the
        # extension's RANK with the points' rank -- the right predicate -- so a (N,) and a (N, 1) extension are
        # both accepted for N > 1 and give the same answer. Its sibling unproject_points_z1 uses the batch size
        # instead and rejects the documented (N, 1) shape for N > 1 (kornia#4282, pinned in TestProjectionZ1).
        # Snippet used to generate expected: points [[1, 2], [3, 4]] with extensions [5, 6] and [[5], [6]]
        # executed 2026-09-05 (torch 2.14.0, cpu and mps, every dtype) -> [[1., 2., 5.], [3., 4., 6.]] for both.
        points = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=dtype)
        expected = torch.tensor([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]], device=device, dtype=dtype)
        flat = unproject_points_orthographic(points, torch.tensor([5.0, 6.0], device=device, dtype=dtype))
        column = unproject_points_orthographic(points, torch.tensor([[5.0], [6.0]], device=device, dtype=dtype))
        self.assert_close(flat, expected, atol=0.0, rtol=0.0)
        self.assert_close(column, expected, atol=0.0, rtol=0.0)
