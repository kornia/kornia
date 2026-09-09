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

from kornia.geometry.camera import project_points_orthographic, unproject_points_orthographic
from kornia.geometry.vector import Vector2, Vector3
from kornia.sensors.camera.projection_model import OrthographicProjection, Z1Projection

from testing.base import BaseTester


class TestProjection(BaseTester):
    @pytest.mark.skip(reason="Unnecessary test")
    def test_smoke(self, device, dtype):
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_gradcheck(self, device):
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_module(self, device, dtype):
        pass

    def test_project(self, device, dtype):
        projection = Z1Projection()
        points = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [6.0, 6.0, 2.0], [9.0, 9.0, 3.0]],
            device=device,
            dtype=dtype,
        )
        expected = torch.tensor([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0], [3.0, 3.0]], device=device, dtype=dtype)
        self.assert_close(projection.project(Vector3(points)).data, expected)

    def test_unproject(self, device, dtype):
        projection = Z1Projection()
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0], [3.0, 3.0]], device=device, dtype=dtype)
        expected = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [6.0, 6.0, 2.0], [9.0, 9.0, 3.0]],
            device=device,
            dtype=dtype,
        )
        self.assert_close(
            projection.unproject(
                Vector2(points),
                torch.tensor([1.0, 1.0, 2.0, 3.0], device=device, dtype=dtype),
            ).data,
            expected,
        )

    def test_wart_project_divides_by_z_with_no_guard_4267(self, device, dtype):
        # Wart pin for kornia#4267 (audit labels 5d-sc-27, 5d-sc-28, Y4-05; pre-finding P2):
        # ``Z1Projection.project`` is a plain perspective division ``xy / z`` with no epsilon and no
        # validation, so a point ON the camera plane projects to inf rather than raising, and a point BEHIND
        # the camera projects to a finite pixel with no warning.  This is one of the five z-divide entry
        # points #4267 lists, and one of the four different answers the five give at (x, y, z) = (1, 2, 0).
        # For the audit's K = (fx, fy, cx, cy) = (100, 100, 4, 3), which the pixel-space members of the family
        # need and this normalized-space one does not: ``project_points`` masks the divide and returns
        # [[104.0, 203.0]], ``PinholeCamera.project`` masks it on the other side of K and returns
        # [[100.0, 200.0]], ``cam2pixel`` adds 1e-12 to z and returns ~[[1e14, 2e14]], while
        # ``project_points_z1`` and this method return inf.  The x and y components of the input differ
        # (1 vs 2) so a swapped reading of the pair is visible in the z = -4 arm.
        # Snippet used to generate expected: Z1Projection().project(Vector3(tensor([[1., 2., 0.]]))).data and
        # the same at z = -4 executed 2026-09-06 on this worktree (torch 2.14.0) -> [[inf, inf]] and
        # [[-0.25, -0.5]], on cpu for float32, float64, float16 and bfloat16 and on mps for float32 and
        # float16.
        # Pins the CURRENT policy; NOT a contract; delete when #4267 is repaired.
        projection = Z1Projection()
        on_plane = projection.project(Vector3(torch.tensor([[1.0, 2.0, 0.0]], device=device, dtype=dtype)))
        assert torch.isinf(on_plane.data).all()
        assert (on_plane.data > 0).all()
        behind = projection.project(Vector3(torch.tensor([[1.0, 2.0, -4.0]], device=device, dtype=dtype)))
        assert torch.isfinite(behind.data).all()
        self.assert_close(behind.data, torch.tensor([[-0.25, -0.5]], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        # The infinity is not the whole answer at z = 0: an axis whose NUMERATOR is also zero computes 0 / 0
        # and comes back nan, so a caller that tests the result with ``isinf`` misses that component.  The
        # arm above uses x = 1 != 0 and y = 2 != 0, which is exactly the case that hides this.
        # Snippet used to generate expected: Z1Projection().project(Vector3(tensor([[0., 2., 0.]]))).data
        # executed 2026-09-06 on this worktree (torch 2.14.0) -> [[nan, inf]], on cpu for float32, float64,
        # float16 and bfloat16 and on mps for float32 and float16.
        zero_numerator = projection.project(Vector3(torch.tensor([[0.0, 2.0, 0.0]], device=device, dtype=dtype)))
        assert torch.isnan(zero_numerator.data[..., 0]).all()
        assert torch.isinf(zero_numerator.data[..., 1]).all()
        assert (zero_numerator.data[..., 1] > 0).all()


class TestOrthographicProjection(BaseTester):
    def test_wart_orthographic_projection_is_a_placeholder_4284(self, device, dtype):
        # Wart pin for kornia#4284 (audit labels 5d-sc-22, 5d-sc-26): ``OrthographicProjection`` is the
        # projection half of the ORTHOGRAPHIC camera model and is a bare ``raise NotImplementedError`` with
        # an EMPTY message in both directions.  It is reachable through the public API --
        # ``CameraModel(..., CameraModelType.ORTHOGRAPHIC, ...)`` constructs and wires this class in -- so the
        # model-level project AND unproject raises pinned in tests/sensors/camera/test_camera_model.py both
        # come from here (the last traceback frame of each is measured as ``OrthographicProjection.project``
        # and ``OrthographicProjection.unproject`` -- a name rather than a line number, which would rot on
        # the next edit of that module), not from a distortion placeholder: ORTHOGRAPHIC pairs this
        # projection with the working ``AffineTransform``, which never fails for it in either direction.
        # Its ``matrix()`` raises from a third site again,
        # ``CameraModelBase.matrix``.  A working equivalent already exists next door as
        # ``kornia.geometry.camera.project_points_orthographic``.  The empty message is asserted rather than
        # described, because #4284's Expected asks at minimum for a message naming the model: a message-only
        # partial fix must flip this pin.
        # Snippet used to generate expected: OrthographicProjection().project(Vector3(tensor([[1., 2., 4.]])))
        # and .unproject(Vector2(tensor([[0.5, 0.25]])), tensor([2.])) executed 2026-09-06 on this worktree
        # (torch 2.14.0) -> NotImplementedError('') for both, on cpu for float32, float64, float16 and
        # bfloat16 and on mps for float32 and float16.
        # Pins the CURRENT behaviour; NOT a contract; delete when #4284 is repaired.
        projection = OrthographicProjection()
        with pytest.raises(NotImplementedError) as raised:
            projection.project(Vector3(torch.tensor([[1.0, 2.0, 4.0]], device=device, dtype=dtype)))
        assert str(raised.value) == ""
        with pytest.raises(NotImplementedError) as raised:
            projection.unproject(
                Vector2(torch.tensor([[0.5, 0.25]], device=device, dtype=dtype)),
                torch.tensor([2.0], device=device, dtype=dtype),
            )
        assert str(raised.value) == ""
        assert isinstance(
            Z1Projection().project(Vector3(torch.tensor([[1.0, 2.0, 4.0]], device=device, dtype=dtype))), Vector2
        )

    def test_unproject_scalar_depth(self, device, dtype):
        """Regression test: scalar depth must preserve device and dtype.

        PR #4340: Z1Projection.unproject used to build a CPU float32 tensor
        for python int/float depth, which raised RuntimeError on every
        accelerator and widened float16/bfloat16 results to float32.
        """
        projection = Z1Projection()
        points = Vector2(torch.tensor([[0.25, 0.5]], device=device, dtype=dtype))
        expected = projection.unproject(
            points,
            torch.tensor([4.0], device=device, dtype=dtype),
        ).data
        for depth in (4, 4.0):
            out = projection.unproject(points, depth)
            assert out.data.device.type == device.type, f"expected device {device.type}, got {out.data.device.type}"
            assert out.data.dtype == dtype, f"expected dtype {dtype}, got {out.data.dtype}"
            self.assert_close(out.data, expected, atol=0.0, rtol=0.0)
        # Also verify a vector depth still works unchanged
        out_vec = projection.unproject(points, torch.tensor([4.0], device=device, dtype=dtype))
        self.assert_close(out_vec.data, expected, atol=0.0, rtol=0.0)

    def test_orthographic_project_matches_geometry(self, device, dtype):
        projection = OrthographicProjection()
        points = torch.tensor(
            [[1.0, 2.0, 3.0], [-4.0, 5.0, 7.0]],
            device=device,
            dtype=dtype,
        )

        expected = project_points_orthographic(points)
        actual = projection.project(Vector3(points)).data

        self.assert_close(actual, expected, atol=0.0, rtol=0.0)

    def test_orthographic_unproject_matches_geometry(self, device, dtype):
        projection = OrthographicProjection()
        points = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            device=device,
            dtype=dtype,
        )
        depth = torch.tensor([5.0, 6.0], device=device, dtype=dtype)

        expected = unproject_points_orthographic(points, depth)
        actual = projection.unproject(Vector2(points), depth).data

        self.assert_close(actual, expected, atol=0.0, rtol=0.0)
