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

from kornia.core.exceptions import ValueCheckError
from kornia.geometry import find_essential, fit_plane, solve_pnp_dlt

from testing.base import DYNAMIC_EXPORT_UNAVAILABLE_REASON, BaseTester, dynamic_export_is_available


class TestTallCudaSvd(BaseTester):
    @pytest.mark.parametrize("count", [16, 17, 32, 33, 100])
    @pytest.mark.parametrize("operation", ["plane", "pnp", "essential"])
    def test_geometry_avoids_tall_cuda_decomposition_6559(self, count, operation, device, dtype, monkeypatch):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("The PnP input contract is float32/float64")

        def guard(original):
            def wrapped(matrix, *args, **kwargs):
                # The 33-row dispatch cliff is independent of matrix values.
                assert not (matrix.is_cuda and matrix.shape[-2] > 32 and matrix.shape[-1] <= 32)
                return original(matrix, *args, **kwargs)

            return wrapped

        monkeypatch.setattr(torch.linalg, "svd", guard(torch.linalg.svd))
        monkeypatch.setattr(torch.linalg, "svdvals", guard(torch.linalg.svdvals))
        points = torch.rand(2, count, 3, device=device, dtype=dtype) * 2 - 1
        if operation == "plane":
            points[..., 2] = 1 + 0.2 * points[..., 0] - 0.3 * points[..., 1]
            result = fit_plane(points)
            distances = torch.einsum("bi,bni->bn", result.normal.data, points) + result.offset.data[:, None]
            self.assert_close(distances, torch.zeros_like(points[..., 0]))
        else:
            camera = points + points.new_tensor([0, 0, 6])
            if operation == "pnp":
                intrinsics = torch.eye(3, device=device, dtype=dtype).expand(2, 3, 3)
                image = camera[..., :2] / camera[..., 2:]
                result = solve_pnp_dlt(points, image, intrinsics)
                expected = torch.cat((intrinsics, points.new_tensor([0, 0, 6]).expand(2, 3)[..., None]), -1)
                self.assert_close(result, expected)
            else:
                image1 = camera[..., :2] / camera[..., 2:]
                camera2 = camera + points.new_tensor([0.4, 0.1, 0.05])
                image2 = camera2[..., :2] / camera2[..., 2:]
                candidates = find_essential(image1, image2)
                ones = torch.ones_like(image1[..., :1])
                x1, x2 = torch.cat((image1, ones), -1), torch.cat((image2, ones), -1)
                residual = torch.einsum("bni,bkij,bnj->bkn", x2, candidates, x1)
                best = residual.square().mean(-1).nan_to_num(nan=float("inf")).min(-1).values
                self.assert_close(best, torch.zeros_like(best))

    def test_differentiable_tall_svd_preserves_forward_mode_6559(self, device):
        # R-only QR cannot be differentiated. JVP must retain the native SVD,
        # including under no_grad, where requires_grad alone is insufficient.
        from kornia.core.utils import _torch_svd_cast, _torch_svd_right

        dtype = torch.float32 if device.type == "mps" else torch.float64
        matrix = torch.randn(2, 40, 3, device=device, dtype=dtype)
        tangent = torch.randn_like(matrix)

        def direct(x):
            return _torch_svd_cast(x, full_matrices=False)[2][..., -1]

        def reduced(x):
            return _torch_svd_right(x)[1][..., -1]

        with torch.no_grad():
            expected = torch.func.jvp(direct, (matrix,), (tangent,))
            actual = torch.func.jvp(reduced, (matrix,), (tangent,))
        self.assert_close(actual[0], expected[0], rtol=0, atol=0)
        self.assert_close(actual[1], expected[1], rtol=0, atol=0)

    def test_dynamo_tall_right_vectors_6559(self, device, dtype, torch_optimizer):
        from kornia.core.utils import _torch_svd_right

        matrix = torch.randn(2, 40, 3, device=device, dtype=dtype)

        def op(x):
            return _torch_svd_right(x)

        with torch.no_grad():
            expected = op(matrix)
            actual = torch_optimizer(op)(matrix)
        self.assert_close(actual[0], expected[0])
        actual_vectors, expected_vectors = actual[1].mT, expected[1].mT
        self.assert_close(
            actual_vectors[..., :, None] * actual_vectors[..., None, :],
            expected_vectors[..., :, None] * expected_vectors[..., None, :],
        )

    def test_tall_rank_checks_reject_degenerate_points_6559(self, device, dtype):
        # With 40 points, CUDA reduces both rank checks by QR; they must still reject degenerate sets.
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("The PnP input contract is float32/float64")
        points = torch.rand(2, 40, 3, device=device, dtype=dtype) * 2 - 1
        planar = torch.stack((points[..., 0], points[..., 1], torch.full_like(points[..., 0], 5.0)), -1)
        intrinsics = torch.eye(3, device=device, dtype=dtype).expand(2, 3, 3)
        with pytest.raises(AssertionError, match="last singular value"):
            solve_pnp_dlt(planar, planar[..., :2] / planar[..., 2:], intrinsics)
        collinear = points[..., :1] * points.new_tensor([0.3, -0.5, 0.8]) + points.new_tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueCheckError, match="not collinear"):
            fit_plane(collinear)

    @pytest.mark.skipif(not dynamic_export_is_available(), reason=DYNAMIC_EXPORT_UNAVAILABLE_REASON)
    @pytest.mark.parametrize("operation", ["plane", "essential"])
    def test_dynamic_export_keeps_point_count_range_6559(self, operation, device, dtype):
        # The tall-CUDA size checks must not become guards that narrow an exported dynamic point count.
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("Checked in float32/float64")
        if device.type == "mps" and operation == "plane":
            pytest.skip("Exporting fit_plane on MPS adds a guard of its own, 6 * count < 8192")
        # Non-strict, the default from torch 2.9: strict export in 2.5 cannot trace Hyperplane. The range starts at
        # 10 because torch 2.5 adds a guard count > 9 to find_essential, with or without the tall-CUDA reduction.
        count = torch.export.Dim("count", min=10, max=100000)

        def scene(n):
            points = torch.rand(2, n, 3, device=device, dtype=dtype) * 2 - 1
            points[..., 2] = 1 + 0.2 * points[..., 0] - 0.3 * points[..., 1]
            return points

        if operation == "plane":

            class Plane(torch.nn.Module):
                def forward(self, points):
                    plane = fit_plane(points)
                    return plane.normal.data, plane.offset.data

            program = torch.export.export(Plane(), (scene(40),), dynamic_shapes=({1: count},), strict=False).module()
            for n in (40, 33, 32, 20, 10):
                points = scene(n)
                normal, offset = program(points)
                distances = torch.einsum("bi,bni->bn", normal, points) + offset[:, None]
                self.assert_close(distances, torch.zeros_like(distances))
        else:

            class Essential(torch.nn.Module):
                def forward(self, points1, points2):
                    return find_essential(points1, points2)

            def views(n):
                camera = torch.rand(2, n, 3, device=device, dtype=dtype) * 2 - 1
                camera = camera + camera.new_tensor([0, 0, 6])
                camera2 = camera + camera.new_tensor([0.4, 0.1, 0.05])
                return camera[..., :2] / camera[..., 2:], camera2[..., :2] / camera2[..., 2:]

            program = torch.export.export(
                Essential(), views(40), dynamic_shapes=({1: count}, {1: count}), strict=False
            ).module()
            for n in (40, 33, 32, 12, 10):
                image1, image2 = views(n)
                candidates = program(image1, image2)
                ones = torch.ones_like(image1[..., :1])
                x1, x2 = torch.cat((image1, ones), -1), torch.cat((image2, ones), -1)
                residual = torch.einsum("bni,bkij,bnj->bkn", x2, candidates, x1)
                best = residual.square().mean(-1).nan_to_num(nan=float("inf")).min(-1).values
                self.assert_close(best, torch.zeros_like(best))
