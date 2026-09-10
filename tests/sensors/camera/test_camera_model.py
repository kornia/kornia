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

from kornia.geometry.camera import project_points, unproject_points
from kornia.geometry.vector import Vector2, Vector3
from kornia.image import ImageSize
from kornia.sensors.camera import CameraModel, CameraModelBase, CameraModelType
from kornia.sensors.camera.camera_model import Orthographic
from kornia.sensors.camera.distortion_model import (
    AffineTransform,
    BrownConradyTransform,
    KannalaBrandtK3Transform,
)
from kornia.sensors.camera.projection_model import OrthographicProjection, Z1Projection

from testing.base import BaseTester

# Asymmetric fixture shared by the convention/wart pins below: fx = 100 != fy = 50 and cx = 4 != cy = 3 on a
# non-square 6 x 8 image, so a transposed or swapped reading of the parameter vector changes the literals.
_PARAMS = (100.0, 50.0, 4.0, 3.0)


def _pinhole(device, dtype, params=_PARAMS):
    """Build the asymmetric PINHOLE ``CameraModel`` the pins below share."""
    return CameraModel(ImageSize(6, 8), CameraModelType.PINHOLE, torch.tensor(params, device=device, dtype=dtype))


def _k3(device, dtype, params=_PARAMS):
    """Build the (1, 3, 3) ``kornia.geometry.camera`` intrinsics matrix for the same parameters."""
    fx, fy, cx, cy = params
    return torch.tensor([[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)


class TestPinholeCamera(BaseTester):
    def _make_rand_data(self, batch_size, device, dtype):
        params = torch.rand(batch_size, 4).to(dtype).to(device)
        image_sizes = torch.randint(1, 100, (batch_size, 2)).to(dtype).to(device)
        return params, ImageSize(image_sizes[:, 0], image_sizes[:, 1])

    def test_smoke(self, device, dtype):
        params, image_size = self._make_rand_data(1, device, dtype)
        cam = CameraModel(image_size, CameraModelType.PINHOLE, params)
        assert isinstance(cam, CameraModel)
        self.assert_close(cam.params, params)
        self.assert_close(image_size.height, cam.height)
        self.assert_close(image_size.width, cam.width)

    @pytest.mark.skip(reason="Unnecessary test")
    def test_cardinality(self, device, dtype):
        pass

    def test_exception(self, device, dtype):
        # test for invalid params for different camera models
        params = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        image_size = ImageSize(100, 100)
        with pytest.raises(ValueError):
            CameraModel(image_size, CameraModelType.PINHOLE, params)
        with pytest.raises(ValueError):
            CameraModel(image_size, CameraModelType.BROWN_CONRADY, params)
        with pytest.raises(ValueError):
            CameraModel(image_size, CameraModelType.KANNALA_BRANDT_K3, params)
        with pytest.raises(ValueError):
            CameraModel(image_size, CameraModelType.ORTHOGRAPHIC, params)

    @pytest.mark.skip(reason="Unnecessary test")
    def test_gradcheck(self, device):
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="Unnecessary test")
    def test_module(self, device, dtype):
        pass

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_project_unproject(self, device, dtype, batch_size):
        params, image_size = self._make_rand_data(batch_size, device, dtype)
        cam = CameraModel(image_size, CameraModelType.PINHOLE, params)
        points = torch.rand((batch_size, 3), device=device, dtype=dtype)
        projected = cam.project(Vector3(points))
        unprojected = cam.unproject(projected, points[..., 2])
        self.assert_close(points, unprojected.data)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_matrix(self, device, dtype, batch_size):
        params, image_size = self._make_rand_data(batch_size, device, dtype)
        cam = CameraModel(image_size, CameraModelType.PINHOLE, params)
        z = torch.zeros(batch_size, dtype=dtype, device=device)
        o = torch.ones(batch_size, dtype=dtype, device=device)
        K = torch.stack([params[:, 0], z, params[:, 2], z, params[:, 1], params[:, 3], z, z, o], dim=1).reshape(
            batch_size, 3, 3
        )
        self.assert_close(cam.matrix(), K)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_properties(self, device, dtype, batch_size):
        params, image_size = self._make_rand_data(batch_size, device, dtype)
        cam = CameraModel(image_size, CameraModelType.PINHOLE, params)
        self.assert_close(cam.fx, params[:, 0])
        self.assert_close(cam.fy, params[:, 1])
        self.assert_close(cam.cx, params[:, 2])
        self.assert_close(cam.cy, params[:, 3])
        self.assert_close(cam.width, image_size.width)
        self.assert_close(cam.height, image_size.height)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_scale(self, device, dtype, batch_size):
        params, image_size = self._make_rand_data(batch_size, device, dtype)
        cam = CameraModel(image_size, CameraModelType.PINHOLE, params)
        scale = torch.rand(batch_size, device=device, dtype=dtype)
        scaled_cam = cam.scale(scale)
        self.assert_close(cam.fx * scale, scaled_cam.fx)
        self.assert_close(cam.fy * scale, scaled_cam.fy)
        self.assert_close(cam.cx * scale, scaled_cam.cx)
        self.assert_close(cam.cy * scale, scaled_cam.cy)
        self.assert_close(cam.width * scale, scaled_cam.width)
        self.assert_close(cam.height * scale, scaled_cam.height)

    def test_convention_projection_matches_geometry_camera_project_points(self, device, dtype):
        # Convention pin (audit labels 5d-sc-01, 5d-sc-02; duplication-ledger row "geometry.camera /
        # sensors.camera", KEEP SEPARATE, kornia#4274): kornia ships two camera type systems and their Pinhole
        # paths share the mathematical mapping through different types. On the shared-intrinsics,
        # exactly representable z=4 fixture below, ``PinholeModel.project`` returns a Vector2 whose data is
        # byte-identical to ``project_points`` with the same K; general depths can differ through rounding.
        # A raw Tensor is rejected with AttributeError rather than accepted (the request in the closed #2708).
        # fx = 100 != fy = 50, cx = 4 != cy = 3 and the point is off-axis, so a transposed reading of the
        # parameter vector moves both components; the second arm changes ONE parameter (fy 50 -> 100) and the
        # v coordinate alone moves, which is what fixes fy as the y-axis scale rather than a shared focal.
        # Snippet used to generate expected: CameraModel(ImageSize(6, 8), CameraModelType.PINHOLE,
        # tensor([100., 50., 4., 3.])).project(Vector3(tensor([[1., 2., 4.]]))).data executed 2026-09-06 on
        # this worktree (torch 2.14.0) -> [[29.0, 28.0]] with torch.equal against project_points True, on cpu
        # for float32, float64, float16 and bfloat16 and on mps for float32 and float16. With fy = 100 the
        # same call gives the audit's [[29.0, 53.0]].
        cam = _pinhole(device, dtype)
        points = torch.tensor([[1.0, 2.0, 4.0]], device=device, dtype=dtype)
        projected = cam.project(Vector3(points))
        assert isinstance(projected, Vector2)
        self.assert_close(projected.data, torch.tensor([[29.0, 28.0]], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        assert torch.equal(projected.data, project_points(points, _k3(device, dtype)))
        symmetric = (100.0, 100.0, 4.0, 3.0)
        square = _pinhole(device, dtype, symmetric).project(Vector3(points))
        self.assert_close(square.data, torch.tensor([[29.0, 53.0]], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        assert torch.equal(square.data, project_points(points, _k3(device, dtype, symmetric)))
        with pytest.raises(AttributeError, match="has no attribute 'z'"):
            cam.project(points)

        # A non-power-of-two depth exposes direct division versus reciprocal multiplication. On CPU
        # float32 at 3ced4c71 (torch 2.14.0), x is 170.66665649414062 here versus 170.6666717529297
        # in project_points. Numerical agreement is approximate; bit identity is not the contract.
        points = torch.tensor([[5.0, 2.0, 3.0]], device=device, dtype=dtype)
        projected = cam.project(Vector3(points)).data
        expected = torch.tensor([[512.0 / 3.0, 109.0 / 3.0]], device=device, dtype=dtype)
        self.assert_close(projected, expected)
        self.assert_close(projected, project_points(points, _k3(device, dtype)))

    @pytest.mark.parametrize("z", [0.0, 1e-9, -1e-9, 1e-8, -1e-8])
    def test_wart_projection_differs_from_geometry_at_small_depth_4267(self, device, dtype, z):
        # Pin the current #4267 policy difference, including the threshold boundary. Retire or update
        # this pin when that policy is repaired. Geometry skips division at abs(z) <= 1e-8; sensors
        # divides unconditionally. In float16 these depths underflow to zero, so expect infinities.
        cam = _pinhole(device, dtype)
        points = torch.tensor([[1.0, 2.0, z]], device=device, dtype=dtype)
        projected = cam.project(Vector3(points)).data
        geometry = project_points(points, _k3(device, dtype))
        expected_geometry = torch.tensor([[104.0, 103.0]], device=device, dtype=dtype)
        self.assert_close(geometry, expected_geometry, atol=0.0, rtol=0.0)
        assert not torch.equal(projected, geometry)
        if points[0, 2] == 0:
            assert torch.isinf(projected).all()
        else:
            assert torch.isfinite(projected).all()
            assert (projected.abs() > 1e9).all()
            assert (projected.sign() == points[0, 2].sign()).all()

    def test_convention_unproject_takes_the_camera_frame_z_as_depth(self, device, dtype):
        # Convention pin (audit labels 5d-sc-03, 5d-sc-40; pre-finding P6): the ``depth`` argument of
        # ``CameraModelBase.unproject`` is the camera-frame z, not a ray length -- it multiplies the z = 1
        # point, so the third component of the result IS the depth that was passed in. On this shared-intrinsics
        # fixture, the result is byte-identical to ``kornia.geometry.camera.unproject_points`` (whose ``normalize``
        # flag would give the ray-length reading instead, and which takes depth as (B, 1) where the sensors API takes
        # it as (B,)). The round trip through project() is not the identity map on the fixture: the projected
        # pixel [[29.0, 28.0]] differs from the first two coordinates [[1.0, 2.0]] of the input, so the
        # equality below is not vacuously true of any pair of inverse functions.
        # Snippet used to generate expected: cam.unproject(Vector2(tensor([[29., 28.]])), tensor([2.])).data
        # and cam.unproject(cam.project(Vector3(X)), X[..., 2]).data executed 2026-09-06 on this worktree
        # (torch 2.14.0) -> [[0.5, 1.0, 2.0]] and [[1.0, 2.0, 4.0]], both torch.equal against
        # unproject_points / against the input, on cpu for float32, float64, float16 and bfloat16 and on mps
        # for float32 and float16.
        cam = _pinhole(device, dtype)
        pixels = torch.tensor([[29.0, 28.0]], device=device, dtype=dtype)
        depth = torch.tensor([2.0], device=device, dtype=dtype)
        unprojected = cam.unproject(Vector2(pixels), depth)
        assert isinstance(unprojected, Vector3)
        expected = torch.tensor([[0.5, 1.0, 2.0]], device=device, dtype=dtype)
        self.assert_close(unprojected.data, expected, atol=0.0, rtol=0.0)
        assert torch.equal(unprojected.data, unproject_points(pixels, depth[:, None], _k3(device, dtype)))
        points = torch.tensor([[1.0, 2.0, 4.0]], device=device, dtype=dtype)
        projected = cam.project(Vector3(points))
        assert not torch.equal(projected.data, points[..., :2])
        assert torch.equal(cam.unproject(projected, points[..., 2]).data, points)

    @pytest.mark.parametrize("shared_intrinsics", [True, False])
    def test_convention_shared_or_paired_intrinsics_match_geometry(self, device, dtype, shared_intrinsics):
        # Shared intrinsics support a point cloud; batched intrinsics agree for one point per camera.
        # These asymmetric values keep both projection and unprojection exactly representable in all dtypes.
        params = torch.tensor([[8.0, 4.0, 1.0, 2.0], [4.0, 8.0, 3.0, 4.0]], device=device, dtype=dtype)
        point_shape = (2, 3) if shared_intrinsics else (2,)
        cam = CameraModel(ImageSize(6, 8), CameraModelType.PINHOLE, params[0] if shared_intrinsics else params)
        points = torch.tensor([1.0, 2.0, 4.0], device=device, dtype=dtype).expand(*point_shape, 3)
        pixels = torch.tensor([9.0, 10.0], device=device, dtype=dtype).expand(*point_shape, 2)
        depth = torch.ones(point_shape, device=device, dtype=dtype)
        expected_projected = torch.tensor([[3.0, 4.0], [4.0, 8.0]], device=device, dtype=dtype)
        expected_unprojected = torch.tensor([[1.0, 2.0, 1.0], [1.5, 0.75, 1.0]], device=device, dtype=dtype)
        if shared_intrinsics:
            expected_projected = expected_projected[0].expand(*point_shape, 2)
            expected_unprojected = expected_unprojected[0].expand(*point_shape, 3)
        projected = cam.project(Vector3(points)).data
        unprojected = cam.unproject(Vector2(pixels), depth).data
        self.assert_close(projected, expected_projected, atol=0.0, rtol=0.0)
        self.assert_close(unprojected, expected_unprojected, atol=0.0, rtol=0.0)
        self.assert_close(projected, project_points(points, cam.matrix()), atol=0.0, rtol=0.0)
        self.assert_close(unprojected, unproject_points(pixels, depth[..., None], cam.matrix()), atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("num_points", [2, 3])
    def test_wart_batched_intrinsics_differ_from_geometry_4274(self, device, dtype, num_points):
        # The two camera APIs discussed in #4274 also differ in broadcasting (PR #4318 review).
        # Sensors aligns (B,) intrinsic components with the trailing axis of (B, N) coordinates;
        # geometry inserts a singleton point axis. With B == N this silently changes camera associations;
        # B = 2, N = 3 instead raises. Pin CURRENT behavior; retire/update if this distinction is repaired.
        # Hand-computed at z=4: camera 0 projects [1, 2, 4] to [3, 4], camera 1 to [4, 8].
        # At unit depth, [9, 10] unprojects to [1, 2, 1] / [1.5, .75, 1], respectively.
        params = torch.tensor([[8.0, 4.0, 1.0, 2.0], [4.0, 8.0, 3.0, 4.0]], device=device, dtype=dtype)
        cam = CameraModel(ImageSize(6, 8), CameraModelType.PINHOLE, params)
        points = torch.tensor([1.0, 2.0, 4.0], device=device, dtype=dtype).expand(2, num_points, 3)
        pixels = torch.tensor([9.0, 10.0], device=device, dtype=dtype).expand(2, num_points, 2)
        depth = torch.ones(2, num_points, device=device, dtype=dtype)
        per_camera_projected = torch.tensor([[3.0, 4.0], [4.0, 8.0]], device=device, dtype=dtype)
        per_camera_unprojected = torch.tensor([[1.0, 2.0, 1.0], [1.5, 0.75, 1.0]], device=device, dtype=dtype)
        geometry_projected = project_points(points, cam.matrix())
        geometry_unprojected = unproject_points(pixels, depth[..., None], cam.matrix())
        self.assert_close(
            geometry_projected, per_camera_projected[:, None].expand(2, num_points, 2), atol=0.0, rtol=0.0
        )
        self.assert_close(
            geometry_unprojected, per_camera_unprojected[:, None].expand(2, num_points, 3), atol=0.0, rtol=0.0
        )
        if num_points == 2:
            projected = cam.project(Vector3(points)).data
            unprojected = cam.unproject(Vector2(pixels), depth).data
            self.assert_close(projected, per_camera_projected[None].expand(2, 2, 2), atol=0.0, rtol=0.0)
            self.assert_close(unprojected, per_camera_unprojected[None].expand(2, 2, 3), atol=0.0, rtol=0.0)
            assert not torch.equal(projected, geometry_projected)
            assert not torch.equal(unprojected, geometry_unprojected)
        else:
            with pytest.raises(RuntimeError, match="size of tensor"):
                cam.project(Vector3(points))
            with pytest.raises(RuntimeError, match="size of tensor"):
                cam.unproject(Vector2(pixels), depth)

    def test_convention_matrix_is_the_three_by_three_intrinsics(self, device, dtype):
        # Convention pin (audit labels 5d-sc-04, 5d-sc-05): ``matrix()`` returns the 3x3 pinhole intrinsics
        # [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] -- the same K that ``kornia.geometry.camera`` takes as an
        # argument, byte-identical, and NOT the 4x4 layout ``kornia.geometry.camera.PinholeCamera`` stores.
        # ``K()`` is an alias of ``matrix()``. Unbatched (4,) params give (3, 3) and (1, 4) params give
        # (1, 3, 3), so the batch axis of the parameters is carried through. fx != fy and cx != cy, so the
        # transpose of this matrix is a different matrix and torch.equal discriminates it.
        # Snippet used to generate expected: cam.matrix() executed 2026-09-06 on this worktree (torch 2.14.0)
        # -> [[100.0, 0.0, 4.0], [0.0, 50.0, 3.0], [0.0, 0.0, 1.0]] with torch.equal against the hand-built K
        # True and shapes (3, 3) / (1, 3, 3), on cpu for float32, float64, float16 and bfloat16 and on mps for
        # float32 and float16.
        cam = _pinhole(device, dtype)
        expected = _k3(device, dtype)
        assert cam.matrix().shape == (3, 3)
        assert torch.equal(cam.matrix(), expected[0])
        assert torch.equal(cam.K(), cam.matrix())
        assert not torch.equal(cam.matrix(), expected[0].transpose(-2, -1))
        batched = CameraModel(
            ImageSize(6, 8), CameraModelType.PINHOLE, torch.tensor([_PARAMS], device=device, dtype=dtype)
        )
        assert batched.matrix().shape == (1, 3, 3)
        assert torch.equal(batched.matrix(), expected)

    def test_wart_scale_rescales_the_principal_point_by_the_half_pixel_rule_4263(self, device, dtype):
        # Wart pin for kornia#4263 (audit labels 5d-sc-06, 5d-sc-07, 5a-al-16; pre-finding P1):
        # ``PinholeModel.scale(s)`` gives cx' = s * cx (2.0 for cx = 4, s = 0.5), the half-pixel / COLMAP
        # rule, although every pixel grid in the library enumerates integer pixel CENTRES, under which the
        # grid-consistent value is s * cx + (s - 1) / 2 = 1.75 for cx and 1.25 for cy. This is one of the four
        # sites #4263 lists; ``PinholeCamera.scale``, ``PinholeCamera.scale_`` and ``scale_pinhole`` apply the
        # same rule and are pinned by the 5a PR (#4294) in tests/geometry/camera/test_pinhole.py -- those
        # pins are not on this branch. cx = 4 != cy = 3 and s = 0.5 != 1, so the two candidate rules differ
        # in both components and by different amounts.
        # Snippet used to generate expected: cam.scale(tensor(0.5)).params executed 2026-09-06 on this
        # worktree (torch 2.14.0) -> [50.0, 25.0, 2.0, 1.5], and the integer-centre rule 0.5 * 4 - 0.25 = 1.75
        # / 0.5 * 3 - 0.25 = 1.25; on cpu for float32, float64, float16 and bfloat16 and on mps for float32
        # and float16.
        # Pins the CURRENT rescale rule; NOT a contract; delete when #4263 is repaired.
        cam = _pinhole(device, dtype)
        scaled = cam.scale(torch.tensor(0.5, device=device, dtype=dtype))
        expected = torch.tensor([50.0, 25.0, 2.0, 1.5], device=device, dtype=dtype)
        self.assert_close(scaled.params, expected, atol=0.0, rtol=0.0)
        integer_centre = torch.tensor([50.0, 25.0, 1.75, 1.25], device=device, dtype=dtype)
        assert not torch.equal(scaled.params, integer_centre)

    def test_wart_scale_turns_the_image_size_fields_into_tensors_4263(self, device, dtype):
        # Wart pin for kornia#4263 (audit label 5d-sc-38): ``PinholeModel.scale`` rebuilds the ImageSize as
        # ``ImageSize(height * scale_factor, width * scale_factor)``, so the python ``int`` height and width
        # that the constructor accepted come back as 0-dim floating tensors on the model's device -- and 3.0
        # / 4.0 rather than the integer pixel counts an ImageSize is meant to hold.  #4263's body covers only
        # the principal-point rule of the same method; #4263 records THIS observation in its comment thread
        # (issuecomment-5556356093), where it is stated as a question the repair of that method has to answer
        # -- whether ImageSize keeps python ints -- because both of #4263's Expected outcomes change only
        # what cx' is.  So the pin is named for #4263 and is deleted with it, like the sibling pin above.
        # Snippet used to generate expected: (type(cam.image_size.height).__name__,
        # type(cam.scale(tensor(0.5)).image_size.height).__name__, cam.scale(tensor(0.5)).image_size) executed
        # 2026-09-06 on this worktree (torch 2.14.0, cpu float32) -> ('int', 'Tensor',
        # ImageSize(height=tensor(3.), width=tensor(4.))); on mps the two fields carry device='mps:0'.
        # Pins the CURRENT types and values; NOT a contract; delete when #4263 is repaired.
        cam = _pinhole(device, dtype)
        assert isinstance(cam.image_size.height, int)
        assert isinstance(cam.image_size.width, int)
        scaled = cam.scale(torch.tensor(0.5, device=device, dtype=dtype))
        assert isinstance(scaled.image_size.height, torch.Tensor)
        assert isinstance(scaled.image_size.width, torch.Tensor)
        assert scaled.image_size.height.shape == ()
        assert scaled.image_size.height.is_floating_point()
        assert scaled.image_size.height.device == cam.params.device
        self.assert_close(scaled.image_size.height, torch.tensor(3.0, device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(scaled.image_size.width, torch.tensor(4.0, device=device, dtype=dtype), atol=0.0, rtol=0.0)
        # The conversion is a property of the ARGUMENT TYPE, not of ``scale``: ``image_size`` is rebuilt by
        # multiplying the stored fields, so a python number multiplies python numbers and keeps them.  The
        # counterexample is in this method's own docstring, whose doctest calls ``cam.scale(2)``.  Without
        # these two arms the assertions above would read as "scale() turns the fields into tensors", which is
        # false.  Snippet used to generate expected: cam.scale(2).image_size and cam.scale(2.0).image_size
        # executed 2026-09-06 on this worktree (torch 2.14.0) -> ImageSize(height=12, width=16) with int
        # fields and ImageSize(height=12.0, width=16.0) with float fields, on cpu for float32, float64,
        # float16 and bfloat16 and on mps for float32 and float16.
        from_int = cam.scale(2)
        assert isinstance(from_int.image_size.height, int)
        assert isinstance(from_int.image_size.width, int)
        assert (from_int.image_size.height, from_int.image_size.width) == (12, 16)
        from_float = cam.scale(2.0)
        assert isinstance(from_float.image_size.height, float)
        assert isinstance(from_float.image_size.width, float)
        assert (from_float.image_size.height, from_float.image_size.width) == (12.0, 16.0)
        from_tensor = cam.scale(torch.tensor(2.0, device=device, dtype=dtype))
        assert isinstance(from_tensor.image_size.height, torch.Tensor)
        assert isinstance(from_tensor.image_size.width, torch.Tensor)


class TestCameraModelTypes(BaseTester):
    # The four CameraModelType members and the parameter-vector contract each one enforces.  Only PINHOLE is
    # usable; see test_wart_the_three_non_pinhole_models_construct_and_then_raise_4284 below.
    _LENGTHS = (
        (CameraModelType.PINHOLE, 4),
        (CameraModelType.BROWN_CONRADY, 12),
        (CameraModelType.KANNALA_BRANDT_K3, 8),
        (CameraModelType.ORTHOGRAPHIC, 4),
    )

    def test_convention_params_length_is_fixed_per_camera_model_type(self, device, dtype):
        # Convention pin (audit labels 5d-sc-12, 5d-sc-13, 5d-sc-14, 5d-sc-15, 5d-sc-17, 5d-sc-18, 5d-sc-20,
        # 5d-sc-21, 5d-sc-35): ``params`` is a (B, N) tensor whose LAST axis is fixed by the model type --
        # 4 for PINHOLE (fx, fy, cx, cy), 12 for BROWN_CONRADY, 8 for KANNALA_BRANDT_K3 and 4 for ORTHOGRAPHIC
        # -- and any other length raises ValueError.  The guard is ``params.shape[-1] != N or
        # len(params.shape) > 2``, so an unbatched (N,) vector and a batched (B, N) one are both accepted
        # while a (B, 1, N) one is rejected with the same message: there is no (B, N, 4) multi-camera form.
        # The rejected lengths are N - 1, one parameter away from the accepted one, rather than a wildly
        # wrong shape.  The four enum members and their values are part of the contract because
        # ``CameraModelType`` is what a caller passes.
        # Snippet used to generate expected: CameraModel(ImageSize(6, 8), t, ones(n)) / ones(2, n) /
        # ones(n - 1) / ones(1, 1, n) for each (t, n) executed 2026-09-06 on this worktree (torch 2.14.0) ->
        # shapes (n,) and (2, n) accepted, and ValueError "params must be of shape (B, 4) for PINHOLE Camera",
        # "params must be of shape (B, 12) for BROWN_CONRADY Camera", "params must be of shape B, 8 for
        # KANNALA_BRANDT_K3 Camera", "params must be of shape B, 4 for ORTHOGRAPHIC Camera" for both rejected
        # shapes; on cpu for float32, float64, float16 and bfloat16 and on mps for float32 and float16.
        assert [(m.name, m.value) for m in CameraModelType] == [
            ("PINHOLE", 0),
            ("BROWN_CONRADY", 1),
            ("KANNALA_BRANDT_K3", 2),
            ("ORTHOGRAPHIC", 3),
        ]
        for model_type, length in self._LENGTHS:
            unbatched = CameraModel(ImageSize(6, 8), model_type, torch.ones(length, device=device, dtype=dtype))
            assert unbatched.params.shape == (length,)
            batched = CameraModel(ImageSize(6, 8), model_type, torch.ones(2, length, device=device, dtype=dtype))
            assert batched.params.shape == (2, length)
            with pytest.raises(ValueError, match="params must be of shape"):
                CameraModel(ImageSize(6, 8), model_type, torch.ones(length - 1, device=device, dtype=dtype))
            with pytest.raises(ValueError, match="params must be of shape"):
                CameraModel(ImageSize(6, 8), model_type, torch.ones(1, 1, length, device=device, dtype=dtype))
        with pytest.raises(ValueError, match=r"params must be of shape \(B, 4\) for PINHOLE Camera"):
            CameraModel(ImageSize(6, 8), CameraModelType.PINHOLE, torch.ones(1, 1, 4, device=device, dtype=dtype))
        with pytest.raises(ValueError, match=r"params must be of shape B, 8 for KANNALA_BRANDT_K3 Camera"):
            CameraModel(ImageSize(6, 8), CameraModelType.KANNALA_BRANDT_K3, torch.ones(7, device=device, dtype=dtype))

    def test_wart_the_three_non_pinhole_models_construct_and_then_raise_4284(self, device, dtype):
        # Wart pin for kornia#4284 (audit labels 5d-sc-14, 5d-sc-16, 5d-sc-17, 5d-sc-19, 5d-sc-20, 5d-sc-22,
        # 5d-sc-23): BROWN_CONRADY, KANNALA_BRANDT_K3 and ORTHOGRAPHIC are exported from
        # ``kornia.sensors.camera.__all__``, validate their parameter vectors and construct without complaint
        # -- and then project, unproject and matrix all raise NotImplementedError with an EMPTY message,
        # from THREE different kinds of site.  Measured raise sites, as the qualified function name of the
        # last frame of each traceback -- ``traceback.extract_tb(exc.__traceback__)[-1]`` -- executed
        # 2026-09-06 on this worktree (torch 2.14.0, cpu float32).  Names rather than line numbers, because
        # a line number in a comment rots the next time either module is edited:
        #   BROWN_CONRADY     project   -> BrownConradyTransform.distort
        #                     unproject -> BrownConradyTransform.undistort
        #   KANNALA_BRANDT_K3 project   -> KannalaBrandtK3Transform.distort
        #                     unproject -> KannalaBrandtK3Transform.undistort
        #   ORTHOGRAPHIC      project   -> OrthographicProjection.project
        #                     unproject -> OrthographicProjection.unproject
        #   all three         matrix    -> CameraModelBase.matrix
        # So the two failure modes of project/unproject are a distortion placeholder (BROWN_CONRADY and
        # KANNALA_BRANDT_K3, which wire up the working Z1Projection and fail in the distortion) and a
        # projection placeholder (ORTHOGRAPHIC, in BOTH directions -- its AffineTransform never fails); those
        # placeholders are pinned at their own level in tests/sensors/camera/test_distortion_model.py and
        # test_projection_model.py.  ``matrix()`` is a THIRD, independent site: the three classes do not
        # override ``CameraModelBase.matrix``, which is itself a bare raise, so implementing the distortions
        # and the orthographic projection (#4284's Expected option 1) would leave ``matrix()`` raising until
        # each class grows its own override the way PinholeModel already has.
        # ``project``/``unproject`` are ``CameraModelBase``'s -- the three classes add no overrides -- so this
        # pins the base class's behaviour on those models too.
        # The empty message is asserted rather than described, because #4284's Expected asks at minimum for a
        # message naming the model: a message-only partial fix must flip these pins.
        # Snippet used to generate expected: project(Vector3([[1., 2., 4.]])) / unproject(Vector2([[0.5,
        # 0.25]]), tensor([2.])) / matrix() on each of the three models executed 2026-09-06 on this worktree
        # (torch 2.14.0) -> NotImplementedError('') for all nine calls, on cpu for float32, float64, float16
        # and bfloat16 and on mps for float32 and float16.
        # Pins the CURRENT behaviour; NOT a contract; delete when #4284 is repaired.
        point3 = Vector3(torch.tensor([[1.0, 2.0, 4.0]], device=device, dtype=dtype))
        point2 = Vector2(torch.tensor([[0.5, 0.25]], device=device, dtype=dtype))
        depth = torch.tensor([2.0], device=device, dtype=dtype)
        for model_type, length in self._LENGTHS[1:]:
            cam = CameraModel(ImageSize(6, 8), model_type, torch.ones(length, device=device, dtype=dtype))
            assert cam.params.shape == (length,)
            for call, args in ((cam.project, (point3,)), (cam.unproject, (point2, depth)), (cam.matrix, ())):
                with pytest.raises(NotImplementedError) as raised:
                    call(*args)
                assert str(raised.value) == ""
        pinhole = CameraModel(ImageSize(6, 8), CameraModelType.PINHOLE, torch.ones(4, device=device, dtype=dtype))
        assert isinstance(pinhole.project(point3), Vector2)
        assert isinstance(pinhole.matrix(), torch.Tensor)


class TestCameraModelBaseParamsValidation:
    """`CameraModelBase` is public and documents a `params` shape it did not check.

    The typed subclasses each apply the same two comparisons, so only the direct
    construction path was unguarded -- and that is the path the class docstring's
    own example uses.
    """

    image_size = ImageSize(6, 8)

    def test_a_short_params_vector_is_rejected_at_construction(self, device, dtype):
        # Was: constructed fine, then raised IndexError from inside
        # AffineTransform.distort, naming neither params nor the camera model.
        params = torch.ones(3, device=device, dtype=dtype)
        with pytest.raises(ValueError, match=r"shape \(B, 4\) or \(4,\) for AffineTransform"):
            CameraModelBase(AffineTransform(), Z1Projection(), self.image_size, params)

    def test_a_rank_three_params_tensor_is_rejected_at_construction(self, device, dtype):
        # Was: constructed, projected without complaint, and silently returned a
        # (1, 1, 2) result. There is no (B, N, 4) multi-camera form, which is why
        # the typed constructors reject exactly this shape.
        params = torch.ones(1, 1, 4, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="rank 1 or 2"):
            CameraModelBase(AffineTransform(), Z1Projection(), self.image_size, params)

    @pytest.mark.parametrize(
        ("distortion", "length"),
        [
            (AffineTransform, 4),
            (BrownConradyTransform, 12),
            (KannalaBrandtK3Transform, 8),
        ],
    )
    def test_the_required_length_follows_the_distortion_model(self, distortion, length, device, dtype):
        """The same lengths the four typed constructors each spell out."""
        ok = torch.ones(length, device=device, dtype=dtype)
        CameraModelBase(distortion(), Z1Projection(), self.image_size, ok)

        wrong = torch.ones(length + 1, device=device, dtype=dtype)
        with pytest.raises(ValueError, match=rf"shape \(B, {length}\)"):
            CameraModelBase(distortion(), Z1Projection(), self.image_size, wrong)

    @pytest.mark.parametrize("shape", [(4,), (1, 4), (5, 4)])
    def test_the_documented_shapes_still_construct_and_project(self, shape, device, dtype):
        """Unbatched and batched both stay valid; this is not a tightening of them."""
        params = torch.ones(*shape, device=device, dtype=dtype)
        cam = CameraModelBase(AffineTransform(), Z1Projection(), self.image_size, params)
        assert tuple(cam.params.shape) == shape
        point = Vector3(torch.tensor([[1.0, 2.0, 4.0]], device=device, dtype=dtype))
        assert cam.project(point).data.shape[-1] == 2

    def test_orthographic_still_reports_its_own_message(self, device, dtype):
        """Orthographic validated after super().__init__, so the base now runs first.

        Moving its guard above the super() call keeps its specific message, and
        stops it half-building the object before rejecting the arguments.
        """
        params = torch.ones(5, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="ORTHOGRAPHIC"):
            Orthographic(self.image_size, params)

    def test_the_class_docstring_example_is_consistent(self, device, dtype):
        """The example paired a Brown-Conrady transform with a 4-element affine vector."""
        params = torch.tensor([328.0, 328.0, 320.0, 240.0], device=device, dtype=dtype)
        cam = CameraModelBase(AffineTransform(), Z1Projection(), ImageSize(480, 640), params)
        assert cam.params.shape == params.shape

    def test_the_orthographic_pair_takes_four(self, device, dtype):
        """AffineTransform + OrthographicProjection is the other 4-parameter pair."""
        cam = CameraModelBase(
            AffineTransform(), OrthographicProjection(), self.image_size, torch.ones(4, device=device, dtype=dtype)
        )
        assert tuple(cam.params.shape) == (4,)
