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

from testing.base import _DTYPE_PRECISIONS, BaseTester

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
        # unproject divides by fx / fy and scales by z, so a focal length or a depth drawn near
        # zero makes the half-precision round trip miss by far more than the dtype tolerance.
        # Keep both in [1, 2) (#4399).
        params[:, :2] += 1.0
        cam = CameraModel(image_size, CameraModelType.PINHOLE, params)
        points = torch.rand((batch_size, 3), device=device, dtype=dtype)
        points[..., 2] += 1.0
        projected = cam.project(Vector3(points))
        unprojected = cam.unproject(projected, points[..., 2])
        # Even there the round trip is not exact: u = fx * x / z + cx lands in [1, 4), where it is
        # rounded to one ULP, and unproject scales that rounding by z / fx < 2. The miss is bounded
        # by ULP(u) * z / fx, which reaches 2 * eps, while the half-precision default atol is about
        # 1 eps. Floor atol at 4 * eps so the verdict does not depend on the draw.
        rtol, atol = _DTYPE_PRECISIONS[dtype]
        self.assert_close(points, unprojected.data, rtol=rtol, atol=max(atol, 4 * torch.finfo(dtype).eps))

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
        # The two camera type systems (#4274) share the pinhole mapping through different types: PinholeModel
        # .project takes a Vector3 and agrees with project_points on the same K. fx != fy and cx != cy, and the
        # second arm changes fy alone so only v moves. A raw Tensor is rejected.
        cam = _pinhole(device, dtype)
        points = torch.tensor([[1.0, 2.0, 4.0]], device=device, dtype=dtype)
        projected = cam.project(Vector3(points))
        assert isinstance(projected, Vector2)
        self.assert_close(projected.data, torch.tensor([[29.0, 28.0]], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(projected.data, project_points(points, _k3(device, dtype)))
        symmetric = (100.0, 100.0, 4.0, 3.0)
        square = _pinhole(device, dtype, symmetric).project(Vector3(points))
        self.assert_close(square.data, torch.tensor([[29.0, 53.0]], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(square.data, project_points(points, _k3(device, dtype, symmetric)))
        with pytest.raises(AttributeError):
            cam.project(points)

        # A non-dyadic depth: agreement is to roundoff, not bit for bit.
        points = torch.tensor([[5.0, 2.0, 3.0]], device=device, dtype=dtype)
        projected = cam.project(Vector3(points)).data
        expected = torch.tensor([[512.0 / 3.0, 109.0 / 3.0]], device=device, dtype=dtype)
        self.assert_close(projected, expected)
        self.assert_close(projected, project_points(points, _k3(device, dtype)))

    @pytest.mark.parametrize("z", [0.0, 1e-9, -1e-9, 1e-8, -1e-8])
    def test_wart_projection_differs_from_geometry_at_small_depth_4267(self, device, dtype, z):
        # Wart pin for #4267: geometry skips the divide at abs(z) <= 1e-8, sensors divides unconditionally, so
        # the two disagree at and below that threshold (float16 underflows these depths to zero, giving inf).
        # Delete or update when #4267 is repaired.
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
        # depth is the camera-frame z (the result's third component), as in unproject_points without
        # normalize; the sensors API takes depth as (B,) where geometry takes (B, 1). The projected pixel differs
        # from the input's (x, y), so the round trip is not vacuous.
        cam = _pinhole(device, dtype)
        pixels = torch.tensor([[29.0, 28.0]], device=device, dtype=dtype)
        depth = torch.tensor([2.0], device=device, dtype=dtype)
        unprojected = cam.unproject(Vector2(pixels), depth)
        assert isinstance(unprojected, Vector3)
        expected = torch.tensor([[0.5, 1.0, 2.0]], device=device, dtype=dtype)
        self.assert_close(unprojected.data, expected, atol=0.0, rtol=0.0)
        self.assert_close(unprojected.data, unproject_points(pixels, depth[:, None], _k3(device, dtype)))
        points = torch.tensor([[1.0, 2.0, 4.0]], device=device, dtype=dtype)
        projected = cam.project(Vector3(points))
        assert not torch.equal(projected.data, points[..., :2])
        self.assert_close(cam.unproject(projected, points[..., 2]).data, points)

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

    @pytest.mark.parametrize("num_points", [1, 2, 3])
    def test_wart_batched_intrinsics_differ_from_geometry_4274(self, device, dtype, num_points):
        # Wart pin for #4274: sensors aligns (B,) intrinsics with the trailing axis of (B, N) points while geometry
        # inserts a point axis, so with B == N the cameras are silently reassociated, N == 1 projects to a (B, B, 2)
        # outer broadcast (unproject raises) and other N raise.
        # Hand-computed: camera 0 projects [1, 2, 4] to [3, 4], camera 1 to [4, 8]; at unit depth [9, 10]
        # unprojects to [1, 2, 1] / [1.5, .75, 1]. Delete or update when #4274 is repaired.
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
        elif num_points == 1:
            projected = cam.project(Vector3(points)).data
            self.assert_close(projected, per_camera_projected[None].expand(2, 2, 2), atol=0.0, rtol=0.0)
            with pytest.raises(RuntimeError):
                cam.unproject(Vector2(pixels), depth)
        else:
            with pytest.raises(RuntimeError):
                cam.project(Vector3(points))
            with pytest.raises(RuntimeError):
                cam.unproject(Vector2(pixels), depth)

    def test_convention_matrix_is_the_three_by_three_intrinsics(self, device, dtype):
        # matrix() (alias K()) is the 3x3 K that kornia.geometry.camera takes, not PinholeCamera's 4x4; the
        # params batch axis is carried through. fx != fy and cx != cy, so the transpose is distinguishable.
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
        # Wart pin for #4263: scale(s) gives cx' = s * cx (2.0 for cx = 4, s = 0.5), the half-pixel rule, while
        # kornia's integer pixel centres give s * cx + (s - 1) / 2 (1.75, and 1.25 for cy). PinholeCamera.scale
        # has the same wart (tests/geometry/camera/test_pinhole.py). Delete when #4263 is repaired.
        cam = _pinhole(device, dtype)
        scaled = cam.scale(torch.tensor(0.5, device=device, dtype=dtype))
        expected = torch.tensor([50.0, 25.0, 2.0, 1.5], device=device, dtype=dtype)
        self.assert_close(scaled.params, expected, atol=0.0, rtol=0.0)
        integer_centre = torch.tensor([50.0, 25.0, 1.75, 1.25], device=device, dtype=dtype)
        assert not torch.equal(scaled.params, integer_centre)

    def test_wart_scale_turns_the_image_size_fields_into_tensors_4263(self, device, dtype):
        # Wart pin for #4263 (raised in its comment thread): scale with a tensor factor turns the int ImageSize
        # fields into 0-dim floating tensors. Delete when #4263 is repaired.
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
        # The field type follows the scale factor's type: python numbers keep python numbers.
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
    # The four CameraModelType members and the parameter-vector length each one enforces.
    _LENGTHS = (
        (CameraModelType.PINHOLE, 4),
        (CameraModelType.BROWN_CONRADY, 12),
        (CameraModelType.KANNALA_BRANDT_K3, 8),
        (CameraModelType.ORTHOGRAPHIC, 4),
    )

    def test_convention_params_length_is_fixed_per_camera_model_type(self, device, dtype):
        # params is (N,) or (B, N) with N fixed by the model type (4 PINHOLE, 12 BROWN_CONRADY,
        # 8 KANNALA_BRANDT_K3, 4 ORTHOGRAPHIC); N - 1 and (B, 1, N) raise ValueError. There is no (B, N, 4) form.
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
        # The message names the model type.
        with pytest.raises(ValueError, match=r"params must be of shape .* for PINHOLE Camera"):
            CameraModel(ImageSize(6, 8), CameraModelType.PINHOLE, torch.ones(1, 1, 4, device=device, dtype=dtype))
        with pytest.raises(ValueError, match=r"params must be of shape .* for KANNALA_BRANDT_K3 Camera"):
            CameraModel(ImageSize(6, 8), CameraModelType.KANNALA_BRANDT_K3, torch.ones(7, device=device, dtype=dtype))


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


class TestNonPinholeCameraMatrix(BaseTester):
    @pytest.mark.parametrize(
        ("model_type", "num_params"),
        [
            (CameraModelType.BROWN_CONRADY, 12),
            (CameraModelType.KANNALA_BRANDT_K3, 8),
            (CameraModelType.ORTHOGRAPHIC, 4),
        ],
    )
    def test_matrix_unbatched(self, device, dtype, model_type, num_params):
        params = torch.zeros(num_params, device=device, dtype=dtype)
        params[:4] = torch.tensor(
            [300.0, 320.0, 160.0, 120.0],
            device=device,
            dtype=dtype,
        )

        cam = CameraModel(ImageSize(240, 320), model_type, params)

        expected = torch.tensor(
            [
                [300.0, 0.0, 160.0],
                [0.0, 320.0, 120.0],
                [0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )

        self.assert_close(cam.matrix(), expected)

    @pytest.mark.parametrize(
        ("model_type", "num_params"),
        [
            (CameraModelType.BROWN_CONRADY, 12),
            (CameraModelType.KANNALA_BRANDT_K3, 8),
            (CameraModelType.ORTHOGRAPHIC, 4),
        ],
    )
    def test_matrix_batched(self, device, dtype, model_type, num_params):
        params = torch.zeros(2, num_params, device=device, dtype=dtype)
        params[:, :4] = torch.tensor(
            [
                [300.0, 320.0, 160.0, 120.0],
                [280.0, 300.0, 150.0, 110.0],
            ],
            device=device,
            dtype=dtype,
        )

        cam = CameraModel(ImageSize(240, 320), model_type, params)

        expected = torch.tensor(
            [
                [
                    [300.0, 0.0, 160.0],
                    [0.0, 320.0, 120.0],
                    [0.0, 0.0, 1.0],
                ],
                [
                    [280.0, 0.0, 150.0],
                    [0.0, 300.0, 110.0],
                    [0.0, 0.0, 1.0],
                ],
            ],
            device=device,
            dtype=dtype,
        )

        self.assert_close(cam.matrix(), expected)


class TestNonPinholeCameraProjectUnproject(BaseTester):
    @pytest.mark.parametrize(
        ("model_type", "num_params"),
        [
            (CameraModelType.BROWN_CONRADY, 12),
            (CameraModelType.KANNALA_BRANDT_K3, 8),
            (CameraModelType.ORTHOGRAPHIC, 4),
        ],
    )
    def test_project_unproject_unbatched(self, device, dtype, model_type, num_params):
        params = torch.zeros(num_params, device=device, dtype=dtype)
        params[:4] = torch.tensor(
            [300.0, 320.0, 160.0, 120.0],
            device=device,
            dtype=dtype,
        )

        cam = CameraModel(ImageSize(240, 320), model_type, params)

        points = torch.tensor(
            [0.3, -0.2, 2.0],
            device=device,
            dtype=dtype,
        )

        projected = cam.project(Vector3(points))
        unprojected = cam.unproject(projected, points[..., 2])

        self.assert_close(unprojected.data, points)

    @pytest.mark.parametrize(
        ("model_type", "num_params"),
        [
            (CameraModelType.BROWN_CONRADY, 12),
            (CameraModelType.KANNALA_BRANDT_K3, 8),
            (CameraModelType.ORTHOGRAPHIC, 4),
        ],
    )
    def test_project_unproject_batched(self, device, dtype, model_type, num_params):
        params = torch.zeros(2, num_params, device=device, dtype=dtype)
        params[:, :4] = torch.tensor(
            [
                [300.0, 320.0, 160.0, 120.0],
                [280.0, 300.0, 150.0, 110.0],
            ],
            device=device,
            dtype=dtype,
        )

        cam = CameraModel(ImageSize(240, 320), model_type, params)

        points = torch.tensor(
            [
                [0.3, -0.2, 2.0],
                [-0.4, 0.25, 3.0],
            ],
            device=device,
            dtype=dtype,
        )

        projected = cam.project(Vector3(points))
        unprojected = cam.unproject(projected, points[..., 2])

        assert projected.data.shape == (2, 2)
        assert unprojected.data.shape == points.shape

        self.assert_close(unprojected.data, points)
