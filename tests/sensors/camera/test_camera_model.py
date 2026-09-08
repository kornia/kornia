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

from kornia.geometry.vector import Vector3
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
