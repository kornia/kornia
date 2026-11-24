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

import torch

from kornia.geometry.calibration import stereo_rectify
from kornia.geometry.camera import StereoCamera
from kornia.geometry.transform import remap

from testing.base import BaseTester


class TestStereoRectify(BaseTester):
    @staticmethod
    def calibration(device, dtype):
        K = torch.tensor([[[20, 0, 3.5], [0, 16, 2.5], [0, 0, 1]]], device=device, dtype=dtype)
        d = torch.zeros((1, 4), device=device, dtype=dtype)
        R = torch.eye(3, device=device, dtype=dtype)[None]
        t = torch.tensor([[-0.12, 0.0, 0.0]], device=device, dtype=dtype)
        return K, K.clone(), d, d.clone(), R, t, K.clone()

    def test_convention_rectified_identity(self, device, dtype):
        args = self.calibration(device, dtype)
        out = stereo_rectify(*args, (6, 8))
        image = torch.arange(48, device=device, dtype=out.map_left.dtype).reshape(1, 1, 6, 8)
        for pixmap, valid in ((out.map_left, out.valid_left), (out.map_right, out.valid_right)):
            assert valid.all()
            sampled = remap(image, pixmap[..., 0], pixmap[..., 1], align_corners=True)
            self.assert_close(sampled, image)
        self.assert_close(out.projection_left[:, :, :3], args[0].to(out.map_left))
        focal = args[0][:, 0, 0].to(out.map_left)
        shift = args[5][:, 0].to(out.map_left)
        self.assert_close(out.projection_right[:, 0, 3], focal * shift)

    def test_convention_epipolar_and_disparity(self, device, dtype):
        K, Kr, d, dr, _, _, K_out = self.calibration(device, dtype)
        # A 90 degree rotation about the camera's optical axis and an oblique baseline.
        R = K.new_tensor([[[0, -1, 0], [1, 0, 0], [0, 0, 1]]])
        t = K.new_tensor([[0.03, -0.12, 0]])
        out = stereo_rectify(K, Kr, d, dr, R, t, K_out, (6, 8))
        xyz_left = out.map_left.new_tensor([[[0.02, 0.1, 2.0], [-0.08, 0.05, 4.0], [0.1, -0.2, 3.0]]])
        xyz_right = xyz_left @ R.to(xyz_left).mT + t.to(xyz_left)[:, None]
        q_left = (xyz_left @ out.rotation_left.mT) @ K_out.to(xyz_left).mT
        q_right = (xyz_right @ out.rotation_right.mT) @ K_out.to(xyz_left).mT
        q_left, q_right = q_left[..., :2] / q_left[..., 2:], q_right[..., :2] / q_right[..., 2:]
        disparity = q_left[..., 0] - q_right[..., 0]
        self.assert_close(q_left[..., 1], q_right[..., 1])
        assert (disparity > 0).all()
        stereo = StereoCamera(out.projection_left, out.projection_right)
        homogeneous = torch.cat((q_left, disparity[..., None], torch.ones_like(disparity[..., None])), -1)
        reconstruction = homogeneous @ stereo.Q.mT
        self.assert_close(reconstruction[..., :3] / reconstruction[..., 3:], xyz_left @ out.rotation_left.mT)

    def test_batch_and_degenerate_baseline(self, device, dtype):
        single = self.calibration(device, dtype)
        full = [x.repeat((2,) + (1,) * (x.ndim - 1)) for x in single]
        full[5][1] = 0
        expected = stereo_rectify(*single, (1, 8))
        both = stereo_rectify(*full, (1, 8))
        for x, y in zip(both, expected):
            self.assert_close(x[:1], y)
            assert not x[1].any()

    def test_convention_independent_lens_maps(self, device, dtype):
        K, Kr, d, dr, R, t, K_out = self.calibration(device, dtype)
        d[0, 0], dr[0, 0] = 0.25, -0.25
        out = stereo_rectify(K, Kr, d, dr, R, t, K_out, (6, 8))
        # At (7, 5), r² = ((7 - 3.5) / 20)² + ((5 - 2.5) / 16)².
        # Multiply the pixel offset from the centre by 1 ± r² / 4.
        gain = ((3.5 / 20) ** 2 + (2.5 / 16) ** 2) / 4
        delta = out.map_left.new_tensor([3.5, 2.5]) * gain
        centre = out.map_left.new_tensor([7, 5])
        self.assert_close(out.map_left[0, 5, 7], centre + delta)
        self.assert_close(out.map_right[0, 5, 7], centre - delta)
        assert out.valid_left.all()
        assert out.valid_right.all()

    def test_gradient_through_masked_distortion_pole(self, device, dtype):
        K = torch.eye(3, device=device, dtype=dtype)[None]
        d = K.new_tensor([[0, 0, 0, 0, 0, -1, 0, 0]]).requires_grad_()
        R = K.clone()
        t = K.new_tensor([[-0.125, 0, 0]])
        target = K.clone()
        target[:, 0, 0] = 2
        out = stereo_rectify(K, K, d, torch.zeros_like(d), R, t, target, (1, 3))
        # At output u=2, normalized x=1, so 1+k4*x^2=0. At u=1, x=1/2 with a valid denominator of 3/4.
        assert out.valid_left.tolist() == [[[True, True, False]]]
        assert not out.map_left[0, 0, 2].any()
        (grad,) = torch.autograd.grad(out.map_left.sum(), d)
        assert grad.isfinite().all()
        # d(u)/d(k1) = x^3 / (1+k4*x^2) = (1/8) / (3/4) for the single nonzero valid ray.
        self.assert_close(grad[0, 0], d.new_tensor(1 / 6))

    def test_gradient_through_masked_tilt_pole(self, device):
        K = torch.eye(3, device=device, dtype=torch.float32)[None]
        d = K.new_zeros(1, 14)
        d[0, 13] = -torch.pi / 4
        d.requires_grad_()
        t = K.new_tensor([[-0.125, 0, 0]])
        target = K.clone()
        target[:, 0, 0] = 2
        out = stereo_rectify(K, K, d, torch.zeros_like(d), K.clone(), t, target, (1, 3))
        # At tau_y = -pi/4, w = (1 - x)/sqrt(2) vanishes at x=1.
        assert out.valid_left.tolist() == [[[True, True, False]]]
        (grad,) = torch.autograd.grad(out.map_left.sum(), d)
        assert grad.isfinite().all()
        self.assert_close(grad[0, 13], d.new_tensor(-3 * 2**0.5))

    def test_geometry_uses_float32_for_half(self, device):
        args = self.calibration(device, torch.bfloat16)
        actual = stereo_rectify(*args, (6, 8))
        expect = stereo_rectify(*(x.float() for x in args), (6, 8))
        assert actual.map_left.dtype == torch.float32
        for x, y in zip(actual, expect):
            self.assert_close(x, y, rtol=0, atol=0)

    def test_gradcheck(self, device):
        K, Kr, d, dr, _, t, target = self.calibration(device, torch.float64)
        # Crop the field of view slightly so the validity mask stays constant under perturbation.
        target[:, :2, :2] *= 2
        t[:, 1] = 0.003
        angle = K.new_tensor(0.03)

        def maps(left, right, left_dist, right_dist, rotation_angle, translation, dest):
            cos_angle, sin_angle = rotation_angle.cos(), rotation_angle.sin()
            zero, one = torch.zeros_like(cos_angle), torch.ones_like(cos_angle)
            rotation = torch.stack((cos_angle, zero, sin_angle, zero, one, zero, -sin_angle, zero, cos_angle))
            output = stereo_rectify(
                left, right, left_dist, right_dist, rotation.reshape(1, 3, 3), translation, dest, (6, 8)
            )
            return output.map_left, output.map_right, output.projection_right

        self.gradcheck(maps, tuple(x.requires_grad_() for x in (K, Kr, d, dr, angle, t, target)))

    def test_dynamo(self, device, dtype, torch_optimizer):
        args = self.calibration(device, dtype)
        baseline = stereo_rectify(*args, (6, 8))
        actual = torch_optimizer(stereo_rectify)(*args, (6, 8))
        for x, y in zip(actual, baseline):
            self.assert_close(x, y)
