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

import math

import pytest
import torch

from kornia.geometry.conversions import euler_from_quaternion, rotation_matrix_to_quaternion
from kornia.geometry.liegroup import Se3, So3
from kornia.geometry.quaternion import Quaternion
from kornia.geometry.vector import Vector3

from testing.base import BaseTester


class TestSe3(BaseTester):
    def _make_rand_se3d(self, device, dtype, batch_size) -> Se3:
        q = Quaternion.random(batch_size, device, dtype)
        t = self._make_rand_data(device, dtype, batch_size, dims=3)
        return Se3(q, t)

    def _make_rand_se3d_vec(self, device, dtype, batch_size) -> Se3:
        q = Quaternion.random(batch_size, device, dtype)
        if batch_size is None:
            shape = ()
        else:
            shape = (batch_size,)
        t = Vector3.random(shape, device, dtype)
        return Se3(So3(q), t)

    def _make_rand_data(self, device, dtype, batch_size, dims):
        shape = [] if batch_size is None else [batch_size]
        return torch.rand([*shape, dims], device=device, dtype=dtype)

    def test_smoke(self, device, dtype):
        q = Quaternion.from_coeffs(1.0, 0.0, 0.0, 0.0)
        q = q.to(device, dtype)
        t = torch.rand(1, 3, device=device, dtype=dtype)
        s = Se3(So3(q), t)
        assert isinstance(s, Se3)
        assert isinstance(s.r, So3)
        self.assert_close(s.r.q.data, q.data)
        self.assert_close(s.t, t)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_cardinality(self, device, dtype, batch_size):
        se: Se3 = self._make_rand_se3d(device, dtype, batch_size)
        assert se.r.q.shape[0] == batch_size

    # TODO: implement me
    def test_exception(self, device, dtype):
        pass

    def test_gradcheck(self, device):
        v = torch.tensor([[1.0, 2.0, 3.0, 0.3, -0.4, 0.5]], device=device, dtype=torch.float64)
        self.gradcheck(lambda x: Se3.exp(x).matrix(), (v,))

    def test_gradient_is_finite_at_the_identity_4404(self, device, dtype):
        # #4404: at omega = 0 autograd walks V, whose sqrt has an unbounded derivative there and
        # whose closed-form terms divide by theta**2 and theta**3.
        # log has the same defect through its own theta, where clamp_min(1e-12) guards the value
        # and not the gradient (#4229) and underflows to 0 in float16 besides.
        v = torch.zeros(1, 6, device=device, dtype=dtype, requires_grad=True)
        Se3.exp(v).matrix().sum().backward()
        assert bool(torch.isfinite(v.grad).all()), v.grad

        data = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype, requires_grad=True)
        Se3(So3(Quaternion(data[:, :4])), data[:, 4:]).log().sum().backward()
        assert bool(torch.isfinite(data.grad).all()), data.grad

    def test_exp_matches_float64_above_40_rad_4965(self, device, dtype):
        # #4965: _so3_small_angle_coefficients divided by theta**3, which overflows float16 above 40.3 rad, so the
        # [omega]_x^2 term of V dropped out and the float16 exp(v).t was off by 0.18 from 41 rad. The reference
        # is the float64 path on the CPU, run on the same rounded input.
        eps = torch.finfo(dtype).eps
        axis = torch.tensor([0.48, 0.6, 0.64], dtype=torch.float64)
        for theta in (41.0, 60.0):
            v = torch.cat((torch.ones(3, dtype=torch.float64), theta * axis)).to(device=device, dtype=dtype)
            t_ref = Se3.exp(v.cpu().double()).t.to(device=device, dtype=dtype)
            self.assert_close(Se3.exp(v).t, t_ref, rtol=8 * eps, atol=8 * eps)

    def test_gradient_at_the_identity_couples_rotation_and_translation_4953(self, device, dtype):
        # #4953: exp fell back to t = upsilon and log to upsilon = t at omega = 0. The values were
        # right, V(0) = I, but the fallback did not depend on omega, so autograd returned
        # d t / d omega = 0 at the identity, the standard initialisation for pose optimisation.
        # The derivative of V(omega) upsilon = upsilon + 0.5 omega x upsilon + O(|omega|^2) is
        # -0.5 [upsilon]_x, and d upsilon / d q_vec of log at the identity is [t]_x.
        v = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
        jac = torch.autograd.functional.jacobian(lambda x: Se3.exp(x).t, v)[0, :, 0, :]
        upsilon = v[0, :3]
        self.assert_close(jac[:, :3], torch.eye(3, device=device, dtype=dtype))
        self.assert_close(jac[:, 3:], -0.5 * So3.hat(upsilon))
        qt = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]], device=device, dtype=dtype)
        jac = torch.autograd.functional.jacobian(lambda x: Se3(So3(Quaternion(x[:, :4])), x[:, 4:]).log(), qt)
        jac = jac[0, :, 0, :]
        self.assert_close(jac[:3, 4:], torch.eye(3, device=device, dtype=dtype))
        self.assert_close(jac[:3, 1:4], So3.hat(qt[0, 4:]))

    # TODO: implement me
    def test_jit(self, device, dtype):
        pass

    # TODO: implement me
    def test_module(self, device, dtype):
        pass

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_init(self, device, dtype, batch_size):
        s1: Se3 = self._make_rand_se3d(device, dtype, batch_size)
        s2 = Se3(s1.r, s1.t)
        assert isinstance(s2, Se3)
        self.assert_close(s1.r.q.data, s2.r.q.data)
        self.assert_close(s1.t, s2.t)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_getitem(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size, device, dtype)
        t = torch.rand(batch_size, 3, device=device, dtype=dtype)
        s = Se3(q, t)
        for i in range(batch_size):
            s1 = s[i]
            self.assert_close(s1.r.q.data, q.data[i])
            self.assert_close(s1.t, t[i])

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_mul(self, device, dtype, batch_size):
        s1 = Se3.identity(batch_size, device, dtype)
        s2: Se3 = self._make_rand_se3d(device, dtype, batch_size)
        s1s2 = s1 * s2
        s2s2inv = s2 * s2.inverse()
        zeros_vec = torch.zeros(3, device=device, dtype=dtype)
        if batch_size is not None:
            zeros_vec = zeros_vec.repeat(batch_size, 1)
        so3_expected = So3.identity(batch_size, device, dtype)
        self.assert_close(s1s2.r.q.data, s2.r.q.data)
        self.assert_close(s1s2.t, s2.t)
        self.assert_close(s2s2inv.r.q.data, so3_expected.q.data)
        self.assert_close(s2s2inv.t, zeros_vec)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_mul_point(self, device, dtype, batch_size):
        world_pose_s1: Se3 = self._make_rand_se3d(device, dtype, batch_size)
        world_pose_s2: Se3 = self._make_rand_se3d(device, dtype, batch_size)
        pt_in_world = self._make_rand_data(device, dtype, batch_size, dims=3)
        s1_pose_s2: Se3 = world_pose_s1.inverse() * world_pose_s2
        pt_in_s1 = world_pose_s1.inverse() * pt_in_world
        pt_in_s2 = world_pose_s2.inverse() * pt_in_world
        pt_in_s1_in_s2 = s1_pose_s2.inverse() * pt_in_s1
        pt_in_s2_in_s1 = s1_pose_s2 * pt_in_s2
        self.assert_close(pt_in_s1, pt_in_s2_in_s1)
        self.assert_close(pt_in_s2, pt_in_s1_in_s2)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_mul_vector(self, device, dtype, batch_size):
        world_pose_s1: Se3 = self._make_rand_se3d(device, dtype, batch_size)
        world_pose_s2: Se3 = self._make_rand_se3d_vec(device, dtype, batch_size)
        if batch_size is None:
            shape = ()
        else:
            shape = (batch_size,)
        pt_in_world = Vector3.random(shape, device, dtype)
        s1_pose_s2: Se3 = world_pose_s1.inverse() * world_pose_s2
        pt_in_s1 = world_pose_s1.inverse() * pt_in_world
        pt_in_s2 = world_pose_s2.inverse() * pt_in_world
        pt_in_s1_in_s2 = s1_pose_s2.inverse() * pt_in_s1
        pt_in_s2_in_s1 = s1_pose_s2 * pt_in_s2
        s3 = Se3.identity(batch_size, device, dtype)
        s4: Se3 = self._make_rand_se3d_vec(device, dtype, batch_size)
        s3s4 = s3 * s4
        s4s4inv = s4 * s4.inverse()
        zeros_vec = torch.zeros(3, device=device, dtype=dtype)
        if batch_size is not None:
            zeros_vec = zeros_vec.repeat(batch_size, 1)
        so3_expected = So3.identity(batch_size, device, dtype)
        self.assert_close(pt_in_s1, pt_in_s2_in_s1)
        self.assert_close(pt_in_s2, pt_in_s1_in_s2)
        self.assert_close(s3s4.r.q.data, s4.r.q.data)
        self.assert_close(s3s4.t, s4.t)
        self.assert_close(s4s4inv.r.q.data, so3_expected.q.data)
        self.assert_close(s4s4inv.t, zeros_vec)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_exp(self, device, dtype, batch_size):
        omega = torch.zeros(3, device=device, dtype=dtype)
        t = torch.rand(3, device=device, dtype=dtype)
        if batch_size is not None:
            omega = omega.repeat(batch_size, 1)
            t = t.repeat(batch_size, 1)
        s = Se3.exp(torch.cat((t, omega), -1))
        quat_expected = Quaternion.identity(batch_size, device, dtype)
        self.assert_close(s.r.q.data, quat_expected.data)
        self.assert_close(s.t, t)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_log(self, device, dtype, batch_size):
        q = Quaternion.identity(batch_size, device, dtype)
        t = self._make_rand_data(device, dtype, batch_size, dims=3)
        s = Se3(So3(q), t)
        zero_vec = torch.zeros(3, device=device, dtype=dtype)
        if batch_size is not None:
            zero_vec = zero_vec.repeat(batch_size, 1)
        self.assert_close(s.log(), torch.cat((t, zero_vec), -1))

    def test_log_is_principal_4925(self, device, dtype):
        # Se3.log inherits So3.log: a small rotation stored with real < 0 got |omega| close to 2 pi and, through
        # V_inv(omega), a translation that was six orders of magnitude off. Both signs now give the principal log.
        theta = {torch.bfloat16: 1e-1, torch.float16: 1e-2, torch.float32: 1e-4, torch.float64: 1e-6}[dtype]
        rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        axis = torch.tensor([[0.48, 0.6, 0.64]], dtype=torch.float64)  # unit length
        q = Quaternion.from_axis_angle(theta * axis).data.to(device=device, dtype=dtype)  # rounded once
        axis = axis.to(device=device, dtype=dtype)
        t = torch.tensor([[1.0, 2.0, 3.0]], device=device, dtype=dtype)
        xi = Se3(So3(Quaternion(-q)), t).log()
        self.assert_close(xi, Se3(So3(Quaternion(q)), t).log())
        self.assert_close(xi[..., 3:], theta * axis, rtol=rtol, atol=0.0)
        self.assert_close(xi[..., :3], t, rtol=0.0, atol=3.0 * theta)
        # every random pose logs to a principal rotation vector
        torch.manual_seed(0)
        omega = Se3.random(64, device=device, dtype=dtype).log()[..., 3:]
        assert bool((omega.norm(dim=-1) <= torch.pi * (1 + rtol)).all())

    def test_exp_log_keep_small_rotations(self, device, dtype):
        # V and V_inv are built from (1 - cos theta) / theta**2 and friends, which evaluate to exactly 0
        # for theta <= 1e-4 in float32 (1e-8 in float64) and lose most of their digits well above that,
        # so the rotation-coupled part of the translation vanished for small rotations.
        theta = {torch.bfloat16: 1e-1, torch.float16: 1e-2, torch.float32: 1e-4, torch.float64: 1e-8}[dtype]
        rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        v = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, theta]], device=device, dtype=dtype)
        t = Se3.exp(v).t
        # translating along x while turning by theta about z ends at ((sin theta) / theta, (1 - cos theta) / theta, 0)
        expected_y = torch.tensor(theta / 2 - theta**3 / 24, device=device, dtype=dtype)
        self.assert_close(t[0, 1], expected_y, rtol=rtol, atol=0.0)
        self.assert_close(t[0, 2], torch.zeros((), device=device, dtype=dtype))
        v = torch.tensor([[1.0, 2.0, 3.0, 0.6 * theta, 0.0, 0.8 * theta]], device=device, dtype=dtype)
        self.assert_close(Se3.exp(v).log(), v, rtol=rtol, atol=0.0)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_exp_log(self, device, dtype, batch_size):
        a = self._make_rand_data(device, dtype, batch_size, dims=6)
        b = Se3.exp(a).log()
        self.assert_close(b, a)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_hat_vee(self, device, dtype, batch_size):
        a = self._make_rand_data(device, dtype, batch_size, dims=6)
        omega_hat = Se3.hat(a)
        b = Se3.vee(omega_hat)
        self.assert_close(b, a)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_matrix(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size, device, dtype)
        t = self._make_rand_data(device, dtype, batch_size, dims=3)
        rot = So3(q)
        s = Se3(rot, t)
        rot_mat = s.matrix()
        assert rot_mat.shape[-2:] == (4, 4)
        if batch_size is not None:
            assert rot_mat.shape[0] == batch_size
        self.assert_close(rot_mat[..., 0:3, 0:3], rot.matrix())
        self.assert_close(rot_mat[..., 0:3, 3], t)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_from_matrix(self, device, dtype, batch_size):
        matrix = torch.tensor(
            ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, -1.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            device=device,
            dtype=dtype,
        )
        if batch_size is not None:
            matrix = matrix.repeat(batch_size, 1, 1)
        s = Se3.from_matrix(matrix)
        self.assert_close(s.r.matrix(), matrix[..., 0:3, 0:3])
        self.assert_close(s.t, matrix[..., 0:3, 3])

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_from_qxyz(self, device, dtype, batch_size):
        qxyz = self._make_rand_data(device, dtype, batch_size, dims=7)
        s = Se3.from_qxyz(qxyz)
        self.assert_close(s.r.q.data, qxyz[..., :4].data)
        self.assert_close(s.t, qxyz[..., 4:])

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_inverse(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size, device, dtype)
        rot = So3(q)
        t = self._make_rand_data(device, dtype, batch_size, dims=3)
        sinv = Se3(rot, t).inverse()
        self.assert_close(sinv.r.inverse().q.data, q.data)
        self.assert_close(sinv.t, sinv.r * (-1 * t))

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_rot_x(self, device, dtype, batch_size):
        x = self._make_rand_data(device, dtype, batch_size, dims=1).squeeze(-1)
        se3 = Se3.rot_x(x)
        quat = rotation_matrix_to_quaternion(se3.so3.matrix())
        quat = Quaternion(quat)
        roll, _, _ = euler_from_quaternion(*quat.coeffs)
        self.assert_close(x, roll)
        self.assert_close(se3.t, torch.zeros_like(se3.t))

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_rot_y(self, device, dtype, batch_size):
        y = self._make_rand_data(device, dtype, batch_size, dims=1).squeeze(-1)
        se3 = Se3.rot_y(y)
        quat = rotation_matrix_to_quaternion(se3.so3.matrix())
        quat = Quaternion(quat)
        _, pitch, _ = euler_from_quaternion(*quat.coeffs)
        self.assert_close(y, pitch)
        self.assert_close(se3.t, torch.zeros_like(se3.t))

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_rot_z(self, device, dtype, batch_size):
        z = self._make_rand_data(device, dtype, batch_size, dims=1).squeeze(-1)
        se3 = Se3.rot_z(z)
        quat = rotation_matrix_to_quaternion(se3.so3.matrix())
        quat = Quaternion(quat)
        _, _, yaw = euler_from_quaternion(*quat.coeffs)
        self.assert_close(z, yaw)
        self.assert_close(se3.t, torch.zeros_like(se3.t))

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_trans(self, device, dtype, batch_size):
        trans = self._make_rand_data(device, dtype, batch_size, dims=3)
        x, y, z = trans[..., 0], trans[..., 1], trans[..., 2]
        se3 = Se3.trans(x, y, z)
        self.assert_close(se3.t, trans)
        self.assert_close(se3.so3.matrix(), So3.identity(batch_size, device, dtype).matrix())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_trans_x(self, device, dtype, batch_size):
        x = self._make_rand_data(device, dtype, batch_size, dims=1).squeeze(-1)
        zs = torch.zeros_like(x)
        se3 = Se3.trans_x(x)
        self.assert_close(se3.t, torch.stack((x, zs, zs), -1))
        self.assert_close(se3.so3.matrix(), So3.identity(batch_size, device, dtype).matrix())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_trans_y(self, device, dtype, batch_size):
        y = self._make_rand_data(device, dtype, batch_size, dims=1).squeeze(-1)
        zs = torch.zeros_like(y)
        se3 = Se3.trans_y(y)
        self.assert_close(se3.t, torch.stack((zs, y, zs), -1))
        self.assert_close(se3.so3.matrix(), So3.identity(batch_size, device, dtype).matrix())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_trans_z(self, device, dtype, batch_size):
        z = self._make_rand_data(device, dtype, batch_size, dims=1).squeeze(-1)
        zs = torch.zeros_like(z)
        se3 = Se3.trans_z(z)
        self.assert_close(se3.t, torch.stack((zs, zs, z), -1))
        self.assert_close(se3.so3.matrix(), So3.identity(batch_size, device, dtype).matrix())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_adjoint(self, device, dtype, batch_size):
        shape = (6,) if batch_size is None else (batch_size, 6)
        x_data = torch.tensor([0.1, -0.2, 0.3, 0.2, -0.1, 0.1], device=device, dtype=dtype).expand(shape)
        y_data = torch.tensor([-0.2, 0.1, 0.2, -0.1, 0.2, 0.1], device=device, dtype=dtype).expand(shape)
        x = Se3.exp(x_data)
        y = Se3.exp(y_data)
        adjoint = x.adjoint()
        inverse_adjoint = x.inverse().adjoint()
        identity = torch.eye(adjoint.shape[-1], device=device, dtype=dtype).expand_as(adjoint)
        half_tolerance = torch.finfo(dtype).eps if dtype in (torch.float16, torch.bfloat16) else None
        self.assert_close(adjoint @ inverse_adjoint, identity, rtol=half_tolerance, atol=half_tolerance)
        self.assert_close((x * y).adjoint(), adjoint @ y.adjoint(), rtol=half_tolerance, atol=half_tolerance)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_random(self, device, dtype, batch_size):
        s = Se3.random(batch_size=batch_size, device=device, dtype=dtype)
        s_in_s = s.inverse() * s
        i = Se3.identity(batch_size=batch_size, device=device, dtype=dtype)
        self.assert_close(s_in_s.so3.q.data, i.so3.q.data)
        self.assert_close(s_in_s.t, i.t)

    def test_user_leaf_translation_receives_the_gradient(self, device, dtype):
        # A tensor that requires grad is kept, not re-wrapped as a new Parameter, so the gradient reaches it (#4943).
        t = torch.tensor([[1.0, 2.0, 3.0]], device=device, dtype=dtype, requires_grad=True)
        s = Se3(So3.identity(1, device, dtype), t)
        (s * torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype)).sum().backward()
        assert t.grad is not None
        self.assert_close(t.grad, torch.ones_like(t))
        assert "_translation" in s.state_dict()
        # a tensor that does not require grad still becomes an optimizable parameter
        plain = Se3(So3.identity(1, device, dtype), torch.zeros(1, 3, device=device, dtype=dtype))
        assert [name for name, _ in plain.named_parameters()] == ["_translation"]

    def test_derived_state_moves_and_serializes(self, device, dtype):
        v = torch.rand(2, 6, device=device, dtype=dtype, requires_grad=True)
        s = Se3.exp(v)
        assert s.t.grad_fn is not None
        assert "_translation" in s.state_dict()
        restored = Se3(So3.identity(2, device, dtype), torch.zeros(2, 3, device=device, dtype=dtype))
        assert list(restored.state_dict()) == list(s.state_dict()) == ["_translation"]
        restored.load_state_dict(s.state_dict())
        self.assert_close(restored.t, s.t.detach())
        other = torch.float16 if dtype == torch.float32 else torch.float32  # float64 is unavailable on MPS
        moved = s.to(other)
        assert moved.t.dtype == other and moved.t.grad_fn is not None
        moved.t.sum().backward()
        assert v.grad is not None

    def test_convention_se3_tangent_is_upsilon_omega_with_V(self, device, dtype):
        # The tangent is (upsilon, omega), translation part first: exp rotates by the rotation vector omega and
        # translates by V(omega) upsilon. Reference, float64: m = scipy.linalg.expm of
        # [[0, 0.3, 0.2, 1], [-0.3, 0, -0.4, -2], [-0.2, 0.4, 0, 3], [0, 0, 0, 0]]; t = m[:3, 3], R = m[:3, :3].
        v = torch.tensor([1.0, -2.0, 3.0, 0.4, 0.2, -0.3], device=device, dtype=dtype)
        g = Se3.exp(v)
        t_expected = torch.tensor(
            [0.8932266973410874, -2.666342656131567, 2.413407159033739], device=device, dtype=dtype
        )
        r_expected = torch.tensor(
            [
                [0.9365557269934556, 0.32475143364814213, 0.13190859175670216],
                [-0.24666617456316434, 0.8779917826797222, -0.4102270442977377],
                [-0.24903648038416887, 0.35166309998400436, 0.9023934261437778],
            ],
            device=device,
            dtype=dtype,
        )
        assert (t_expected - v[:3]).abs().max() > 0.1  # V is not the identity at this rotation
        self.assert_close(g.t, t_expected)
        self.assert_close(g.r.matrix(), r_expected)
        self.assert_close(g.r.log(), v[3:])

    def test_convention_se3_hat_layout(self, device, dtype):
        # hat(upsilon, omega) = [[hat(omega), upsilon], [0, 0]]: the cross-product matrix of omega in the top-left
        # block, upsilon in the last column, a zero bottom row. Its matrix exponential is exp(v).matrix().
        v = torch.tensor([1.0, -2.0, 3.0, 0.4, 0.2, -0.3], device=device, dtype=dtype)
        expected = torch.tensor(
            [[0.0, 0.3, 0.2, 1.0], [-0.3, 0.0, -0.4, -2.0], [-0.2, 0.4, 0.0, 3.0], [0.0, 0.0, 0.0, 0.0]],
            device=device,
            dtype=dtype,
        )
        m = Se3.hat(v)
        self.assert_close(m, expected)
        # matrix_exp has no half-precision kernel and MPS has no float64: take it in float64 on the CPU. Move first,
        # then cast: on torch 2.14 a single m.to("cpu", torch.float64) from MPS returns zeros.
        m_exp = torch.linalg.matrix_exp(m.cpu().double()).to(device=device, dtype=dtype)
        self.assert_close(m_exp, Se3.exp(v).matrix())

    def test_convention_se3_adjoint_conjugates_the_tangent(self, device, dtype):
        # g exp(v) = exp(Ad(g) v) g, with Ad(g) = [[R, hat(t) R], [0, R]] for the (upsilon, omega) tangent.
        g = Se3.exp(torch.tensor([1.0, -2.0, 3.0, 0.4, 0.2, -0.3], device=device, dtype=dtype))
        v = torch.tensor([0.3, 0.1, -0.2, -0.1, 0.3, 0.05], device=device, dtype=dtype)
        ad_v = (g.adjoint() @ v[:, None])[:, 0]
        assert (ad_v - v).abs().max() > 0.1  # g does not commute with exp(v)
        self.assert_close((g * Se3.exp(v)).matrix(), (Se3.exp(ad_v) * g).matrix())

    def test_convention_se3_composition_is_left_matrix_product(self, device, dtype):
        # a * b is the matrix product a b: b acts first on a point.
        a = Se3.exp(torch.tensor([1.0, -2.0, 3.0, 0.4, 0.2, -0.3], device=device, dtype=dtype))
        b = Se3.exp(torch.tensor([0.3, 0.1, -0.2, -0.1, 0.3, 0.05], device=device, dtype=dtype))
        ab, ba = a.matrix() @ b.matrix(), b.matrix() @ a.matrix()
        assert (ab - ba).abs().max() > 0.1  # a non-commuting pair
        self.assert_close((a * b).matrix(), ab)
        p = torch.tensor([0.7, -1.3, 2.1], device=device, dtype=dtype)
        self.assert_close((a * b) * p, a * (b * p))

    def test_convention_se3_from_qxyz_is_wxyz_then_xyz(self, device, dtype):
        # The seven numbers are the quaternion (w, x, y, z), then the translation (x, y, z).
        # Reference: 95 * scipy.spatial.transform.Rotation.from_quat(q).as_matrix(), which takes (x, y, z, w), for
        # q = (1, -3, 2, 9) (the wxyz reading) and q = (9, 1, -3, 2) (the xyzw reading).
        q = torch.tensor([9.0, 1.0, -3.0, 2.0], dtype=torch.float64) / math.sqrt(95.0)
        qxyz = torch.cat((q, torch.tensor([5.0, 6.0, 7.0], dtype=torch.float64))).to(device=device, dtype=dtype)
        wxyz = torch.tensor([[69.0, -42.0, -50.0], [30.0, 85.0, -30.0], [58.0, 6.0, 75.0]], device=device, dtype=dtype)
        xyzw = torch.tensor(
            [[75.0, 30.0, -50.0], [6.0, -85.0, -42.0], [-58.0, 30.0, -69.0]], device=device, dtype=dtype
        )
        assert (wxyz - xyzw).abs().max() > 50.0  # the two readings differ
        g = Se3.from_qxyz(qxyz)
        self.assert_close(g.r.matrix(), wxyz / 95.0)
        self.assert_close(g.t.data, torch.tensor([5.0, 6.0, 7.0], device=device, dtype=dtype))

    def test_convention_se3_from_matrix_ignores_the_bottom_row(self, device, dtype):
        # from_matrix reads the rotation block and the last column; it does not check the bottom row.
        clean = Se3.exp(torch.tensor([1.0, -2.0, 3.0, 0.4, 0.2, -0.3], device=device, dtype=dtype)).matrix().detach()
        junk = clean.clone()
        junk[3] = torch.tensor([0.5, -4.0, 7.0, 2.0], device=device, dtype=dtype)
        self.assert_close(Se3.from_matrix(junk).matrix(), clean)

    def test_wart_se3_translation_type_depends_on_constructor_4931(self, device, dtype):
        from_exp = Se3.exp(torch.zeros(1, 6, device=device, dtype=dtype))
        identity = Se3.identity(1, device, dtype)
        self.assert_close(identity.matrix(), from_exp.matrix())  # the same group element
        assert isinstance(from_exp.t, torch.Tensor)
        # https://github.com/kornia/kornia/issues/4931: identity stores its translation as a Vector3, which is not a
        # tensor, so tensor indexing of the translation fails for it and works for the exp-built element.
        assert isinstance(identity.t, Vector3)
        assert not isinstance(identity.t, torch.Tensor)
        self.assert_close(from_exp.t[..., 0], torch.zeros(1, device=device, dtype=dtype))
        with pytest.raises(RuntimeError):
            identity.t[..., 0]
        assert isinstance((from_exp * identity).t, Vector3)  # a product with the identity inherits it

    def test_wart_se3_load_state_dict_restores_translation_not_rotation_4923(self, device, dtype):
        src = Se3.exp(torch.tensor([[1.0, -2.0, 3.0, 0.4, 0.2, -0.3]], device=device, dtype=dtype))
        dst = Se3(Quaternion.identity(1, device, dtype), torch.zeros(1, 3, device=device, dtype=dtype))
        eye = torch.eye(3, device=device, dtype=dtype)[None]
        assert (src.r.matrix() - eye).abs().max() > 0.1  # the source rotation is not the identity
        # https://github.com/kornia/kornia/issues/4923: the quaternion is not registered, so the state dict holds
        # only the translation; loading reports every key matched, restores the translation and keeps the old
        # rotation.
        assert list(src.state_dict()) == ["_translation"]
        assert list(Se3.identity(1, device, dtype).state_dict()) == []  # a Vector3 translation is not saved either
        result = dst.load_state_dict(src.state_dict())
        assert not result.missing_keys and not result.unexpected_keys
        self.assert_close(dst.t, src.t.detach())
        self.assert_close(dst.r.matrix(), eye)
