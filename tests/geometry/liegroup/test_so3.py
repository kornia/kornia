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

from kornia.geometry.conversions import euler_from_quaternion
from kornia.geometry.liegroup import So3
from kornia.geometry.liegroup.so3 import _so3_small_angle_coefficients
from kornia.geometry.quaternion import Quaternion
from kornia.geometry.vector import Vector3

from testing.base import BaseTester


class TestSo3(BaseTester):
    def _make_rand_data(self, device, dtype, batch_size, dims):
        shape = [] if batch_size is None else [batch_size]
        return torch.rand([*shape, dims], device=device, dtype=dtype)

    def test_smoke(self, device, dtype):
        q = Quaternion.from_coeffs(1.0, 0.0, 0.0, 0.0)
        q = q.to(device, dtype)
        s = So3(q)
        assert isinstance(s, So3)
        self.assert_close(s.q.data, q.data)

    # TODO: implement me
    def test_cardinality(self, device, dtype):
        pass

    # TODO: implement me
    def test_exception(self, device, dtype):
        pass

    def test_gradcheck(self, device):
        v = torch.tensor([[0.3, -0.4, 0.5]], device=device, dtype=torch.float64)
        self.gradcheck(lambda x: So3.exp(x).matrix(), (v,))
        q = torch.tensor([[0.8, 0.2, -0.4, 0.4]], device=device, dtype=torch.float64)
        q = q / q.norm(dim=-1, keepdim=True)
        self.gradcheck(lambda x: So3(Quaternion(x)).log(), (q,))

    def test_gradient_is_finite_at_the_identity_4404(self, device, dtype):
        # #4404: exp's small-angle branch is selected at theta = 0, but torch.where differentiates
        # the branch it does not select too, and sin(theta / 2) / theta is a 0/0 there -- so
        # 0 * nan = nan used to reach every component of the gradient at the single most common
        # input, the identity that pose optimisation starts from. The forward was always correct.
        v = torch.zeros(1, 3, device=device, dtype=dtype, requires_grad=True)
        So3.exp(v).matrix().sum().backward()
        assert bool(torch.isfinite(v.grad).all()), v.grad

        # log is singular the same way at both ends: at the identity through sqrt, the division by
        # theta and acos(+-1), and at a half turn (real = 0) through the small-angle branch's own
        # division. Each is in the branch the other case selects.
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype, requires_grad=True)
        So3(Quaternion(identity)).log().sum().backward()
        assert bool(torch.isfinite(identity.grad).all()), identity.grad

        half_turn = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device, dtype=dtype, requires_grad=True)
        So3(Quaternion(half_turn)).log().sum().backward()
        assert bool(torch.isfinite(half_turn.grad).all()), half_turn.grad

    def test_log_is_principal_4925(self, device, dtype):
        # q and -q are the same rotation. log used 2 * acos(real), which for real < 0 returned the vector of
        # length 2 pi - theta about the negated axis, so the same matrix had two different logs and exp(v).log()
        # was not the principal vector for |v| > pi.
        # the quaternions are built in float64 and rounded once, so that exp's own rounding stays out of the test
        rtol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        v = torch.tensor([[0.5, 0.1, -0.3]], dtype=torch.float64)
        q = Quaternion.from_axis_angle(v).data.to(device=device, dtype=dtype)
        v = v.to(device=device, dtype=dtype)
        self.assert_close(So3(Quaternion(-q)).log(), v, rtol=rtol, atol=1e-3 if rtol == 2e-2 else 1e-6)
        self.assert_close(So3(Quaternion(-q)).log(), So3(Quaternion(q)).log())
        axis = torch.tensor([[0.48, 0.6, 0.64]], dtype=torch.float64)  # unit length
        q = Quaternion.from_axis_angle(4.0 * axis).data.to(device=device, dtype=dtype)
        expected = ((4.0 - 2.0 * torch.pi) * axis).to(device=device, dtype=dtype)
        self.assert_close(So3(Quaternion(q)).log(), expected, rtol=rtol, atol=0.0)

    def test_log_keeps_small_rotations(self, device, dtype):
        # log used 2 * acos(real) for the angle. acos loses all of its digits next to real = 1, so in
        # float32 every rotation below 1e-4 rad came back as exactly 0 and 1e-3 rad came back 2% short.
        # quaternion_to_axis_angle measures the same angle with atan2 and keeps full precision.
        theta = {torch.bfloat16: 1e-1, torch.float16: 1e-2, torch.float32: 1e-4, torch.float64: 1e-8}[dtype]
        rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        v = torch.tensor([[0.6, 0.0, 0.8]], device=device, dtype=dtype) * theta
        self.assert_close(So3.exp(v).log(), v, rtol=rtol, atol=0.0)
        # the half turn, where real = 0, keeps its value
        half_turn = So3(Quaternion(torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device, dtype=dtype)))
        self.assert_close(half_turn.log(), torch.tensor([[0.0, torch.pi, 0.0]], device=device, dtype=dtype))

    def test_jacobians_keep_small_rotations(self, device, dtype):
        # (1 - cos theta) / theta**2 and (theta - sin theta) / theta**3 are 0/0 at theta = 0 and evaluate
        # to exactly 0 instead of 1/2 and 1/6 for theta <= 1e-4 in float32 (1e-8 in float64), so the
        # Jacobians were nan at the identity and the identity matrix next to it.
        theta = {torch.bfloat16: 1e-1, torch.float16: 1e-2, torch.float32: 1e-4, torch.float64: 1e-8}[dtype]
        rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        I = torch.eye(3, device=device, dtype=dtype)  # noqa: E741
        zero = torch.zeros(1, 3, device=device, dtype=dtype)
        self.assert_close(So3.right_jacobian(zero)[0], I)
        self.assert_close(So3.left_jacobian(zero)[0], I)
        v = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype) * theta
        # for a rotation about z the [0, 1] entry is (1 - cos theta) / theta = theta / 2 - theta**3 / 24
        expected = torch.tensor(theta / 2 - theta**3 / 24, device=device, dtype=dtype)
        self.assert_close(So3.right_jacobian(v)[0, 0, 1], expected, rtol=rtol, atol=0.0)
        self.assert_close(So3.left_jacobian(v)[0, 0, 1], -expected, rtol=rtol, atol=0.0)

    def test_small_angle_coefficients_across_the_switch(self, device, dtype):
        # The three coefficients on both sides of the series/closed-form switch (0.2 rad in float64, 0.5 in
        # float32), against 50-digit references. Generated with mpmath (mp.dps = 50):
        #   a = (1 - cos t) / t**2, b = (t - sin t) / t**3, c = (1 - (t / 2) * cot(t / 2)) / t**2
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("the closed forms above the switch keep only a few bits in half precision")
        ref = torch.tensor(
            [
                (1e-6, 0.4999999999999583, 0.16666666666665833, 0.08333333333333472),
                (1e-3, 0.4999999583333347, 0.16666665833333352, 0.08333333472222225),
                (0.05, 0.4998958420135014, 0.16664583457336965, 0.08333680576224836),
                (0.19, 0.4984976421808776, 0.16636609177714273, 0.08338351535672024),
                (0.21, 0.498165198998906, 0.16629955230541296, 0.08339464771681691),
                (0.45, 0.4916192476411016, 0.16498727998649976, 0.08361594626029004),
                (0.55, 0.48752224112560083, 0.16416391326425744, 0.08375652128283852),
                (1.0, 0.4596976941318603, 0.1585290151921035, 0.08475613914377404),
                (2.0, 0.3540367091367856, 0.1363378216467898, 0.08947684601641732),
                (3.0, 0.2211102774000495, 0.10588444414593084, 0.09929197039400237),
            ],
            device=device,
            dtype=dtype,
        )
        # measured worst case over [1e-9, pi]: 5.2e-14 (float64, c just above 0.2), 5.9e-6 (float32, c near 0.52)
        rtol = 2e-13 if dtype == torch.float64 else 2e-5
        a, b, c = _so3_small_angle_coefficients(ref[:, 0])
        self.assert_close(torch.stack((a, b, c), -1), ref[:, 1:], rtol=rtol, atol=0.0)
        # exact at the identity
        zero = _so3_small_angle_coefficients(torch.zeros(1, device=device, dtype=dtype))
        self.assert_close(torch.cat(zero), torch.tensor([1 / 2, 1 / 6, 1 / 12], device=device, dtype=dtype))

    def test_convention_log_identity_gradient_is_the_on_manifold_limit_4404(self, device, dtype):
        # The value the guard leaves in place, pinned rather than merely asserted finite: at the
        # identity log evaluates 2 * vec / real, so d(omega_x)/dq_x = 2 -- exact in every dtype,
        # which is why this runs on the dtype fixture rather than pinning float64 (MPS cannot hold
        # float64 at all, and this test's name does not carry the "gradcheck" that conftest skips
        # on that device). Central differences taken through the ambient 4-space disagree (they
        # return 0) because a perturbed (1, h, 0, 0) is not a unit quaternion and lands in the
        # other branch, where acos(1) = 0 kills the result. Along the unit sphere --
        # q(h) = (sqrt(1 - h^2), h, 0, 0), the only path that stays a rotation -- the difference
        # quotient is 2.000000017 at h = 1e-4 in float64, which is this value.
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype, requires_grad=True)
        So3(Quaternion(q)).log()[0, 0].backward()
        self.assert_close(q.grad, torch.tensor([[0.0, 2.0, 0.0, 0.0]], device=device, dtype=dtype))

    def test_exp_matches_float64_at_large_angles_4928(self, device, dtype):
        # #4928: exp took the Taylor branch 0.5 - theta**2 / 48 for sin(theta / 2) / theta below
        # finfo(dtype).eps * 1e3, which is 7.8 rad in bfloat16, so every bfloat16 exp used the two
        # terms: at theta = 3 the quaternion had norm 0.94 and the rotation matrix was off by 0.23.
        axis = torch.tensor([[1.0, 0.0, 0.0], [0.48, 0.6, 0.64]], dtype=torch.float64)
        for theta in (0.97, 3.0):
            v = theta * axis
            q_ref = So3.exp(v).q.data.to(device=device, dtype=dtype)
            q = So3.exp(v.to(device=device, dtype=dtype)).q.data
            self.assert_close(q, q_ref)
            self.assert_close(q.norm(dim=-1), torch.ones(2, device=device, dtype=dtype))

    # TODO: implement me
    def test_jit(self, device, dtype):
        pass

    # TODO: implement me
    def test_module(self, device, dtype):
        pass

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_init(self, device, dtype, batch_size):
        q = Quaternion.identity(batch_size, device, dtype)
        s1 = So3(q)
        s2 = So3(s1.q)
        assert isinstance(s2, So3)
        self.assert_close(s1.q.data, s2.q.data)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_getitem(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size, device, dtype)
        s = So3(q)
        for i in range(batch_size):
            s1 = s[i]
            self.assert_close(s1.q.data, q.data[i])

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_mul(self, device, dtype, batch_size):
        q1 = Quaternion.identity(batch_size, device, dtype)
        q2 = Quaternion.random(batch_size, device, dtype)
        t = self._make_rand_data(device, dtype, batch_size, dims=3)
        s1 = So3(q1)
        s2 = So3(q2)
        self.assert_close((s1 * s2).q.data, s2.q.data)
        self.assert_close((s2 * s2.inverse()).q.data, s1.q.data)
        self.assert_close((s1 * t), t)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_mul_vec(self, device, dtype, batch_size):
        q1 = Quaternion.identity(batch_size, device, dtype)
        q2 = Quaternion.random(batch_size, device, dtype)
        if batch_size is None:
            shape = ()
        else:
            shape = (batch_size,)
        t = Vector3.random(shape, device, dtype)
        s1 = So3(q1)
        s2 = So3(q2)
        self.assert_close((s1 * s2).q.data, s2.q.data)
        self.assert_close((s2 * s2.inverse()).q.data, s1.q.data)
        self.assert_close((s1 * t), t)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_unit_norm(self, device, dtype, batch_size):
        q1 = Quaternion.random(batch_size, device, dtype)
        q2 = Quaternion.random(batch_size, device, dtype)
        s1 = So3(q1)
        s2 = So3(q2)
        s3 = s1 * s2
        s4 = s1.inverse()
        s5 = s2.inverse()
        s6 = s3.inverse()

        ones_vec = torch.tensor(1.0, device=device, dtype=dtype)
        if batch_size is None:
            self.assert_close(s1.q.norm(), ones_vec)
            return

        for i in range(batch_size):
            self.assert_close(s1[i].q.norm(), ones_vec)
            self.assert_close(s2[i].q.norm(), ones_vec)
            self.assert_close(s3[i].q.norm(), ones_vec)
            self.assert_close(s4[i].q.norm(), ones_vec)
            self.assert_close(s5[i].q.norm(), ones_vec)
            self.assert_close(s6[i].q.norm(), ones_vec)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_exp(self, device, dtype, batch_size):
        q = Quaternion.identity(batch_size, device, dtype)
        s = So3(q)
        zero_vec = 0 * self._make_rand_data(device, dtype, batch_size, dims=3)
        self.assert_close(s.exp(zero_vec).q.data, q.data)  # exp of zero vec is identity

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_log(self, device, dtype, batch_size):
        q = Quaternion.identity(batch_size, device, dtype)
        s = So3(q)
        zero_vec = 0 * self._make_rand_data(device, dtype, batch_size, dims=3)
        self.assert_close(s.log(), zero_vec)  # log of identity quat is zero vec

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_exp_log(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size, device, dtype)
        s = So3(q)
        a = self._make_rand_data(device, dtype, batch_size, dims=3)
        b = s.exp(a).log()
        self.assert_close(b, a)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_hat(self, device, dtype, batch_size):
        v = torch.tensor([1, 2, 3], device=device, dtype=dtype)
        expected = v
        if batch_size is not None:
            v = v.repeat(batch_size, 1)
        hat = So3.hat(v)
        if batch_size is None:
            hat = hat[None]
        self.assert_close(hat.unique()[-3:], expected)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_vee(self, device, dtype, batch_size):
        omega = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=device, dtype=dtype)
        expected = torch.tensor([8, 3, 4], device=device, dtype=dtype)
        if batch_size is not None:
            omega = omega.repeat(batch_size, 1, 1)
            expected = expected.repeat(batch_size, 1)
        self.assert_close(So3.vee(omega), expected)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_hat_vee(self, device, dtype, batch_size):
        a = self._make_rand_data(device, dtype, batch_size, dims=3)
        omega = So3.hat(a)
        b = So3.vee(omega)
        self.assert_close(b, a)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_matrix(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size, device, dtype)
        r = So3(q).matrix()
        if batch_size is None:
            q = Quaternion(q.data[None])
            r = r[None]
        for i in range(r.shape[0]):
            q1 = q[i]
            r1 = r[i, :, :]
            pvec = torch.rand(3, device=device, dtype=dtype)
            pquat = Quaternion(torch.cat([torch.tensor([0], device=device, dtype=dtype), pvec]))
            qp_ = q1 * pquat * q1.inv()
            rp_ = torch.matmul(r1, pvec)
            self.assert_close(rp_, qp_.vec)  # p_ = R*p = q*p*q_inv
            self.assert_close(rp_.norm(), pvec.norm())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_from_wxyz(self, device, dtype, batch_size):
        wxyz = self._make_rand_data(device, dtype, batch_size, dims=4)
        s = So3.from_wxyz(wxyz)
        self.assert_close(s.q.data, wxyz)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_ortho(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size, device, dtype)
        b_R_a = So3(q).matrix()
        a_R_b = So3(q).inverse().matrix()
        a_R_a = (So3(q) * So3(q).inverse()).matrix()

        eye_mat = torch.eye(3, device=device, dtype=dtype)
        if batch_size is None:
            eye_mat = eye_mat[None]
            a_R_a = a_R_a[None]
            a_R_b = a_R_b[None]
            b_R_a = b_R_a[None]
        if batch_size is not None:
            eye_mat = eye_mat.repeat(batch_size, 1, 1)

        self.assert_close(a_R_a, eye_mat)

        for i in range(eye_mat.shape[0]):
            self.assert_close(a_R_a[i, :, :], eye_mat[i])
            self.assert_close(a_R_b[i, :, :] @ b_R_a[i, :, :], eye_mat[i])
            self.assert_close(b_R_a[i, :, :] @ a_R_b[i, :, :], eye_mat[i])

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_inverse(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size, device, dtype)
        self.assert_close(So3(q).inverse().inverse().q.data, q.data)
        self.assert_close(So3(q).inverse().inverse().matrix(), So3(q).matrix())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_rot_x(self, device, dtype, batch_size):
        x = self._make_rand_data(device, dtype, batch_size, dims=1).squeeze(-1)
        so3 = So3.rot_x(x)
        roll, _, _ = euler_from_quaternion(*so3.q.coeffs)
        self.assert_close(x, roll)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_rot_y(self, device, dtype, batch_size):
        y = self._make_rand_data(device, dtype, batch_size, dims=1).squeeze(-1)
        so3 = So3.rot_y(y)
        _, pitch, _ = euler_from_quaternion(*so3.q.coeffs)
        self.assert_close(y, pitch)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_rot_z(self, device, dtype, batch_size):
        z = self._make_rand_data(device, dtype, batch_size, dims=1).squeeze(-1)
        so3 = So3.rot_z(z)
        _, _, yaw = euler_from_quaternion(*so3.q.coeffs)
        self.assert_close(z, yaw)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_adjoint(self, device, dtype, batch_size):
        shape = (3,) if batch_size is None else (batch_size, 3)
        x = So3.exp(torch.tensor([0.1, -0.2, 0.3], device=device, dtype=dtype).expand(shape))
        y = So3.exp(torch.tensor([-0.3, 0.1, 0.2], device=device, dtype=dtype).expand(shape))
        adjoint = x.adjoint()
        inverse_adjoint = x.inverse().adjoint()
        identity = torch.eye(adjoint.shape[-1], device=device, dtype=dtype).expand_as(adjoint)
        half_tolerance = torch.finfo(dtype).eps if dtype in (torch.float16, torch.bfloat16) else None
        self.assert_close(adjoint @ inverse_adjoint, identity, rtol=half_tolerance, atol=half_tolerance)
        self.assert_close((x * y).adjoint(), adjoint @ y.adjoint(), rtol=half_tolerance, atol=half_tolerance)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_random(self, device, dtype, batch_size):
        s = So3.random(batch_size=batch_size, device=device, dtype=dtype)
        s_in_s = s.inverse() * s
        i = So3.identity(batch_size=batch_size, device=device, dtype=dtype)
        self.assert_close(s_in_s.q.data, i.q.data)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_right_jacobian(self, device, dtype, batch_size):
        vec = self._make_rand_data(device, dtype, batch_size, dims=3)
        Jr = So3.right_jacobian(vec)
        I = torch.eye(3, device=device, dtype=dtype).expand_as(Jr)  # noqa: E741
        self.assert_close(vec[..., None], Jr @ vec[..., None])
        self.assert_close(Jr.transpose(-1, -2) @ Jr, I, atol=0.1, rtol=0.1)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_left_jacobian(self, device, dtype, batch_size):
        vec = self._make_rand_data(device, dtype, batch_size, dims=3)
        Jl = So3.left_jacobian(vec)
        I = torch.eye(3, device=device, dtype=dtype).expand_as(Jl)  # noqa: E741
        self.assert_close(vec[..., None], Jl @ vec[..., None])
        self.assert_close(Jl.transpose(-1, -2) @ Jl, I, atol=0.1, rtol=0.1)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_right_left_jacobian(self, device, dtype, batch_size):
        vec = self._make_rand_data(device, dtype, batch_size, dims=3)
        Jr = So3.right_jacobian(vec)
        Jl = So3.left_jacobian(vec)
        self.assert_close(Jl, Jr.transpose(-1, -2))
