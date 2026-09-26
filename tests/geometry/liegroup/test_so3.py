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
from torch import nn

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

    def test_small_angle_coefficients_gradient_past_the_series_range_4965(self, device, dtype):
        # #4965: the series were evaluated on every angle and discarded above the switch by torch.where, which
        # still differentiates them. In float16 their Horner terms overflow from about 50 rad, so 0 * inf = nan
        # reached the gradient of the coefficients and of both Jacobians although the values came from the closed
        # forms. The closed forms then divided by theta**2, which overflows float16 above 256 rad and made the
        # gradient nan again; they now divide by theta twice, so 300 and 1000 rad are finite as well.
        theta = torch.tensor([50.0, 100.0, 200.0, 300.0, 1000.0], device=device, dtype=dtype, requires_grad=True)
        sum(c.sum() for c in _so3_small_angle_coefficients(theta)).backward()
        assert bool(torch.isfinite(theta.grad).all()), theta.grad
        # [omega]_x^2 itself overflows float16 above 256 rad, so the Jacobians are checked up to 200 rad.
        axis = torch.tensor([0.48, 0.6, 0.64], device=device, dtype=dtype)
        for jacobian in (So3.right_jacobian, So3.left_jacobian):
            v = (theta.detach()[:3, None] * axis).requires_grad_(True)
            jacobian(v).sum().backward()
            assert bool(torch.isfinite(v.grad).all()), (jacobian.__name__, v.grad)

    def test_small_angle_coefficients_and_jacobians_above_40_rad_4965(self, device, dtype):
        # #4965: the closed forms divided by theta**3 and theta**2, which overflow float16 above 40.3 and 256
        # rad. From 41 rad b was 0, so the [omega]_x^2 term of both Jacobians dropped out: at 41 rad about
        # [0.48, 0.6, 0.64] the float16 right_jacobian was off by 0.79. 50-digit references generated with
        # mpmath (mp.dps = 50): a = (1 - cos t) / t**2, b = (t - sin t) / t**3, c = (1 - (t / 2) * cot(t / 2)) / t**2.
        finfo = torch.finfo(dtype)
        ref = torch.tensor(
            [
                (41.0, 0.0011822363340415387, 0.0005971855119456292, 0.001568257196741034),
                (60.0, 0.00054233693900421, 0.00027918893806065843, 0.001578777379124938),
                (288.0, 5.813614141459137e-06, 1.2092140495844536e-05, 0.0030921829471326203),
            ],
            device=device,
            dtype=dtype,
        )
        # a and b at 288 rad are subnormal in float16, hence the atol of 8 subnormal steps
        a, b, c = _so3_small_angle_coefficients(ref[:, 0])
        self.assert_close(torch.stack((a, b, c), -1), ref[:, 1:], rtol=8 * finfo.eps, atol=8 * finfo.eps * finfo.tiny)
        # the Jacobians against the float64 path on the CPU, below the float16 overflow of [omega]_x^2 itself
        v = ref[:2, :1] * torch.tensor([0.48, 0.6, 0.64], device=device, dtype=dtype)
        for jacobian in (So3.right_jacobian, So3.left_jacobian):
            expected = jacobian(v.cpu().double()).to(device=device, dtype=dtype)
            self.assert_close(jacobian(v), expected, rtol=8 * finfo.eps, atol=8 * finfo.eps)

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

    def test_exp_across_the_series_switch_4928(self, device, dtype):
        # exp about z is (cos(t / 2), 0, 0, sin(t / 2)) on both sides of the 0.5 rad switch, against
        # 50-digit references. Generated with mpmath (mp.dps = 50): cos(t / 2), sin(t / 2).
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("half precision keeps too few bits to see the series coefficients")
        ref = torch.tensor(
            [
                (1e-3, 0.9999998750000026, 0.0004999999791666669),
                (0.3, 0.9887710779360422, 0.14943813247359922),
                (0.49, 0.9701373249726354, 0.24255632478857206),
                (0.51, 0.9676632956885756, 0.25224540863437805),
                (1.9, 0.5816830894638836, 0.8134155047893737),
                (3.0, 0.0707372016677029, 0.9974949866040544),
                (6.0, -0.9899924966004454, 0.1411200080598672),
            ],
            device=device,
            dtype=dtype,
        )
        zero = torch.zeros_like(ref[:, :1])
        q = So3.exp(torch.cat((zero, zero, ref[:, :1]), -1)).q.data
        rtol = 1e-15 if dtype == torch.float64 else 1e-6
        self.assert_close(q, torch.cat((ref[:, 1:2], zero, zero, ref[:, 2:]), -1), rtol=rtol, atol=0.0)

    def test_exp_gradient_is_finite_at_large_angles(self, device, dtype):
        # The series branch is differentiated even where the closed form is selected; evaluated on the
        # raw angle its powers overflow float16 above about 90 rad and 0 * inf = nan reached v.grad.
        axis = torch.tensor([0.48, 0.6, 0.64], device=device, dtype=dtype)
        v = (torch.tensor([[0.0], [0.3], [3.0], [120.0]], device=device, dtype=dtype) * axis).requires_grad_(True)
        So3.exp(v).q.data.sum().backward()
        assert bool(torch.isfinite(v.grad).all()), v.grad

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


class _RotationHolder(nn.Module):
    """A module that keeps a Quaternion and an So3 as attributes, as a pose-holding model would."""

    def __init__(self, data: torch.Tensor, as_parameter: bool) -> None:
        super().__init__()
        self.quat = Quaternion(nn.Parameter(data.clone()) if as_parameter else data.clone())
        self.rot = So3(Quaternion(nn.Parameter(data.clone()) if as_parameter else data.clone()))


class TestSo3Conventions(BaseTester):
    def test_convention_so3_exp_is_rotation_vector(self, device, dtype):
        # exp(v) is the rotation by |v| about v / |v|, the matrix exponential of hat(v). Expected matrix from scipy:
        #   Rotation.from_rotvec([0.3, -0.5, 0.2]).as_matrix()
        v = torch.tensor([[0.3, -0.5, 0.2]], device=device, dtype=dtype)
        expected = torch.tensor(
            [
                [
                    [0.8595338985586632, -0.2602267140480945, -0.43986763295823095],
                    [0.11491695393636675, 0.937032437284918, -0.3297943376922552],
                    [0.4979915370029221, 0.23292116428443665, 0.8353156052067087],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(So3.exp(v).matrix(), expected)
        # the opposite vector is the inverse rotation
        self.assert_close(So3.exp(-v).matrix(), So3.exp(v).inverse().matrix())

    def test_convention_so3_hat_is_cross_product_matrix(self, device, dtype):
        v = torch.tensor([[0.3, -0.5, 0.2]], device=device, dtype=dtype)
        p = torch.tensor([[1.0, 2.0, 3.0]], device=device, dtype=dtype)
        # hat(v) @ p = v x p; the transposed matrix would give p x v
        self.assert_close((So3.hat(v) @ p[..., None])[..., 0], torch.linalg.cross(v, p, dim=-1))
        self.assert_close(So3.vee(So3.hat(v)), v)

    def test_convention_so3_mul_point_is_R_p(self, device, dtype):
        s = So3.exp(torch.tensor([[0.3, -0.5, 0.2]], device=device, dtype=dtype))
        s2 = So3.exp(torch.tensor([[-0.1, 0.4, 0.25]], device=device, dtype=dtype))
        p = torch.tensor([[1.0, 2.0, 3.0]], device=device, dtype=dtype)
        # s * p rotates the point: R @ p, not R^T @ p. Expected point from scipy:
        #   Rotation.from_rotvec([0.3, -0.5, 0.2]).apply([1.0, 2.0, 3.0])
        expected = torch.tensor(
            [[-0.9805224284122186, 0.9995988154294373, 3.469780681191921]], device=device, dtype=dtype
        )
        self.assert_close(s * p, expected)
        self.assert_close(s * p, (s.matrix() @ p[..., None])[..., 0])
        # precondition: the pair does not commute
        assert ((s * s2).matrix() - (s2 * s).matrix()).abs().max() > 0.1
        # s * s2 applies s2 first: its matrix is R @ R2, and (s * s2) * p = s * (s2 * p)
        self.assert_close((s * s2).matrix(), s.matrix() @ s2.matrix())
        # both sides round two rotations of a point with components up to 3.5, where one float16 ulp is 2e-3 (the
        # default float16 atol is 1e-3; the bfloat16 default already scales with the value)
        tol = 1e-2 if dtype == torch.float16 else None
        self.assert_close((s * s2) * p, s * (s2 * p), rtol=tol, atol=tol)

    def test_convention_so3_adjoint_conjugates_the_tangent(self, device, dtype):
        # Ad(g) xi is the tangent of g exp(xi) g^-1: exp(Ad(g) xi) = g exp(xi) g^-1, with Ad(g) = R
        g = So3.exp(torch.tensor([[0.7, -0.2, 0.4]], device=device, dtype=dtype))
        xi = torch.tensor([[0.1, 0.3, -0.2]], device=device, dtype=dtype)
        moved = (g.adjoint() @ xi[..., None])[..., 0]
        self.assert_close(So3.exp(moved).matrix(), (g * So3.exp(xi) * g.inverse()).matrix())

    def test_convention_so3_jacobian_sides(self, device, dtype):
        # exp(w + d) = exp(w) exp(Jr(w) d) = exp(Jl(w) d) exp(w) to first order in d. The references are central
        # differences in float64 on the CPU with h = 1e-6, whose error is below 1e-9.
        w = torch.tensor([[0.7, -0.2, 0.4]], dtype=torch.float64)
        h = 1e-6
        base = So3.exp(w)
        right, left = [], []
        for i in range(3):
            step = torch.zeros(1, 3, dtype=torch.float64)
            step[0, i] = h
            plus, minus = So3.exp(w + step), So3.exp(w - step)
            right.append(((base.inverse() * plus).log() - (base.inverse() * minus).log()) / (2 * h))
            left.append(((plus * base.inverse()).log() - (minus * base.inverse()).log()) / (2 * h))
        right = torch.stack(right, -1).to(device=device, dtype=dtype)
        left = torch.stack(left, -1).to(device=device, dtype=dtype)
        # precondition: the two sides differ by 2 a(theta) hat(w), 0.66 here
        assert (right - left).abs().max() > 0.1
        w = w.to(device=device, dtype=dtype)
        # measured float64 error of the differences: 1e-10
        tol = 1e-8 if dtype == torch.float64 else None
        self.assert_close(So3.right_jacobian(w), right, rtol=tol, atol=tol)
        self.assert_close(So3.left_jacobian(w), left, rtol=tol, atol=tol)
        self.assert_close(So3.left_jacobian(w), So3.right_jacobian(-w))

    def test_convention_so3_rot_axes_are_right_handed(self, device, dtype):
        # rot_x, rot_y, rot_z by +theta turn y toward z, z toward x and x toward y
        theta = torch.tensor([0.3], device=device, dtype=dtype)
        c, s = math.cos(0.3), math.sin(0.3)
        cases = [
            (So3.rot_x, [0.0, 1.0, 0.0], [0.0, c, s]),
            (So3.rot_y, [0.0, 0.0, 1.0], [s, 0.0, c]),
            (So3.rot_z, [1.0, 0.0, 0.0], [c, s, 0.0]),
        ]
        for rot, p, expected in cases:
            p = torch.tensor([p], device=device, dtype=dtype)
            self.assert_close(rot(theta) * p, torch.tensor([expected], device=device, dtype=dtype))

    def test_wart_so3_non_unit_quaternion_not_normalised_4942(self, device, dtype):
        # #4942 https://github.com/kornia/kornia/issues/4942: So3 stores the quaternion as given, and matrix() and
        # So3 * p use the unit-quaternion formulas, so a non-unit q gives a scaled non-rotation matrix and points
        # scaled by |q|^2. This test turns red when So3 normalises its quaternion.
        data = torch.tensor([[2.0, 0.2, -0.6, 0.4]], device=device, dtype=dtype)
        squared_norm = 4.56
        s = So3(Quaternion(data))
        p = torch.tensor([[1.0, 2.0, 3.0]], device=device, dtype=dtype)
        m = s.matrix()
        # det = m0 . (m1 x m2), written out so it runs where torch.linalg.det has no half-precision kernel
        det = (m[:, 0] * torch.linalg.cross(m[:, 1], m[:, 2], dim=-1)).sum(-1)
        assert bool((det > 2.0).all()), det
        self.assert_close((s * p).norm(dim=-1), squared_norm * p.norm(dim=-1))
        # control: the same data through Quaternion.matrix(), and through So3 once normalised, is a rotation
        self.assert_close(So3(Quaternion(data).normalize()).matrix(), Quaternion(data).matrix())
        self.assert_close((So3(Quaternion(data).normalize()) * p).norm(dim=-1), p.norm(dim=-1))

    def test_wart_rotation_state_not_registered_4923(self, device, dtype):
        # #4923 https://github.com/kornia/kornia/issues/4923: a Quaternion built from a plain tensor keeps it as an
        # unregistered attribute, so a module holding it (directly or through So3) saves no key for the rotation,
        # and load_state_dict reports success while keeping the old rotation. This test turns red when the
        # rotation is registered.
        saved = torch.tensor([[0.8, 0.2, -0.4, 0.4]], device=device, dtype=dtype)
        stale = torch.tensor([[0.0, 0.6, 0.0, 0.8]], device=device, dtype=dtype)
        source, target = _RotationHolder(saved, as_parameter=False), _RotationHolder(stale, as_parameter=False)
        assert list(source.state_dict()) == []
        result = target.load_state_dict(source.state_dict())
        assert not result.missing_keys and not result.unexpected_keys
        self.assert_close(target.quat.data, stale)
        self.assert_close(target.rot.q.data, stale)
        # control: a rotation stored as an nn.Parameter is saved and restored
        source, target = _RotationHolder(saved, as_parameter=True), _RotationHolder(stale, as_parameter=True)
        assert list(source.state_dict()) == ["quat._data", "rot._q._data"]
        target.load_state_dict(source.state_dict())
        self.assert_close(target.quat.data, saved)
        self.assert_close(target.rot.q.data, saved)
