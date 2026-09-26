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

from kornia.geometry.liegroup import Se2, So2
from kornia.geometry.vector import Vector2

from testing.base import BaseTester


class TestSe2(BaseTester):
    def _make_rand_data(self, device, dtype, input_shape):
        batch_size = input_shape[0]
        shape = input_shape[1:] if batch_size is None else input_shape
        return torch.rand(shape, device=device, dtype=dtype)

    def test_smoke(self, device, dtype):
        z = torch.rand((2,), dtype=torch.cfloat, device=device)
        so2 = So2(z)
        t = torch.rand((1, 2), device=device, dtype=dtype)
        s = Se2(so2, t)
        assert isinstance(s, Se2)
        assert isinstance(s.r, So2)
        self.assert_close(s.r.z.data, z)
        self.assert_close(s.t, t)

    @pytest.mark.parametrize("input_shape", [(1,), (2,), (5,), ()])
    def test_cardinality(self, device, dtype, input_shape):
        t_input_shape = (*input_shape, 2)
        z = torch.randn((*input_shape, 2), dtype=dtype, device=device)
        t = torch.randn(t_input_shape, dtype=dtype, device=device)
        s = Se2(So2(torch.complex(z[..., 0], z[..., 1])), t)
        theta = torch.rand((*input_shape, 3), dtype=dtype, device=device)
        assert s.so2.z.shape == input_shape
        assert s.t.shape == t_input_shape
        assert (s * s).so2.z.shape == input_shape
        assert (s * s).t.shape == t_input_shape
        assert s.exp(theta).so2.z.shape == input_shape
        assert s.exp(theta).t.shape == t_input_shape
        assert s.log().shape == (*input_shape, 3)
        if not any(input_shape):
            expected_hat_shape = (3, 3)
        else:
            expected_hat_shape = (input_shape[0], 3, 3)
        assert s.hat(theta).shape == expected_hat_shape
        assert s.inverse().so2.z.shape == input_shape
        assert s.inverse().t.shape == t_input_shape

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_exception(self, device, dtype, batch_size):
        with pytest.raises(ValueError):
            r = So2.random(batch_size)
            t1 = torch.randn((batch_size, 1), dtype=dtype, device=device)
            t2 = torch.randn((batch_size, 3), dtype=dtype, device=device)
            Se2(r, t1)
            Se2(r, t2)
        with pytest.raises(ValueError):
            theta = torch.rand((batch_size, 2), dtype=dtype, device=device)
            Se2.exp(theta)
        with pytest.raises(ValueError):
            v = torch.rand((batch_size, 2), dtype=dtype, device=device)
            Se2.hat(v)
        with pytest.raises(ValueError):
            omega = torch.rand((4, 4), dtype=dtype, device=device)
            Se2.vee(omega)
        with pytest.raises(TypeError):
            Se2.identity(1, device, dtype) * [1.0, 2.0, 1.0]
        with pytest.raises(ValueError):
            theta = torch.rand((batch_size, 2), dtype=dtype, device=device)
            Se2.hat(theta)
        with pytest.raises(Exception):
            Se2.identity(batch_size=0)
        with pytest.raises(Exception):
            Se2.random(batch_size=0)
        with pytest.raises(Exception):
            x = torch.rand(5, dtype=dtype, device=device)
            y = torch.rand(3, dtype=dtype, device=device)
            Se2.trans(x, y)

    def test_gradcheck(self, device):
        v = torch.tensor([[1.0, 2.0, 0.4]], device=device, dtype=torch.float64)
        self.gradcheck(lambda x: Se2.exp(x).matrix(), (v,))

    def test_gradient_at_the_identity_4404_4924(self, device, dtype):
        # #4404: both quotients that build the translation block are 0/0 at theta = 0, and the
        # torch.where that discards them still differentiates them, so 0 * nan = nan reached the
        # gradient at the identity. #4924: the value that where fell back to there was 0 where the
        # limit is 1, so the (vx, vy) gradient of exp(v).t was 0 and the theta gradient of log was
        # nan. Pin the values: with t = V(theta) (vx, vy), d sum(matrix()) / dv is
        # [1, 1, (vx - vy) / 2] at theta = 0 and d sum(log(exp(v))) / dv is [1, 1, 1].
        if dtype == torch.bfloat16:
            # Se2 holds its rotation as a complex So2, and torch.complex has no bfloat16 overload,
            # so most of this class already cannot run at that dtype -- unrelated to the guard.
            pytest.skip("torch.complex has no bfloat16 overload, so So2 cannot be built at all")
        v = torch.tensor([[1.0, 2.0, 0.0]], device=device, dtype=dtype, requires_grad=True)
        Se2.exp(v).matrix().sum().backward()
        self.assert_close(v.grad, torch.tensor([[1.0, 1.0, -0.5]], device=device, dtype=dtype))
        v.grad = None
        Se2.exp(v).log().sum().backward()
        self.assert_close(v.grad, torch.tensor([[1.0, 1.0, 1.0]], device=device, dtype=dtype))

    # TODO: implement me
    def test_jit(self, device, dtype):
        pass

    # TODO: implement me
    def test_module(self, device, dtype):
        pass

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_init(self, device, dtype, batch_size):
        s1 = Se2.random(batch_size, device, dtype)
        s2 = Se2(s1.r, s1.t)
        assert isinstance(s2, Se2)
        self.assert_close(s1.r.z, s2.r.z)
        self.assert_close(s1.t, s2.t)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_getitem(self, device, dtype, batch_size):
        z = torch.rand(batch_size, dtype=torch.cfloat, device=device)
        t = torch.rand((batch_size, 2), device=device, dtype=dtype)
        s = Se2(So2(z), t)
        for i in range(batch_size):
            s1 = s[i]
            self.assert_close(s1.r.z, z[i])
            self.assert_close(s1.t, t[i])

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_mul(self, device, dtype, batch_size):
        s1 = Se2.identity(batch_size, device, dtype)
        s2 = Se2.random(batch_size, device, dtype)
        s1_pose_s2 = s1 * s2
        s2_pose_s2 = s2 * s2.inverse()
        zeros_vec = torch.zeros(2, device=device, dtype=dtype)
        if batch_size is not None:
            zeros_vec = zeros_vec.repeat(batch_size, 1)
        so2_expected = So2.identity(batch_size, device, dtype)
        self.assert_close(s1_pose_s2.r.z, s2.r.z)
        self.assert_close(s1_pose_s2.t, s2.t)
        self.assert_close(s2_pose_s2.r.z.real, so2_expected.z.real)
        self.assert_close(s2_pose_s2.r.z.imag, so2_expected.z.imag)
        self.assert_close(s2_pose_s2.t, zeros_vec)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_mul_vector(self, device, dtype, batch_size):
        s1 = Se2.identity(batch_size, device, dtype)
        if batch_size is None:
            shape = ()
        else:
            shape = (batch_size,)
        s2 = Se2(So2.identity(batch_size, device, dtype), Vector2.random(shape, device, dtype))
        s1_pose_s2 = s1 * s2
        s2_pose_s2 = s2 * s2.inverse()
        zeros_vec = torch.zeros(2, device=device, dtype=dtype)
        if batch_size is not None:
            zeros_vec = zeros_vec.repeat(batch_size, 1)
        so2_expected = So2.identity(batch_size, device, dtype)
        self.assert_close(s1_pose_s2.r.z, s2.r.z)
        self.assert_close(s1_pose_s2.t, s2.t)
        self.assert_close(s2_pose_s2.r.z.real, so2_expected.z.real)
        self.assert_close(s2_pose_s2.r.z.imag, so2_expected.z.imag)
        self.assert_close(s2_pose_s2.t, zeros_vec)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_exp(self, device, dtype, batch_size):
        t = self._make_rand_data(device, dtype, (batch_size, 2))
        theta = torch.zeros(batch_size if batch_size is not None else (), device=device, dtype=dtype)
        s = Se2.exp(torch.cat((t, theta[..., None]), -1))
        self.assert_close(s.r.z, So2.exp(theta).z)
        # V(0) is the identity, so a pure translation keeps its translation (#4924)
        self.assert_close(s.t, t)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_log(self, device, dtype, batch_size):
        t = self._make_rand_data(device, dtype, (batch_size, 2))
        s = Se2(So2.identity(batch_size, device, dtype), t)
        # V(0)^-1 is the identity, so the log of a pure translation is (t, 0) (#4924)
        self.assert_close(s.log(), torch.cat((t, torch.zeros_like(t[..., :1])), -1))

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_exp_log(self, device, dtype, batch_size):
        a = self._make_rand_data(device, dtype, (batch_size, 3))
        b = Se2.exp(a).log()
        self.assert_close(b, a, low_tolerance=True)

    def test_exp_log_keep_small_rotations_4924(self, device, dtype):
        # #4924: exp guarded a = sin(theta) / theta and b = (1 - cos(theta)) / theta with a fallback of 0
        # at theta = 0, where their limits are 1 and 0, and log guarded (theta / 2) cot(theta / 2) the same
        # way, so a pure translation went into and out of the tangent space as [0, 0]. Just above
        # theta = 0 the closed forms cancelled: exp(v).t was off by 1e-4 and log(exp(v)) by 2.0 at
        # theta = 1e-4 in float32 (2e-2 in float16, 1e-8 in float64).
        if dtype == torch.bfloat16:
            pytest.skip("torch.complex has no bfloat16 overload, so So2 cannot be built at all")
        theta = {torch.float16: 2e-2, torch.float32: 1e-4}.get(dtype, 1e-8)
        v = torch.tensor([[1.0, 2.0, 0.0], [1.0, 2.0, theta], [1.0, 2.0, -theta]], dtype=dtype)
        # V(theta) (vx, vy) for the rounded angle, in float64 on the CPU (MPS has no float64) and
        # without cancellation: 1 - cos(theta) = 2 sin(theta / 2)^2
        th = v[..., 2].double()
        a = torch.where(th == 0, torch.ones_like(th), torch.sin(th) / th)
        b = torch.where(th == 0, torch.zeros_like(th), 2 * torch.sin(th / 2) ** 2 / th)
        x, y = v[..., 0].double(), v[..., 1].double()
        t_ref = torch.stack((a * x - b * y, b * x + a * y), -1).to(device=device, dtype=dtype)
        v = v.to(device)
        eps = torch.finfo(dtype).eps
        self.assert_close(Se2.exp(v).t, t_ref, rtol=8 * eps, atol=8 * eps)
        g = Se2(So2.exp(v[..., 2]), t_ref)  # the element exp(v), rounded to dtype
        self.assert_close(g.log(), v, rtol=8 * eps, atol=8 * eps)

    def test_exp_keeps_large_angles_4924(self, device, dtype):
        # Past the small-angle branch exp takes sin(theta) / theta and 2 sin(theta / 2)^2 / theta directly.
        # 1 - theta^2 (theta - sin(theta)) / theta^3 cancels wherever sin(theta) / theta is small (37.7 is
        # within 1e-3 of 12 pi), 1 - cos(theta) cancels there too, and theta^3 overflows float16 above 40.3 rad,
        # which made that coefficient 0 and exp(v).t close to (vx, vy); the series that torch.where discards
        # overflows from 50 rad and made the gradient nan.
        if dtype == torch.bfloat16:
            pytest.skip("torch.complex has no bfloat16 overload, so So2 cannot be built at all")
        angles = (-300.0, -100.0, -41.0, 12.5, 37.7, 41.0, 100.0, 300.0)
        v = torch.tensor([[1.0, 2.0, th] for th in angles], dtype=dtype)
        # V(theta) (vx, vy) for the rounded angle, in float64 on the CPU (MPS has no float64)
        th = v[..., 2].double()
        a, b = torch.sin(th) / th, 2 * torch.sin(th / 2) ** 2 / th
        x, y = v[..., 0].double(), v[..., 1].double()
        t_ref = torch.stack((a * x - b * y, b * x + a * y), -1).to(device=device, dtype=dtype)
        v = v.to(device).requires_grad_(True)
        t = Se2.exp(v).t
        self.assert_close(t, t_ref, rtol=16 * torch.finfo(dtype).eps, atol=0.0)
        t.sum().backward()
        assert bool(torch.isfinite(v.grad).all()), v.grad

    def test_exp_gradient_below_the_switch_4924(self, device, dtype):
        # Below 0.5 rad the angle gradient comes from the series: autograd of sin(theta) / theta is
        # cos(theta) / theta - sin(theta) / theta^2, which loses about three digits at theta = 0.06.
        # Reference: the derivative series of a = sin(theta) / theta and b = (1 - cos(theta)) / theta in float64.
        if dtype == torch.bfloat16:
            pytest.skip("torch.complex has no bfloat16 overload, so So2 cannot be built at all")
        v = torch.tensor([[1.0, 2.0, 0.06], [1.0, 2.0, -0.3]], dtype=dtype)
        th = v[..., 2].double()
        da = sum((-1) ** k * 2 * k * th ** (2 * k - 1) / float(math.factorial(2 * k + 1)) for k in range(1, 12))
        db = sum((-1) ** (k + 1) * (2 * k - 1) * th ** (2 * k - 2) / float(math.factorial(2 * k)) for k in range(1, 12))
        # d sum(t) / d theta with t = (a x - b y, b x + a y)
        x, y = v[..., 0].double(), v[..., 1].double()
        ref = (da * (x + y) + db * (x - y)).to(device=device, dtype=dtype)
        v = v.to(device).requires_grad_(True)
        Se2.exp(v).t.sum().backward()
        eps = torch.finfo(dtype).eps
        self.assert_close(v.grad[..., 2], ref, rtol=8 * eps, atol=8 * eps)

    def test_exp_log_negative_angles_past_the_series_switch_4924(self, device, dtype):
        # The coefficients are even in theta, so exp and log evaluate them at |theta|. The So3 helper takes its
        # series below a positive switch point, so a signed theta would take the truncated series for every
        # negative angle; at theta = -3 that is off by about 1e-4 in log. Pin both signs past the switch.
        if dtype == torch.bfloat16:
            pytest.skip("torch.complex has no bfloat16 overload, so So2 cannot be built at all")
        v = torch.tensor([[1.0, 2.0, th] for th in (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)], dtype=dtype)
        # V(theta) (vx, vy) for the rounded angle, in float64 on the CPU (MPS has no float64)
        th = v[..., 2].double()
        a, b = torch.sin(th) / th, 2 * torch.sin(th / 2) ** 2 / th
        x, y = v[..., 0].double(), v[..., 1].double()
        t_ref = torch.stack((a * x - b * y, b * x + a * y), -1).to(device=device, dtype=dtype)
        v = v.to(device)
        eps = torch.finfo(dtype).eps
        self.assert_close(Se2.exp(v).t, t_ref, rtol=8 * eps, atol=8 * eps)
        g = Se2(So2.exp(v[..., 2]), t_ref)  # the element exp(v), rounded to dtype
        self.assert_close(g.log(), v, rtol=8 * eps, atol=8 * eps)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_hat(self, device, dtype, batch_size):
        v = self._make_rand_data(device, dtype, (batch_size, 2))
        theta = self._make_rand_data(device, dtype, (batch_size, 1))
        s_hat = Se2.hat(torch.cat((v, theta), -1))
        self.assert_close(v, s_hat[..., 2, 0:2])
        self.assert_close(s_hat[..., 0:2, 0:2].squeeze(), So2.hat(theta).squeeze())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_vee(self, device, dtype, batch_size):
        omega = self._make_rand_data(device, dtype, input_shape=(batch_size, 3, 3))
        v = Se2.vee(omega)
        self.assert_close(torch.stack((v[..., 0], v[..., 1]), -1), omega[..., 2, :2])
        self.assert_close(v[..., -1], omega[..., 0, 1])

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_hat_vee(self, device, dtype, batch_size):
        a = self._make_rand_data(device, dtype, (batch_size, 3))
        omega_hat = Se2.hat(a)
        b = Se2.vee(omega_hat)
        self.assert_close(b, a)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_identity(self, device, dtype, batch_size):
        s = Se2.random(batch_size)
        s_pose_s = s * Se2.identity(batch_size)
        self.assert_close(s_pose_s.so2.z.real, s.so2.z.real)
        self.assert_close(s_pose_s.so2.z.imag, s.so2.z.imag)
        self.assert_close(s.t, s_pose_s.t)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_matrix(self, device, dtype, batch_size):
        theta = self._make_rand_data(device, dtype, (batch_size,))
        t = self._make_rand_data(device, dtype, (batch_size, 2))
        s = So2.exp(theta)
        p1 = s * t
        p2 = s.matrix() @ t[..., None]
        self.assert_close(p1, p2.squeeze(-1))

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_from_matrix(self, device, dtype, batch_size):
        theta = self._make_rand_data(device, dtype, (batch_size,))
        t = self._make_rand_data(device, dtype, (batch_size, 2))
        s = So2.exp(theta)
        p1 = s * t
        RT = torch.eye(3, device=device, dtype=dtype)
        if batch_size is not None:
            RT = RT.repeat(batch_size, 1, 1)
        RT[..., :2, :2] = s.matrix()
        p2 = Se2.from_matrix(RT) * t
        self.assert_close(p1, p2)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_inverse(self, device, batch_size, dtype):
        s = Se2.random(batch_size, device, dtype)
        s_in_in = s.inverse().inverse()
        self.assert_close(s_in_in.so2.z.real, s.so2.z.real)
        self.assert_close(s_in_in.so2.z.imag, s.so2.z.imag)
        self.assert_close(s_in_in.t, s.t)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_random(self, device, dtype, batch_size):
        s = So2.random(batch_size=batch_size, device=device, dtype=dtype)
        t = self._make_rand_data(device, dtype, (batch_size, 2))
        se2 = Se2(s, t)
        se2_in_se2 = se2.inverse() * se2
        i = Se2.identity(batch_size=batch_size, device=device, dtype=dtype)
        self.assert_close(se2_in_se2.so2.z.real, i.so2.z.real)
        self.assert_close(se2_in_se2.so2.z.imag, i.so2.z.imag)
        self.assert_close(se2_in_se2.t, i.t)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_trans(self, device, dtype, batch_size):
        trans = self._make_rand_data(device, dtype, (batch_size, 2))
        x, y = trans[..., 0], trans[..., 1]
        se2 = Se2.trans(x, y)
        self.assert_close(se2.t, trans)
        self.assert_close(se2.so2.matrix(), So2.identity(batch_size, device, dtype).matrix())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_trans_x(self, device, dtype, batch_size):
        x = self._make_rand_data(device, dtype, (batch_size, 1)).squeeze(-1)
        zs = torch.zeros_like(x)
        se2 = Se2.trans_x(x)
        self.assert_close(se2.t, torch.stack((x, zs), -1))
        self.assert_close(se2.so2.matrix(), So2.identity(batch_size, device, dtype).matrix())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_trans_y(self, device, dtype, batch_size):
        y = self._make_rand_data(device, dtype, (batch_size, 1)).squeeze(-1)
        zs = torch.zeros_like(y)
        se2 = Se2.trans_y(y)
        self.assert_close(se2.t, torch.stack((zs, y), -1))
        self.assert_close(se2.so2.matrix(), So2.identity(batch_size, device, dtype).matrix())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_adjoint(self, device, dtype, batch_size):
        x = Se2.random(batch_size)
        y = Se2.random(batch_size)
        self.assert_close(x.inverse().adjoint(), x.adjoint().inverse())
        self.assert_close((x * y).adjoint(), x.adjoint() @ y.adjoint())

    def test_user_leaf_translation_receives_the_gradient(self, device, dtype):
        # A tensor that requires grad is kept, not re-wrapped as a new Parameter, so the gradient reaches it (#4943).
        if dtype == torch.bfloat16:
            pytest.skip("torch has no complex bfloat16 dtype, which So2 stores its rotation in")
        t = torch.tensor([[1.0, 2.0]], device=device, dtype=dtype, requires_grad=True)
        s = Se2(So2.identity(1, device, dtype), t)
        (s * torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)).sum().backward()
        assert t.grad is not None
        self.assert_close(t.grad, torch.ones_like(t))
        assert "_translation" in s.state_dict()

    def test_derived_state_moves_and_serializes(self, device, dtype):
        # A group built from a tensor with autograd history keeps that history; its state must
        # still be registered so ``state_dict`` and ``.to()`` / ``.double()`` reach it.
        v = torch.rand(2, 3, device=device, dtype=dtype, requires_grad=True)
        s = Se2.exp(v)
        assert s.t.grad_fn is not None and s.so2.z.grad_fn is not None
        assert set(s.state_dict()) == {"_translation", "_rotation._z"}
        restored = Se2(So2.identity(2, device, dtype), torch.zeros(2, 2, device=device, dtype=dtype))
        restored.load_state_dict(s.state_dict())
        self.assert_close(restored.matrix(), s.matrix().detach())
        # ``.half()`` / ``.float()`` convert the floating buffers in place and keep the graph
        # (float64 is unavailable on MPS, so convert towards float16 from float32)
        converted = s.half() if dtype == torch.float32 else s.float()
        assert converted.t.dtype == (torch.float16 if dtype == torch.float32 else torch.float32)
        assert converted.t.grad_fn is not None
        converted.t.sum().backward()
        assert v.grad is not None
