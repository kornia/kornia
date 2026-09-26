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
import warnings

import pytest
import torch

from kornia.geometry.conversions import angle_to_rotation_matrix
from kornia.geometry.liegroup import So2
from kornia.geometry.vector import Vector2

from testing.base import BaseTester


class TestSo2(BaseTester):
    def _make_rand_data(self, device, dtype, input_shape):
        batch_size = input_shape[0]
        shape = input_shape[1:] if batch_size is None else input_shape
        return torch.rand(shape, device=device, dtype=dtype)

    @pytest.mark.parametrize("cdtype", (torch.cfloat, torch.cdouble))
    def test_smoke(self, device, cdtype):
        z = torch.randn(2, 1, dtype=cdtype, device=device)
        s = So2(z)
        assert isinstance(s, So2)
        self.assert_close(s.z.data, z.data)

    @pytest.mark.parametrize("input_shape", [(1,), (2,), (5,), ()])
    @pytest.mark.parametrize("cdtype", (torch.cfloat, torch.cdouble))
    def test_cardinality(self, device, dtype, input_shape, cdtype):
        z = torch.randn(input_shape, dtype=cdtype, device=device)
        s = So2(z)
        theta = torch.rand(input_shape, dtype=dtype, device=device)
        assert s.z.shape == input_shape
        assert (s * s).z.shape == input_shape
        assert s.exp(theta).z.shape == input_shape
        assert s.log().shape == input_shape
        if not any(input_shape):
            expected_hat_shape = (2, 2)
        else:
            expected_hat_shape = (input_shape[0], 2, 2)
        assert s.hat(theta).shape == expected_hat_shape
        assert s.inverse().z.shape == input_shape

    @pytest.mark.parametrize("input_shape", [(1, 2, 2), (2, 2, 2), (5, 2, 2), (2, 2)])
    def test_matrix_cardinality(self, device, dtype, input_shape):
        matrix = torch.rand(input_shape, dtype=dtype, device=device)
        matrix[..., 0, 1] = -matrix[..., 1, 0]
        matrix[..., 1, 1] = matrix[..., 0, 0]
        s = So2.from_matrix(matrix)
        assert s.matrix().shape == input_shape

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    @pytest.mark.parametrize("cdtype", (torch.cfloat, torch.cdouble))
    def test_exception(self, batch_size, device, dtype, cdtype):
        with pytest.raises(ValueError):
            z = torch.randn(batch_size, 2, dtype=cdtype, device=device)
            assert So2(z)
        with pytest.raises(TypeError):
            assert So2.identity(1, device, dtype) * [1.0, 2.0, 1.0]
        with pytest.raises(ValueError):
            theta = torch.rand((2, 2), dtype=dtype, device=device)
            assert So2.exp(theta)
        with pytest.raises(ValueError):
            theta = torch.rand((2, 2), dtype=dtype, device=device)
            assert So2.hat(theta)
        with pytest.raises(ValueError):
            m = torch.rand((2, 2, 1), dtype=dtype, device=device)
            assert So2.from_matrix(m)
        with pytest.raises(ValueError):
            m = torch.rand((2, 2, 1), dtype=dtype, device=device)
            assert So2.from_matrix(m)
        with pytest.raises(Exception):
            assert So2.identity(batch_size=0)

    # TODO: implement me
    def test_gradcheck(self, device):
        pass

    # TODO: implement me
    def test_jit(self, device, dtype):
        pass

    # TODO: implement me
    def test_module(self, device, dtype):
        pass

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    @pytest.mark.parametrize("cdtype", (torch.cfloat, torch.cdouble))
    def test_init(self, device, dtype, batch_size, cdtype):
        z1 = self._make_rand_data(device, cdtype, (batch_size,))
        z2 = self._make_rand_data(device, cdtype, (batch_size, 1))
        z3_real = self._make_rand_data(device, dtype, (batch_size,))
        z3_imag = self._make_rand_data(device, dtype, (batch_size,))
        z3 = torch.complex(z3_real, z3_imag)
        s1 = So2(z1)
        s2 = So2(s1.z)
        assert isinstance(s2, So2)
        self.assert_close(s1.z, s2.z)
        self.assert_close(So2(z1).z, z1)
        self.assert_close(So2(z2).z, z2)
        self.assert_close(So2(z3).z, z3)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    @pytest.mark.parametrize("cdtype", (torch.cfloat, torch.cdouble))
    def test_getitem(self, device, batch_size, cdtype):
        z = self._make_rand_data(device, cdtype, (batch_size,))
        s = So2(z)
        n = 1 if batch_size is None else batch_size
        for i in range(n):
            if batch_size is None:
                expected = s.z
                actual = z
            else:
                expected = s[i].z.data.squeeze()
                actual = z[i]
            self.assert_close(expected, actual)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_mul(self, device, dtype, batch_size):
        s1 = So2.identity(batch_size, device, dtype)
        z = self._make_rand_data(device, dtype, (batch_size, 2))
        s2 = So2(torch.complex(z[..., 0], z[..., 1]))
        t1 = self._make_rand_data(device, dtype, (batch_size, 2))
        t2 = self._make_rand_data(device, dtype, (2,))
        s1_pose_s2 = s1 * s2
        s2_pose_s2 = s2 * s2.inverse()
        self.assert_close(s1_pose_s2.z.real, s2.z.real)
        self.assert_close(s1_pose_s2.z.imag, s2.z.imag)
        self.assert_close(s2_pose_s2.z.real, s1.z.real)
        self.assert_close(s2_pose_s2.z.imag, s1.z.imag)
        self.assert_close((s1 * t1), t1)
        self.assert_close((So2.identity(device=device, dtype=dtype) * t2), t2)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_mul_vector(self, device, dtype, batch_size):
        s1 = So2.identity(batch_size, device, dtype)
        if batch_size is None:
            shape = ()
        else:
            shape = (batch_size,)
        t1 = Vector2.random(shape, device, dtype)
        t2 = Vector2.random(shape, device, dtype)
        self.assert_close((s1 * t1), t1)
        self.assert_close((So2.identity(device=device, dtype=dtype) * t2), t2)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_exp(self, device, dtype, batch_size):
        theta = self._make_rand_data(device, dtype, (batch_size, 1))
        s = So2.exp(theta)
        self.assert_close(s.z.real, theta.cos())
        self.assert_close(s.z.imag, theta.sin())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    @pytest.mark.parametrize("cdtype", (torch.cfloat, torch.cdouble))
    def test_log(self, device, batch_size, cdtype):
        z = self._make_rand_data(device, cdtype, (batch_size,))
        t = So2(z).log()
        self.assert_close(t, z.imag.atan2(z.real))

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_exp_log(self, device, dtype, batch_size):
        theta = self._make_rand_data(device, dtype, (batch_size, 1))
        self.assert_close(So2.exp(theta).log(), theta)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_hat(self, device, dtype, batch_size):
        theta = self._make_rand_data(device, dtype, (batch_size,))
        m = So2.hat(theta)
        o = torch.ones((2, 1), device=device, dtype=dtype)
        self.assert_close((m @ o).reshape(-1, 2, 1), theta.reshape(-1, 1, 1).repeat(1, 2, 1))

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_vee(self, device, dtype, batch_size):
        omega = self._make_rand_data(device, dtype, (batch_size, 2, 2))
        theta = So2.vee(omega)
        self.assert_close(omega[..., 0, 1], theta)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_hat_vee(self, device, dtype, batch_size):
        a = self._make_rand_data(device, dtype, (batch_size,))
        omega = So2.hat(a)
        b = So2.vee(omega)
        self.assert_close(b, a)

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
        matrix = torch.eye(2, device=device, dtype=dtype)
        if batch_size is not None:
            matrix = matrix.repeat(batch_size, 1, 1)
            one = torch.ones((batch_size,), device=device, dtype=dtype)
            zero = torch.zeros((batch_size,), device=device, dtype=dtype)
        else:
            one = torch.tensor(1, device=device, dtype=dtype)
            zero = torch.tensor(0, device=device, dtype=dtype)
        s = So2.from_matrix(matrix)
        self.assert_close(s.z.real, one)
        self.assert_close(s.z.imag, zero)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    @pytest.mark.parametrize("cdtype", (torch.cfloat, torch.cdouble))
    def test_inverse(self, device, batch_size, cdtype):
        z = self._make_rand_data(device, cdtype, (batch_size,))
        s = So2(z)
        s_in_in = s.inverse().inverse()
        self.assert_close(s_in_in.z.real, z.real)
        self.assert_close(s_in_in.z.imag, z.imag)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_random(self, device, dtype, batch_size):
        s = So2.random(batch_size=batch_size, device=device, dtype=dtype)
        s_in_s = s.inverse() * s
        i = So2.identity(batch_size=batch_size, device=device, dtype=dtype)
        self.assert_close(s_in_s.z.real, i.z.real)
        self.assert_close(s_in_s.z.imag, i.z.imag)

    def test_random_is_a_uniform_unit_rotation_4930(self, device, dtype):
        # #4930: random drew independent uniform real and imaginary parts on [0, 1), so |z| ranged over (0, sqrt 2),
        # matrix() was a rotation scaled by |z| with det = |z|**2 down to 3.5e-6, and every angle was in [0, pi / 2].
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("torch.complex has no bfloat16 overload and linalg.det has no float16 CPU kernel")
        torch.manual_seed(0)
        s = So2.random(1000, device=device, dtype=dtype)
        self.assert_close(s.z.abs(), torch.ones(1000, device=device, dtype=dtype))
        self.assert_close(torch.linalg.det(s.matrix()), torch.ones(1000, device=device, dtype=dtype))
        # a uniform angle on [-pi, pi) puts about 250 of 1000 draws in each quadrant (standard deviation 14)
        theta = s.log()
        counts = [
            int(((lo <= theta) & (theta < lo + torch.pi / 2)).sum())
            for lo in (-torch.pi, -torch.pi / 2, 0.0, torch.pi / 2)
        ]
        assert min(counts) > 150, counts
        # the draws reach both ends of the range: P(none of 1000 within 0.05 of -pi, or of pi) = 3e-4 each
        assert theta.min() < -torch.pi + 0.05 and theta.max() > torch.pi - 0.05, (theta.min(), theta.max())
        self.assert_close(So2.random(device=device, dtype=dtype).z.abs(), torch.tensor(1.0, device=device, dtype=dtype))

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_adjoint(self, device, dtype, batch_size):
        s = So2.identity(batch_size, device=device, dtype=dtype)
        self.assert_close(s.matrix(), s.adjoint())

    def test_user_leaf_receives_the_gradient(self, device, dtype):
        # A complex leaf that requires grad is kept, not re-wrapped as a new Parameter, so the gradient reaches it
        # (#4943). d/dz of sum(R @ (1, 0)) = d(re + im)/dz.
        if dtype == torch.bfloat16:
            pytest.skip("torch has no complex bfloat16 dtype, which So2 stores its rotation in")
        re = torch.tensor([0.6], device=device, dtype=dtype)
        z = torch.complex(re, re + 0.2).requires_grad_(True)
        s = So2(z)
        (s * torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)).sum().backward()
        assert z.grad is not None
        assert "_z" in s.state_dict()

    def test_derived_state_moves_and_serializes(self, device, dtype):
        theta = torch.rand(2, device=device, dtype=dtype, requires_grad=True)
        s = So2.exp(theta)
        assert s.z.grad_fn is not None
        assert list(s.state_dict()) == ["_z"] == list(So2.identity(2, device, dtype).state_dict())
        assert dict(s.named_buffers()).keys() == {"_z"}  # registered, so ``.to(device)`` reaches it
        restored = So2.identity(2, device, dtype)
        restored.load_state_dict(s.state_dict())
        self.assert_close(restored.matrix(), s.matrix().detach())
        s.to(device).matrix().sum().backward()
        assert theta.grad is not None

    def test_convention_so2_positive_angle_rotates_x_toward_y(self, device, dtype):
        if dtype == torch.bfloat16:
            pytest.skip("torch has no complex bfloat16 dtype, which So2 stores its rotation in")
        # A positive angle turns the x axis toward the y axis, counter-clockwise in a y-up frame, and matrix() is
        # [[cos, -sin], [sin, cos]]. Reference: math.cos(0.3), math.sin(0.3).
        c, s = 0.955336489125606, 0.29552020666133955
        g = So2.exp(torch.tensor(0.3, device=device, dtype=dtype))
        axes = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device, dtype=dtype)
        rotated = g * axes
        assert rotated[0, 1] > 0.25  # x moved toward +y
        self.assert_close(rotated, torch.tensor([[c, s], [-s, c]], device=device, dtype=dtype))
        self.assert_close(g.matrix(), torch.tensor([[c, -s], [s, c]], device=device, dtype=dtype))

    def test_convention_so2_is_transpose_of_angle_to_rotation_matrix(self, device, dtype):
        if dtype == torch.bfloat16:
            pytest.skip("torch has no complex bfloat16 dtype, which So2 stores its rotation in")
        # kornia.geometry.conversions.angle_to_rotation_matrix takes degrees and returns [[cos, sin], [-sin, cos]],
        # the transpose of So2's matrix: the same angle turns the other way.
        theta = torch.tensor([0.3, -1.2], device=device, dtype=dtype)
        m = So2.exp(theta).matrix()
        assert (m - m.mT).abs().max() > 0.5  # a non-symmetric fixture, so a transpose is visible
        self.assert_close(m, angle_to_rotation_matrix(torch.rad2deg(theta)).mT)

    def test_convention_so2_log_is_principal(self, device, dtype):
        if dtype == torch.bfloat16:
            pytest.skip("torch has no complex bfloat16 dtype, which So2 stores its rotation in")
        # log returns the angle in [-pi, pi], so exp(3.5) logs to 3.5 - 2 pi. Reference: 3.5 - 2 * math.pi.
        theta = torch.tensor([3.5, -3.5, 0.3], device=device, dtype=dtype)
        g = So2.exp(theta)
        expected = torch.tensor([3.5 - 2 * math.pi, 2 * math.pi - 3.5, 0.3], device=device, dtype=dtype)
        self.assert_close(g.log(), expected)
        self.assert_close(So2.exp(g.log()).matrix(), g.matrix())

    def test_convention_so2_from_matrix_rejects_a_reflection(self, device, dtype):
        if dtype == torch.bfloat16:
            pytest.skip("torch has no complex bfloat16 dtype, which So2 stores its rotation in")
        # from_matrix checks m00 == m11 and m01 == -m10. The first reflection fails only the diagonal check, the
        # axis swap only the off-diagonal one.
        for reflection in ([[1.0, 0.0], [0.0, -1.0]], [[0.0, 1.0], [1.0, 0.0]]):
            reflection = torch.tensor(reflection, device=device, dtype=dtype)
            assert torch.linalg.det(reflection.float()) < 0
            with pytest.raises(ValueError, match="Invalid SO2 rotation matrix"):
                So2.from_matrix(reflection)
        rotation = So2.exp(torch.tensor(0.3, device=device, dtype=dtype)).matrix().detach()
        self.assert_close(So2.from_matrix(rotation).log(), torch.tensor(0.3, device=device, dtype=dtype))

    def test_wart_so2_random_is_not_a_unit_rotation_4930(self, device, dtype):
        if dtype == torch.bfloat16:
            pytest.skip("torch has no complex bfloat16 dtype, which So2 stores its rotation in")
        # Unseeded on purpose: 25% of the unit square lies within 0.1 of the unit circle, so all 1000 draws land
        # there with probability 0.25^1000, and seeding here would shift the global RNG stream of every later test.
        s = So2.random(1000, device=device, dtype=dtype)
        radius = (s.z.real**2 + s.z.imag**2).sqrt()
        # https://github.com/kornia/kornia/issues/4930: both parts of z are drawn from U[0, 1), so |z| spans
        # [0, sqrt(2)) and every angle lies in [0, pi / 2]. A uniform rotation has |z| = 1 and angles of both signs.
        assert (radius - 1).abs().max() > 0.1
        assert s.log().min() >= 0

    def test_wart_so2_column_angle_outer_broadcasts_4932(self, device, dtype):
        if dtype == torch.bfloat16:
            pytest.skip("torch has no complex bfloat16 dtype, which So2 stores its rotation in")
        theta = torch.tensor([0.3, 0.5, 0.7], device=device, dtype=dtype)
        p = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]], device=device, dtype=dtype)
        paired = So2.exp(theta) * p  # the (B,) layout rotates point i by angle i
        assert paired.shape == (3, 2)
        # https://github.com/kornia/kornia/issues/4932: the documented (B, 1) layout broadcasts z against the (B,)
        # coordinates, so entry [i, j] is R(theta_i) p_j: every rotation applied to every point.
        out = So2.exp(theta[:, None]) * p
        assert out.shape == (3, 3, 2)
        self.assert_close(out.diagonal(dim1=0, dim2=1).mT, paired)
        self.assert_close(out[0, 2], So2.exp(theta[0]) * p[2])

    def test_wart_so2_real_dtype_cast_drops_the_imaginary_part_4923(self, device, dtype):
        if dtype == torch.bfloat16:
            pytest.skip("torch has no complex bfloat16 dtype, which So2 stores its rotation in")
        theta = torch.tensor([0.3, -1.2], device=device, dtype=dtype)
        g = So2.exp(theta)
        assert g.z.is_complex()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # torch warns, once per process, that the cast discards the imaginary part
            cast = g.to(torch.float32)
        # https://github.com/kornia/kornia/issues/4923: nn.Module.to(float32) casts the complex state z to a real
        # tensor, which keeps cos(theta) and drops sin(theta), so the rotation is lost and matrix() raises.
        assert not cast.z.is_complex()
        self.assert_close(cast.z, theta.cos().float())
        with pytest.raises(RuntimeError):
            cast.matrix()
