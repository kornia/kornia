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
from typing import Union

import pytest
import torch
from torch.nn import Parameter

from kornia.geometry.quaternion import Quaternion, average_quaternions

from testing.base import BaseTester


class TestQuaternion(BaseTester):
    def _make_rand_data(self, device, dtype, batch_size):
        shape = [] if batch_size is None else [batch_size]
        return torch.rand([*shape, 4], device=device, dtype=dtype)

    def test_smoke(self, device, dtype):
        q = Quaternion.from_coeffs(1.0, 0.0, 0.0, 0.0)
        q = q.to(device, dtype)
        q_data = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
        assert isinstance(q, Quaternion)
        assert q.shape == (4,)
        self.assert_close(q.data, q_data)
        self.assert_close(q.q, q_data)
        self.assert_close(q.real, q_data[..., 0])
        self.assert_close(q.scalar, q_data[..., 0])
        self.assert_close(q.vec, q_data[..., 1:])

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_init(self, device, dtype, batch_size):
        q1 = Quaternion.identity(batch_size, device, dtype)
        q2 = Quaternion(q1.data)
        assert isinstance(q2, Quaternion)
        self.assert_close(q1, q2)

    def test_init_fail(self, device, dtype):
        with pytest.raises(Exception):
            _ = Quaternion("q")

        with pytest.raises(Exception):
            _ = Quaternion([1, 0, 0, 0])

        with pytest.raises(Exception):
            _ = Quaternion(1, [0, 0, 0])

    def test_constructor_scalar_tensre(self, device, dtype):
        with pytest.raises(ValueError):
            Quaternion(torch.tensor(1.0, device=device, dtype=dtype))

    def test_constructor_wrong_last_dim(self, device, dtype):
        with pytest.raises(ValueError):
            Quaternion(torch.randn(3, 5, device=device, dtype=dtype))

    def test_constructor_valid_shape(self, device, dtype):
        data = torch.randn(2, 4, device=device, dtype=dtype)
        q = Quaternion(data)
        self.assert_close(q.data, data)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_random(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size, device, dtype)
        q_n = q.normalize().norm()
        self.assert_close(q_n, q_n.new_ones(q_n.shape))

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_minus(self, device, dtype, batch_size):
        data = self._make_rand_data(device, dtype, batch_size)
        q = Quaternion(data)
        q = q.to(device, dtype)
        self.assert_close(-q, -data)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_add(self, device, dtype, batch_size):
        d1 = self._make_rand_data(device, dtype, batch_size)
        d2 = self._make_rand_data(device, dtype, batch_size)
        q1 = Quaternion(d1)
        q2 = Quaternion(d2)
        q3 = q1 + q2
        assert isinstance(q3, Quaternion)
        self.assert_close(q3, d1 + d2)
        q1 += q2
        self.assert_close(q1, q3)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_subtract(self, device, dtype, batch_size):
        d1 = self._make_rand_data(device, dtype, batch_size)
        d2 = self._make_rand_data(device, dtype, batch_size)
        q1 = Quaternion(d1)
        q2 = Quaternion(d2)
        q3 = q1 - q2
        assert isinstance(q3, Quaternion)
        self.assert_close(q3, d1 - d2)
        q1 -= q2
        self.assert_close(q1, q3)

    def test_multiplication_of_bases(self, device, dtype):
        one = Quaternion.from_coeffs(1.0, 0.0, 0.0, 0.0).to(device, dtype)
        i = Quaternion.from_coeffs(0.0, 1.0, 0.0, 0.0).to(device, dtype)
        j = Quaternion.from_coeffs(0.0, 0.0, 1.0, 0.0).to(device, dtype)
        k = Quaternion.from_coeffs(0.0, 0.0, 0.0, 1.0).to(device, dtype)

        self.assert_close(i * i, j * j)
        self.assert_close(j * j, k * k)
        self.assert_close(k * k, i * j * k)
        self.assert_close(i * j * k, -one)

        self.assert_close(i * j, k)
        self.assert_close(i * i, -one)
        self.assert_close(i * k, -j)
        self.assert_close(j * i, -k)
        self.assert_close(j * j, -one)
        self.assert_close(j * k, i)
        self.assert_close(k * i, j)
        self.assert_close(k * j, -i)
        self.assert_close(k * k, -one)
        self.assert_close(i * j * k, -one)

    def test_division_of_bases(self, device, dtype):
        one = Quaternion.from_coeffs(1.0, 0.0, 0.0, 0.0).to(device, dtype)
        i = Quaternion.from_coeffs(0.0, 1.0, 0.0, 0.0).to(device, dtype)
        j = Quaternion.from_coeffs(0.0, 0.0, 1.0, 0.0).to(device, dtype)
        k = Quaternion.from_coeffs(0.0, 0.0, 0.0, 1.0).to(device, dtype)

        self.assert_close(i / i, j / j)
        self.assert_close(j / j, k / k)
        self.assert_close(k / k, one)
        self.assert_close(k / -k, -one)

        self.assert_close(i / j, -k)
        self.assert_close(i / i, one)
        self.assert_close(i / k, j)
        self.assert_close(j / i, k)
        self.assert_close(j / j, one)
        self.assert_close(j / k, -i)
        self.assert_close(k / i, -j)
        self.assert_close(k / j, i)
        self.assert_close(k / k, one)
        self.assert_close(i / -j, k)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_pow(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size, device, dtype)
        q1 = Quaternion.identity(batch_size, device, dtype)
        self.assert_close(q**0, q1)
        self.assert_close(q**1, q)
        self.assert_close(q**2, q * q)
        self.assert_close(q**-1, q.inv())
        self.assert_close((q**0.5) * (q**0.5), q)
        self.assert_close((q1**1), q1)
        self.assert_close((q1**2), q1)

    def test_pow_non_unit(self, device, dtype):
        # issue #4926: q**t keeps the norm, |q**t| == |q|**t, so it agrees with * and inv() for non-unit q
        data = [[1.0, 0.5, 0.0, 0.0], [0.5, -0.25, 0.5, 0.25], [1.5, 0.0, 0.5, -0.5], [0.0, 0.0, 0.75, 0.0]]
        q = Quaternion(torch.tensor(data, device=device, dtype=dtype))
        self.assert_close(q**0, Quaternion.identity(4, device, dtype))
        self.assert_close(q**1, q)
        self.assert_close(q**2, q * q)
        self.assert_close(q**-1, q.inv())
        self.assert_close((q**0.5) * (q**0.5), q)
        self.assert_close((q**0.5).norm(), q.norm() ** 0.5)
        # the direction is the power of the unit quaternion, as before
        self.assert_close((q**0.5).normalize(), q.normalize() ** 0.5)

    def test_pow_real_axis(self, device, dtype):
        q = Quaternion(torch.tensor([[2.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype))
        self.assert_close(q**2, q * q)
        self.assert_close(q**-1, q.inv())
        expected = torch.tensor([2.0**0.5, 0.0, 0.0, 0.0], device=device, dtype=dtype)
        self.assert_close((q**0.5).data[0], expected)

    @pytest.mark.parametrize("t", (-1.0, 0.5, 2.0))
    def test_pow_gradcheck(self, device, t):
        # the first quaternion lies on the real axis, where the vector part has zero norm; the last is pure imaginary
        # (w = 0), where the unselected real-axis arm divides by w
        data = torch.tensor(
            [[2.0, 0.0, 0.0, 0.0], [1.0, 0.5, -0.3, 0.2], [0.0, 0.3, -0.4, 0.5]], device=device, dtype=torch.float64
        )
        self.gradcheck(lambda x: (Quaternion(x) ** t).data, (data,))

    @pytest.mark.parametrize("t", (-1.0, 2.0, 3.0))
    def test_pow_gradcheck_negative_real_axis(self, device, t):
        # q = -2 has theta = pi, where the real-axis limit t * cos(t * theta) / w carries the signs of cos(t * pi) and
        # of w. Only integer t: for a non-integer t the negative real axis is a branch cut with no derivative.
        data = torch.tensor([[-2.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float64)
        self.gradcheck(lambda x: (Quaternion(x) ** t).data, (data,))

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_quaternion_scalar_multiplication(self, device, dtype, batch_size):
        """Test scalar multiplication for issue #3101."""
        # Create a quaternion with parameters to test gradient flow
        q_data = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype, requires_grad=True)
        q = Quaternion(Parameter(q_data))

        # This should not raise a TypeError
        result = q * q * 5

        # Verify the result has proper gradient tracking
        assert result.data.requires_grad

        # Backward pass should work
        loss = result.data.sum()
        loss.backward()

        # The quaternion's parameter should have gradients
        assert q.data.grad is not None

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_inverse(self, device, dtype, batch_size):
        q1 = Quaternion.random(batch_size, device, dtype)
        q2 = Quaternion.identity(batch_size, device, dtype)
        self.assert_close(q1 * q1.inv(), q2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_conjugate(self, device, dtype, batch_size):
        q1 = Quaternion.random(batch_size, device, dtype)
        q2 = Quaternion.random(batch_size, device, dtype)
        self.assert_close((q1 * q2).conj(), q2.conj() * q1.conj())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_double_conjugate(self, device, dtype, batch_size):
        q1 = Quaternion.random(batch_size, device, dtype)
        self.assert_close(q1, q1.conj().conj())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_norm(self, device, dtype, batch_size):
        q1 = Quaternion.random(batch_size, device, dtype)
        q2 = Quaternion.random(batch_size, device, dtype)
        self.assert_close((q1 * q2).norm(), q1.norm() * q2.norm())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_norm_shape(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size, device, dtype)
        expected_shape = () if batch_size is None else (batch_size,)
        self.assert_close(tuple(q.norm().shape), expected_shape)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_normalize(self, device, dtype, batch_size):
        q1 = Quaternion.random(batch_size, device, dtype)
        q1_n = q1.normalize().norm()
        self.assert_close(q1_n, q1_n.new_ones(q1_n.shape))

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_matrix(self, device, dtype, batch_size):
        q1 = Quaternion.random(batch_size, device, dtype)
        m1 = q1.matrix()
        q2 = Quaternion.from_matrix(m1)
        for qq1, qq2 in zip(q1.data, q2.data):
            try:
                self.assert_close(qq1, qq2)
            except Exception:
                self.assert_close(qq1, -qq2)

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_getitem(self, device, dtype, batch_size):
        q = Quaternion.random(batch_size, device, dtype)
        for i in range(batch_size):
            q1 = q[i]
            self.assert_close(q1.data, q.data[i])

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_axis_angle(self, device, dtype, batch_size):
        q1 = Quaternion.random(batch_size, device, dtype)
        angle = 2 * q1.scalar.arccos()[..., None]
        axis = q1.vec / (angle / 2).sin()
        axis_angle = axis * angle
        q2 = Quaternion.from_axis_angle(axis_angle)
        q2 = q2.to(device, dtype)
        self.assert_close(q1, q2)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_to_axis_angle(self, device, dtype, batch_size):
        # batch_s = 5
        # random_coefs = Quaternion.random(batch_s).data
        random_coefs = torch.tensor(
            [
                [2.5398e-04, -2.2677e-01, -8.3897e-01, 4.9467e-01],
                [-1.7005e-01, -1.0974e-01, 3.7635e-01, -9.0410e-01],
                [9.1273e-01, 4.8935e-02, -6.2994e-03, 4.0558e-01],
                [-9.8316e-01, 5.4078e-03, 1.4471e-01, 1.1145e-01],
                [4.5794e-02, -7.0831e-01, 6.7577e-01, 1.9883e-01],
            ],
            device=device,
            dtype=dtype,
        )

        q = Quaternion(random_coefs)
        axis_angle_actual = q.to_axis_angle()

        axis_angle_expected = torch.tensor(
            [
                [-0.7123, -2.6353, 1.5538],
                [0.3118, -1.0693, 2.5687],
                [0.1008, -0.0130, 0.8356],
                [-0.0109, -0.2911, -0.2242],
                [-2.1626, 2.0632, 0.6071],
            ],
            device=device,
            dtype=dtype,
        )

        self.assert_close(axis_angle_expected, axis_angle_actual, 1e-4, 1e-4)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_slerp(self, device, dtype, batch_size):
        for axis in torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]):
            axis = axis.to(device, dtype)
            if batch_size is not None:
                axis = axis.repeat(batch_size, 1)
            q1 = Quaternion.from_axis_angle(axis * 0)
            q1.to(device, dtype)
            q2 = Quaternion.from_axis_angle(axis * 3.14159)
            q2.to(device, dtype)
            for t in torch.linspace(0.1, 1, 10):
                q3 = q1.slerp(q2, t)
                q4 = Quaternion.from_axis_angle(axis * t * 3.14159)
                self.assert_close(q3, q4)

    def test_slerp_takes_the_short_arc(self, device, dtype):
        # q and -q are the same rotation; the interpolation must not depend on the stored sign (#4944).
        # a and b are 0.33 rad apart with dot(a, b) = 0.986, so dot(a, -b) < 0.
        a = Quaternion.from_axis_angle(torch.tensor([[0.3, 0.2, -0.1]], device=device, dtype=dtype))
        b = Quaternion.from_axis_angle(torch.tensor([[0.5, 0.1, 0.15]], device=device, dtype=dtype))
        rel_angle = (a.inv() * b).to_axis_angle().norm()
        for t in (0.25, 0.5, 0.75):
            short = a.slerp(b, t).matrix()
            self.assert_close(a.slerp(-b, t).matrix(), short)
            # the interpolant sits at t times the relative angle from a
            self.assert_close((a.inv() * a.slerp(-b, t)).to_axis_angle().norm(), t * rel_angle)

    def test_slerp_exact_half_turn_follows_the_stored_sign(self, device, dtype):
        # At an exact half turn both arcs are equally short. The arc follows the sign of the vector part of
        # q0^-1 q1, so q1 and -q1 take opposite arcs, each of length t * pi from q0 (see the slerp docstring).
        # Every product below is exact, so the real part of q0^-1 q1 is exactly zero in every dtype.
        q0 = Quaternion(torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=device, dtype=dtype))
        q1 = q0 * Quaternion(torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device, dtype=dtype))
        for t in (0.25, 0.5):
            for sign in (1.0, -1.0):
                rel = (q0.inv() * q0.slerp(q1 * sign, t)).to_axis_angle()
                expected = torch.tensor([[sign * t * math.pi, 0.0, 0.0]], device=device, dtype=dtype)
                self.assert_close(rel, expected)

    def test_slerp_gradient_is_finite_at_equal_endpoints(self, device, dtype):
        # slerp(q, q, t) = q is smooth in both endpoints; its gradient must not be nan (#4927).
        q = Quaternion.from_axis_angle(torch.tensor([[0.3, 0.2, -0.1]], device=device, dtype=dtype)).data
        q0 = q.clone().requires_grad_(True)
        q1 = q.clone().requires_grad_(True)
        Quaternion(q0).slerp(Quaternion(q1), 0.3).data.sum().backward()
        assert torch.isfinite(q0.grad).all()
        assert torch.isfinite(q1.grad).all()

    def test_slerp_gradcheck(self, device):
        q0 = Quaternion.from_axis_angle(torch.tensor([[0.3, 0.2, -0.1]], device=device, dtype=torch.float64)).data
        q1 = Quaternion.from_axis_angle(torch.tensor([[0.5, 0.1, 0.15]], device=device, dtype=torch.float64)).data
        self.gradcheck(lambda a, b: Quaternion(a).slerp(Quaternion(b), 0.3).data, (q0, q1))
        self.gradcheck(lambda a, b: Quaternion(a).slerp(Quaternion(b), 0.3).data, (q0, q0.detach().clone()))

    def test_from_to_euler_values(self, device, dtype):
        # num_samples = 5
        # data = 2 * torch.rand(3, num_samples, device=device, dtype=dtype) - 1
        # roll, pitch, yaw = torch.pi * data
        roll = torch.tensor(
            [2.6518599987, 0.0612506270, 1.2417907715, 2.8829660416, -1.9961174726, 0], device=device, dtype=dtype
        )

        pitch = torch.tensor(
            [2.3267219067, -2.7309591770, -1.4011553526, -2.1962766647, 2.1454355717, 0], device=device, dtype=dtype
        )

        yaw = torch.tensor(
            [-0.8856627345, 0.2605336905, 0.4579202533, -1.3095731735, 0.6096843481, 0], device=device, dtype=dtype
        )

        euler_expected = torch.tensor(
            [
                [-0.4897327125, 0.8148705959, 2.2559301853],
                [-3.0803420544, -0.4106334746, -2.8810589314],
                [1.2417914867, -1.4011553526, 0.4579201937],
                [-0.2586266696, -0.9453159571, 1.8320195675],
                [1.1454752684, 0.9961569905, -2.5319085121],
                [0, 0, 0],
            ],
            device=device,
            dtype=dtype,
        )

        q = Quaternion.from_euler(roll, pitch, yaw)
        euler = q.to_euler()
        euler = torch.stack(euler, -1)

        self.assert_close(euler, euler_expected, 1e-4, 1e-4)


def _to_tensor(x: Union[torch.Tensor, Quaternion]) -> torch.Tensor:
    # Unwrap Quaternion/Parameter to a plain Tensor for comparisons
    if isinstance(x, Quaternion):
        x = x.data
    if isinstance(x, torch.nn.Parameter):
        x = x.data
    if x.ndim == 2 and x.shape[0] == 1:
        x = x.squeeze(0)
    return x


def _align_sign(q: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    # Flip sign of q so it points roughly in the same direction as ref
    if torch.dot(q, ref) < 0:
        return -q
    return q


class TestQuaternionAverage(BaseTester):
    @pytest.mark.parametrize("M", [1, 2, 5, 10])
    def test_average_identity(self, device, dtype, M):
        """All identity quaternions → should return identity"""
        Q = Quaternion.identity(M, device=device, dtype=dtype)
        out = average_quaternions(Q)
        q = _to_tensor(out)

        expected = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
        q = _align_sign(q, expected)

        self.assert_close(q, expected, rtol=1e-6, atol=1e-6)

    def test_output_is_normalized(self, device, dtype):
        """Averaged quaternion should always have unit norm"""
        Q = Quaternion.random(6, device=device, dtype=dtype).normalize()
        out = average_quaternions(Q)
        q = _to_tensor(out)

        self.assert_close(q.norm(), torch.tensor(1.0, device=device, dtype=dtype), rtol=1e-6, atol=1e-6)

    def test_weighted_bias(self, device, dtype):
        """Heavier weights should bias the average toward the corresponding quaternion"""
        q1 = Quaternion(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype))
        q2 = Quaternion(torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device, dtype=dtype))
        Q = Quaternion(torch.cat([q1.data, q2.data], dim=0))

        w = torch.tensor([0.9, 0.1], device=device, dtype=dtype)
        out = average_quaternions(Q, w=w)
        q = _to_tensor(out)

        dot1 = torch.dot(q, q1.data.squeeze())
        dot2 = torch.dot(q, q2.data.squeeze())
        assert dot1 > dot2  # should align closer to q1

    def test_single_quaternion_returns_itself(self, device, dtype):
        """Averaging a single quaternion should return it"""
        q = Quaternion.random(1, device=device, dtype=dtype).normalize()
        out = average_quaternions(q)
        out_t = _to_tensor(out)
        q_t = _to_tensor(q)

        out_t = _align_sign(out_t, q_t)
        self.assert_close(out_t, q_t, rtol=1e-6, atol=1e-6)

    def test_opposite_quaternions(self, device, dtype):
        """Opposite quaternions should average to something consistent (sign ambiguity)"""
        q1 = Quaternion.identity(1, device=device, dtype=dtype)
        q2 = Quaternion(-q1.data.clone())
        Q = Quaternion(torch.cat([q1.data, q2.data], dim=0))

        out = average_quaternions(Q)
        q = _to_tensor(out)

        # Should still be a valid unit quaternion
        self.assert_close(q.norm(), torch.tensor(1.0, device=device, dtype=dtype), rtol=1e-6, atol=1e-6)

    def test_invalid_weights_raise(self, device, dtype):
        """Mismatched number of weights should raise"""
        Q = Quaternion.random(3, device=device, dtype=dtype)
        w = torch.tensor([0.5, 0.5], device=device, dtype=dtype)  # wrong length
        with pytest.raises(ValueError):
            average_quaternions(Q, w=w)


class TestQuaternionConventions(BaseTester):
    def _unit(self, data, device, dtype):
        # normalised in float64 and rounded once, so the fixture is unit to the working dtype's resolution
        q = torch.tensor(data, dtype=torch.float64)
        return (q / q.norm(dim=-1, keepdim=True)).to(device=device, dtype=dtype)

    def test_convention_quaternion_storage_is_wxyz(self, device, dtype):
        # The data is (w, x, y, z), real part first. Expected matrix from scipy:
        #   Rotation.from_quat([0.8, 0.2, -0.4, 0.4], scalar_first=True).as_matrix()
        data = torch.tensor([[0.8, 0.2, -0.4, 0.4]], device=device, dtype=dtype)
        expected = torch.tensor(
            [[[0.36, -0.8, -0.48], [0.48, 0.6, -0.64], [0.8, 0.0, 0.6]]], device=device, dtype=dtype
        )
        q = Quaternion(data)
        self.assert_close(q.w, data[:, 0])
        self.assert_close(q.vec, data[:, 1:])
        self.assert_close(q.matrix(), expected)
        # the same four numbers read as (x, y, z, w) are another rotation, far from this one
        assert (Quaternion(data[:, [1, 2, 3, 0]]).matrix() - expected).abs().max() > 1.0

    def test_convention_quaternion_mul_is_hamilton_matrix_product(self, device, dtype):
        q1 = Quaternion(self._unit([[0.9, 0.1, -0.3, 0.2]], device, dtype))
        q2 = Quaternion(self._unit([[0.7, -0.4, 0.2, 0.5]], device, dtype))
        # precondition: the pair does not commute (|q1 q2 - q2 q1| = 2 |v1 x v2| = 0.53)
        assert ((q1 * q2).data - (q2 * q1).data).norm() > 0.1
        # q1 * q2 is the Hamilton product, the rotation "q2 first, then q1": its matrix is R1 @ R2
        self.assert_close((q1 * q2).matrix(), q1.matrix() @ q2.matrix())

    def test_convention_quaternion_scalar_operand_is_real_part(self, device, dtype):
        # A Python number or a tensor operand is the real quaternion (s, 0, 0, 0): + and - move only w, and * and /
        # scale all four components.
        data = torch.tensor([[0.75, 0.125, -0.375, 0.25]], device=device, dtype=dtype)
        q = Quaternion(data)
        real = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
        self.assert_close((q + 1.0).data, data + real)
        self.assert_close((1.0 + q).data, data + real)
        self.assert_close((q - 0.5).data, data - 0.5 * real)
        self.assert_close((q + torch.tensor([2.0], device=device, dtype=dtype)).data, data + 2.0 * real)
        self.assert_close((q * 2.0).data, 2.0 * data)
        self.assert_close((2.0 * q).data, 2.0 * data)
        self.assert_close((q * torch.tensor([2.0], device=device, dtype=dtype)).data, 2.0 * data)
        self.assert_close((q / 2.0).data, 0.5 * data)
        # a tensor holds one scalar per quaternion of the batch
        batch = torch.cat((data, 2.0 * data))
        per_item = torch.tensor([2.0, 3.0], device=device, dtype=dtype)
        self.assert_close((Quaternion(batch) + per_item).data, batch + per_item[:, None] * real)
        self.assert_close((Quaternion(batch) * per_item).data, batch * per_item[:, None])

    def test_convention_quaternion_matrix_normalises(self, device, dtype):
        data = torch.tensor([[2.0, 0.2, -0.6, 0.4]], device=device, dtype=dtype)
        q = Quaternion(data)
        # precondition: |q| = sqrt(4.56) = 2.14, far from unit
        assert (q.norm() - 1.0).abs().max() > 1.0
        # matrix() is the rotation of q / |q|, a proper rotation (So3.matrix() is not, #4942)
        rotation = q.matrix()
        self.assert_close(rotation, q.normalize().matrix())
        eye = torch.eye(3, device=device, dtype=dtype)[None]
        self.assert_close(rotation @ rotation.transpose(-2, -1), eye)

    def test_convention_quaternion_polar_angle_is_half_rotation_angle(self, device, dtype):
        # polar_angle is the angle a of q = |q| (cos a + n sin a), half the rotation angle, whatever |q|.
        # The rotation is 0.788 rad about an off-axis unit vector, built in float64 and rounded once.
        axis_angle = 0.788 * torch.tensor([[0.48, 0.6, 0.64]], dtype=torch.float64)
        data = Quaternion.from_axis_angle(axis_angle).data.to(device=device, dtype=dtype)
        expected = torch.tensor([0.394], device=device, dtype=dtype)
        angle = Quaternion(data).polar_angle
        assert angle.shape == (1,)
        self.assert_close(angle, expected)
        self.assert_close(Quaternion(3.0 * data).polar_angle, expected)
        # the range is [0, pi]: -q, the same rotation, has the supplementary angle
        self.assert_close(Quaternion(-data).polar_angle, math.pi - expected)

    def test_convention_quaternion_slerp_normalises_inputs(self, device, dtype):
        q1 = self._unit([[0.9, 0.1, -0.3, 0.2]], device, dtype)
        q2 = self._unit([[0.7, -0.4, 0.2, 0.5]], device, dtype)
        # precondition: dot(q1, q2) = 0.67 > 0, so the short arc ends at q2 itself and not at -q2
        assert (q1 * q2).sum() > 0.5
        a, b = Quaternion(2.0 * q1), Quaternion(3.0 * q2)
        # both endpoints are normalised first, and the result is a unit quaternion
        self.assert_close(a.slerp(b, 0.0).data, q1)
        self.assert_close(a.slerp(b, 1.0).data, q2)
        # Expected value from scipy, whose Slerp also normalises:
        #   Slerp([0, 1], Rotation.from_quat([q1, q2], scalar_first=True))([0.3]).as_quat(scalar_first=True)
        expected = torch.tensor(
            [[0.9297807322749576, -0.06174707057580068, -0.1602249202393802, 0.3256118304052157]],
            device=device,
            dtype=dtype,
        )
        self.assert_close(a.slerp(b, 0.3).data, expected)

    def test_convention_quaternion_slerp_extrapolates_outside_unit_interval(self, device, dtype):
        q1 = Quaternion(self._unit([[0.9, 0.1, -0.3, 0.2]], device, dtype))
        q2 = Quaternion(self._unit([[0.7, -0.4, 0.2, 0.5]], device, dtype))
        # t is not validated or clamped: t = 2 continues the arc by one more step, q1 (q1^-1 q2)^2 = q2 q1^-1 q2,
        # and t = -1 steps back from q1, q1 (q1^-1 q2)^-1 = q1 q2^-1 q1
        self.assert_close(q1.slerp(q2, 2.0).data, (q2 * q1.conj() * q2).data)
        self.assert_close(q1.slerp(q2, -1.0).data, (q1 * q2.conj() * q1).data)

    def test_convention_average_quaternions_is_the_chordal_mean(self, device, dtype):
        # The weighted chordal (eigenvector) mean of Markley et al., "Averaging Quaternions" (2007), the same as
        # scipy's Rotation.mean. Expected rotation vectors from scipy:
        #   A, B = Rotation.from_rotvec([0.9, -0.3, 0.2]), Rotation.from_rotvec([-0.2, 1.1, 0.5])
        #   Rotation.concatenate([A, B]).mean(weights=w).as_rotvec()  for w = [1, 1] and w = [1, 3]
        # A normalised linear mean of the quaternions gives [0.084, 0.783, 0.446] for w = [1, 3].
        rotvecs = torch.tensor([[0.9, -0.3, 0.2], [-0.2, 1.1, 0.5]], dtype=torch.float64)
        data = Quaternion.from_axis_angle(rotvecs).data.to(device=device, dtype=dtype)
        uniform = torch.tensor([[0.3823365186803237, 0.41998759136458735, 0.37339616309716916]], dtype=torch.float64)
        weighted = torch.tensor([[0.02860900532725633, 0.8470541010728433, 0.4576793994589019]], dtype=torch.float64)
        # a half-precision fixture is off by one rounding of the quaternions, about 1e-2 rad in bfloat16
        tol = {torch.bfloat16: 2e-2, torch.float16: 3e-3}.get(dtype, 1e-5)
        # the weights are relative ([1, 3] and [2, 6] agree), and the sign of a member does not matter (-A is A)
        cases = [(None, uniform), ([1.0, 3.0], weighted), ([2.0, 6.0], weighted)]
        for members in (data, torch.stack((-data[0], data[1]))):
            for w, expected in cases:
                weights = None if w is None else torch.tensor(w, device=device, dtype=dtype)
                out = average_quaternions(Quaternion(members), w=weights)
                assert out.shape == (1, 4)
                # to_axis_angle is the principal log, so the arbitrary sign of the eigenvector does not matter
                self.assert_close(out.to_axis_angle(), expected.to(device=device, dtype=dtype), rtol=0.0, atol=tol)

    def test_wart_quaternion_polar_angle_gradient_nan_at_identity_4927(self, device, dtype):
        # #4927 https://github.com/kornia/kornia/issues/4927: polar_angle is acos(w / |q|), and acos has an infinite
        # derivative at 1, so at the identity -- the usual initialisation -- the gradient is nan in every component.
        # This test turns red when the gradient there becomes finite.
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype, requires_grad=True)
        Quaternion(identity).polar_angle.sum().backward()
        assert bool(torch.isnan(identity.grad).all()), identity.grad
        # control: away from the identity the same expression has a finite gradient
        q = self._unit([[0.9, 0.1, -0.3, 0.2]], device, dtype).requires_grad_(True)
        Quaternion(q).polar_angle.sum().backward()
        assert bool(torch.isfinite(q.grad).all()), q.grad

    def test_wart_average_quaternions_weights_by_member_norm_4974(self, device, dtype):
        # https://github.com/kornia/kornia/issues/4974: average_quaternions forms sum_i w_i q_i q_i^T from the stored
        # quaternions, so a member stored as 3 q counts 9 times, while every other rotation of Quaternion ignores a
        # positive scale; and a negative weight is accepted where scipy's Rotation.mean raises. This test turns red
        # when the members are normalised or negative weights are rejected.
        rotvecs = torch.tensor([[0.9, -0.3, 0.2], [-0.2, 1.1, 0.5]], dtype=torch.float64)
        data = Quaternion.from_axis_angle(rotvecs).data.to(device=device, dtype=dtype)
        unit = average_quaternions(Quaternion(data)).to_axis_angle()
        scaled = average_quaternions(Quaternion(torch.stack((3.0 * data[0], data[1])))).to_axis_angle()
        # the same rotations, one stored at norm 3: scipy's mean with weights [9, 1] is [0.839, -0.208, 0.224], the
        # unweighted mean [0.382, 0.420, 0.373]
        assert (scaled - unit).abs().max() > 0.3
        weights = torch.tensor([1.0, -0.5], device=device, dtype=dtype)
        out = average_quaternions(Quaternion(data), w=weights)
        assert bool(torch.isfinite(out.data).all())
