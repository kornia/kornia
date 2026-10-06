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

from kornia.geometry.conversions import angle_to_rotation_matrix
from kornia.geometry.liegroup import Se2, So2
from kornia.geometry.vector import Vector2

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, dynamo_is_available


class TestSo2(BaseTester):
    def _make_rand_data(self, device, dtype, input_shape):
        batch_size = input_shape[0]
        shape = input_shape[1:] if batch_size is None else input_shape
        return torch.rand(shape, device=device, dtype=dtype)

    @pytest.mark.parametrize("cdtype", [torch.cfloat, torch.cdouble])
    def test_smoke(self, device, cdtype):
        z = torch.randn(2, 1, dtype=cdtype, device=device)
        s = So2(z)
        assert isinstance(s, So2)
        assert s.z.shape == (2,)  # a (B, 1) z is read as (B,) (#4932)
        self.assert_close(s.z.data, z.data[:, 0])

    @pytest.mark.parametrize("input_shape", [(1,), (2,), (5,), ()])
    @pytest.mark.parametrize("cdtype", [torch.cfloat, torch.cdouble])
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

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    @pytest.mark.parametrize("cdtype", [torch.cfloat, torch.cdouble])
    def test_exception(self, batch_size, device, dtype, cdtype):
        z = torch.randn(batch_size, 2, dtype=cdtype, device=device)
        with pytest.raises(ValueError):
            assert So2(z)
        with pytest.raises(TypeError):
            assert So2.identity(1, device, dtype) * [1.0, 2.0, 1.0]
        theta = torch.rand((2, 2), dtype=dtype, device=device)
        with pytest.raises(ValueError):
            assert So2.exp(theta)
        theta = torch.rand((2, 2), dtype=dtype, device=device)
        with pytest.raises(ValueError):
            assert So2.hat(theta)
        m = torch.rand((2, 2, 1), dtype=dtype, device=device)
        with pytest.raises(ValueError):
            assert So2.from_matrix(m)
        m = torch.rand((2, 2, 1), dtype=dtype, device=device)
        with pytest.raises(ValueError):
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

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
    @pytest.mark.parametrize("cdtype", [torch.cfloat, torch.cdouble])
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
        self.assert_close(So2(z2).z, z2.flatten())  # (B, 1) is squeezed to (B,); (1,) for batch_size None stays
        self.assert_close(So2(z3).z, z3)

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
    @pytest.mark.parametrize("cdtype", [torch.cfloat, torch.cdouble])
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

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
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

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
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

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
    def test_exp(self, device, dtype, batch_size):
        theta = self._make_rand_data(device, dtype, (batch_size, 1))
        s = So2.exp(theta)
        self.assert_close(s.z.real, theta.flatten().cos())
        self.assert_close(s.z.imag, theta.flatten().sin())

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
    @pytest.mark.parametrize("cdtype", [torch.cfloat, torch.cdouble])
    def test_log(self, device, batch_size, cdtype):
        z = self._make_rand_data(device, cdtype, (batch_size,))
        t = So2(z).log()
        self.assert_close(t, z.imag.atan2(z.real))

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
    def test_exp_log(self, device, dtype, batch_size):
        theta = self._make_rand_data(device, dtype, (batch_size, 1))
        self.assert_close(So2.exp(theta).log(), theta.flatten())

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
    def test_convention_so2_hat_and_vee_are_the_generator_4929(self, device, dtype, batch_size):
        # https://github.com/kornia/kornia/issues/4929: hat used to be the symmetric [[0, t], [t, 0]] and vee read
        # its [0, 1] entry. Pin both layouts separately, because vee(hat(theta)) round-trips either way.
        theta = self._make_rand_data(device, dtype, (batch_size,))
        m = So2.hat(theta)
        o = torch.ones((2, 1), device=device, dtype=dtype)
        expected = torch.stack((-theta, theta), -1).reshape(-1, 2, 1)
        self.assert_close((m @ o).reshape(-1, 2, 1), expected)
        self.assert_close(m.transpose(-1, -2), -m)
        omega = self._make_rand_data(device, dtype, (batch_size, 2, 2))
        self.assert_close(So2.vee(omega), omega[..., 1, 0])
        self.assert_close(So2.vee(m), theta)

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
    def test_matrix_exp_of_hat_is_exp(self, device, dtype, batch_size):
        if dtype == torch.bfloat16:
            pytest.skip("torch has no complex bfloat16 dtype, which So2 stores its rotation in")
        theta = self._make_rand_data(device, dtype, (batch_size,))
        # The reference runs in float64 on the CPU: MPS has no float64 and CPU matrix_exp has no
        # half-precision kernel.
        expm = torch.linalg.matrix_exp(So2.hat(theta).cpu().double()).to(device=device, dtype=dtype)
        self.assert_close(expm, So2.exp(theta).matrix())

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
    def test_matrix(self, device, dtype, batch_size):
        theta = self._make_rand_data(device, dtype, (batch_size,))
        t = self._make_rand_data(device, dtype, (batch_size, 2))
        s = So2.exp(theta)
        p1 = s * t
        p2 = s.matrix() @ t[..., None]
        self.assert_close(p1, p2.squeeze(-1))

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
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

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
    @pytest.mark.parametrize("cdtype", [torch.cfloat, torch.cdouble])
    def test_inverse(self, device, batch_size, cdtype):
        z = self._make_rand_data(device, cdtype, (batch_size,))
        s = So2(z)
        s_in_in = s.inverse().inverse()
        self.assert_close(s_in_in.z.real, z.real)
        self.assert_close(s_in_in.z.imag, z.imag)

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
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
        assert theta.min() < -torch.pi + 0.05, (theta.min(), theta.max())
        assert theta.max() > torch.pi - 0.05, (theta.min(), theta.max())
        self.assert_close(So2.random(device=device, dtype=dtype).z.abs(), torch.tensor(1.0, device=device, dtype=dtype))

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
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

    def test_so2_column_angle_is_squeezed_4932(self, device, dtype):
        if dtype == torch.bfloat16:
            pytest.skip("torch has no complex bfloat16 dtype, which So2 stores its rotation in")
        theta = torch.tensor([0.3, 0.5, 0.7], device=device, dtype=dtype)
        p = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]], device=device, dtype=dtype)
        paired = So2.exp(theta) * p  # the (B,) layout rotates point i by angle i
        assert paired.shape == (3, 2)
        # https://github.com/kornia/kornia/issues/4932 (fixed): a (B, 1) z or angle used to keep its singleton axis, so
        # matrix() and hat() returned (B, 1, 2, 2), vee() rejected the latter and `*` broadcast z against the (B,)
        # point coordinates into (B, B, 2), every rotation applied to every point. It is read as (B,).
        column = theta[:, None]
        col = So2.exp(column)
        assert col.z.shape == (3,)
        self.assert_close(col.z, So2.exp(theta).z)
        assert So2(col.z[:, None]).z.shape == (3,)
        assert col.matrix().shape == (3, 2, 2)
        column_hat = So2.hat(column)
        assert column_hat.shape == (3, 2, 2)
        self.assert_close(So2.vee(column_hat), theta)
        out = col * p
        assert out.shape == (3, 2)
        self.assert_close(out, paired)
        moved = Se2(col, torch.zeros(3, 2, device=device, dtype=dtype)) * p  # used to raise on the (3, 1) z
        self.assert_close(moved, paired)
        # (1, 1) becomes (1,), while (1,) and () are unchanged
        assert So2.exp(theta[:1, None]).z.shape == (1,)
        assert So2.exp(theta[:1]).z.shape == (1,)
        assert So2.exp(theta[0]).z.shape == ()
        # the (B,) reading is a view, so the gradient reaches the (B, 1) angle
        column = column.clone().requires_grad_()
        (So2.exp(column) * p).sum().backward()
        assert column.grad.shape == (3, 1)
        assert torch.isfinite(column.grad).all()

    def test_so2_column_parameter_stays_the_module_parameter_4932(self, device, dtype):
        # Reading a (B, 1) z as (B,) must not take a caller's nn.Parameter away from the module: it stays its own
        # parameter under the same state_dict key and shape, so an optimizer over So2.parameters() trains it.
        if dtype == torch.bfloat16:
            pytest.skip("torch has no complex bfloat16 dtype, which So2 stores its rotation in")
        theta = torch.tensor([0.3, 0.5, 0.7], device=device, dtype=dtype)
        param = torch.nn.Parameter(torch.complex(theta.cos(), theta.sin())[:, None])
        s = So2(param)
        assert s.z.shape == (3,)
        assert [(name, t is param) for name, t in s.named_parameters()] == [("_z", True)]
        assert s.state_dict()["_z"].shape == (3, 1)
        assert s[1].z.shape == ()  # indexing reads the (B,) view, as for a (B,) z
        before = torch.view_as_real(param.detach()).clone()  # torch.equal has no complex float16 kernel
        s.log().sum().backward()
        torch.optim.SGD(s.parameters(), lr=0.1).step()
        assert not torch.equal(torch.view_as_real(param.detach()), before)

    @pytest.mark.parametrize("method", ["to", "bfloat16"])
    @pytest.mark.parametrize("as_parameter", [False, True])
    @pytest.mark.parametrize("cdtype", [torch.cfloat, torch.cdouble])
    def test_bfloat16_conversion_keeps_the_rotation_4923(self, device, method, as_parameter, cdtype):
        # No complex bfloat16 exists: keep both rotation components at their original precision while moving them.
        source_dtype = torch.float32 if cdtype == torch.cfloat else torch.float64
        angle = torch.tensor([0.3], dtype=source_dtype, requires_grad=not as_parameter)
        data = torch.complex(angle.cos(), angle.sin())
        rotation = So2(torch.nn.Parameter(data) if as_parameter else data)
        before = rotation.z.detach().clone()
        if as_parameter:
            (rotation.z.real.sum() + 2 * rotation.z.imag.sum()).backward()
        parent = torch.nn.ModuleDict({"rotation": rotation, "linear": torch.nn.Linear(2, 2)})
        if method == "to":
            parent.to(device=device, dtype=torch.bfloat16)
        else:
            parent.to(device=device).bfloat16()

        assert parent["linear"].weight.dtype == torch.bfloat16
        assert parent["linear"].weight.device == device
        assert rotation.z.dtype == before.dtype
        assert rotation.z.device == device
        assert isinstance(rotation._z, torch.nn.Parameter) == as_parameter
        state = dict(rotation.named_parameters()) if as_parameter else dict(rotation.named_buffers())
        assert state["_z"] is rotation._z
        self.assert_close(rotation.z, before.to(device))
        c, s = before.real.to(device), before.imag.to(device)
        expected_matrix = torch.stack((c, -s, s, c), -1).reshape(1, 2, 2)
        self.assert_close(rotation.matrix(), expected_matrix)
        point = torch.tensor([[1.0, 2.0]], device=device, dtype=torch.bfloat16)
        self.assert_close(rotation * point, torch.stack((c - 2 * s, s + 2 * c), -1))
        if as_parameter:
            assert rotation._z.grad.dtype == before.dtype
            assert rotation._z.grad.device == device
            self.assert_close(rotation._z.grad.real, torch.ones_like(c))
            self.assert_close(rotation._z.grad.imag, torch.full_like(s, 2.0))
        else:
            (rotation * point).sum().backward()
            self.assert_close(angle.grad, -3 * angle.detach().sin() - angle.detach().cos())


class _So2PointTransform(torch.nn.Module):
    def __init__(self, rotation: So2) -> None:
        super().__init__()
        self.rotation = rotation

    def forward(self, point: torch.Tensor) -> torch.Tensor:
        return (self.rotation.matrix() @ point[..., None]).squeeze(-1)


class TestSo2DtypeMigration(BaseTester):
    @pytest.fixture(autouse=True)
    def _require_complex_dtype(self, dtype):
        if dtype == torch.bfloat16:
            pytest.skip("PyTorch has no complex bfloat16 dtype; module bfloat16 conversion is tested separately")

    def _source_dtype(self, device, dtype):
        # Exercise a precision change in both directions, without constructing float64 on MPS.
        return torch.float64 if dtype == torch.float32 and device.type != "mps" else torch.float32

    def _cast(self, module, dtype, method):
        if method == "to":
            return module.to(dtype=dtype)
        name = {torch.float16: "half", torch.float32: "float", torch.float64: "double"}[dtype]
        return getattr(module, name)()

    @pytest.mark.parametrize("method", ["to", "convenience"])
    @pytest.mark.parametrize("as_parameter", [False, True])
    @pytest.mark.parametrize("shape", [(), (2,), (2, 1)])
    def test_real_dtype_conversion_4923(self, device, dtype, method, as_parameter, shape):
        source_dtype = self._source_dtype(device, dtype)
        theta = torch.tensor([0.3, -1.2] if shape else 0.3, device=device, dtype=source_dtype)
        theta = theta.reshape(shape).requires_grad_(not as_parameter)
        data = torch.complex(theta.cos(), theta.sin())
        rotation = So2(torch.nn.Parameter(data) if as_parameter else data)
        before = rotation.z.detach().clone()

        assert self._cast(rotation, dtype, method) is rotation
        assert rotation.z.is_complex() and rotation.z.real.dtype == dtype
        assert rotation.state_dict()["_z"].shape == shape
        self.assert_close(rotation.z.real, before.real.to(dtype))
        self.assert_close(rotation.z.imag, before.imag.to(dtype))
        c, s = before.real.to(dtype), before.imag.to(dtype)
        expected = torch.stack((c, -s, s, c), -1).reshape(*c.shape, 2, 2)
        self.assert_close(rotation.matrix(), expected)
        if as_parameter:
            assert dict(rotation.named_parameters())["_z"] is rotation._z
            assert not dict(rotation.named_buffers())
        else:
            assert dict(rotation.named_buffers())["_z"] is rotation._z
            assert not dict(rotation.named_parameters())

    @pytest.mark.parametrize("method", ["to", "convenience"])
    def test_parent_conversion_preserves_recursion(self, device, dtype, method):
        source_dtype = self._source_dtype(device, dtype)
        rotation = So2.exp(torch.tensor([0.3], device=device, dtype=source_dtype, requires_grad=True))
        rotation.extra = torch.nn.Module()
        rotation.extra.register_buffer("value", torch.ones(1, device=device, dtype=source_dtype))
        rotation.extra.weight = torch.nn.Parameter(torch.ones(1, device=device, dtype=source_dtype))
        rotation.extra.weight.sum().backward()
        parent = torch.nn.ModuleDict({"rotation": rotation})

        assert self._cast(parent, dtype, method) is parent
        assert rotation.z.is_complex() and rotation.z.real.dtype == dtype
        assert rotation.extra.value.dtype == dtype
        assert rotation.extra.weight.dtype == dtype and rotation.extra.weight.grad.dtype == dtype
        assert dict(parent.named_parameters())["rotation.extra.weight"] is rotation.extra.weight
        assert dict(parent.named_buffers())["rotation._z"] is rotation._z

    def test_parameter_gradient_and_optimizer(self, device, dtype):
        theta = torch.tensor([0.3, -1.2], device=device, dtype=self._source_dtype(device, dtype))
        parameter = torch.nn.Parameter(torch.complex(theta.cos(), theta.sin()))
        rotation = So2(parameter)
        assert rotation._z is parameter
        before = parameter.detach().clone()
        (rotation.z.real.sum() + 2 * rotation.z.imag.sum()).backward()
        rotation.to(dtype=dtype)

        assert isinstance(rotation._z, torch.nn.Parameter)
        assert dict(rotation.named_parameters())["_z"] is rotation._z
        assert rotation._z.grad.is_complex() and rotation._z.grad.real.dtype == dtype
        self.assert_close(rotation._z.grad.real, torch.ones(2, device=device, dtype=dtype))
        self.assert_close(rotation._z.grad.imag, torch.full((2,), 2.0, device=device, dtype=dtype))
        torch.optim.SGD(rotation.parameters(), lr=0.125).step()
        self.assert_close(rotation.z.real, before.real.to(dtype) - 0.125)
        self.assert_close(rotation.z.imag, before.imag.to(dtype) - 0.25)

    @pytest.mark.parametrize("method", ["to", "convenience"])
    def test_angle_gradient_matches_real_reference(self, device, dtype, method):
        angle = torch.tensor([0.3, -1.2], device=device, dtype=self._source_dtype(device, dtype), requires_grad=True)
        reference_angle = angle.detach().clone().requires_grad_()
        rotation = So2.exp(angle)
        self._cast(rotation, dtype, method)
        point = torch.tensor([[1.0, 2.0], [-3.0, 1.0]], device=device, dtype=dtype)
        weights = torch.tensor([[2.0, -1.0], [1.0, 3.0]], device=device, dtype=dtype)
        c, s = reference_angle.cos().to(dtype), reference_angle.sin().to(dtype)
        x, y = point.unbind(-1)
        expected = torch.stack((c * x - s * y, s * x + c * y), -1)
        actual = rotation * point
        self.assert_close(actual, expected)
        actual_grad = torch.autograd.grad((actual * weights).sum(), angle)[0]
        expected_grad = torch.autograd.grad((expected * weights).sum(), reference_angle)[0]
        self.assert_close(actual_grad, expected_grad)

    def test_conjugate_view_preserves_gradient(self, device, dtype):
        theta = torch.tensor([0.3], device=device, dtype=self._source_dtype(device, dtype))
        data = torch.complex(theta.cos(), theta.sin()).requires_grad_()
        rotation = So2(data.conj())
        assert rotation.z.is_conj()
        rotation.to(dtype=dtype)
        self.assert_close(rotation.z.real, data.real.to(dtype))
        self.assert_close(rotation.z.imag, -data.imag.to(dtype))
        (rotation * torch.tensor([[1.0, 2.0]], device=device, dtype=dtype)).sum().backward()
        self.assert_close(data.grad.real, torch.full_like(data.real, 3.0))
        self.assert_close(data.grad.imag, torch.ones_like(data.imag))

    @pytest.mark.parametrize("conjugate", [False, True])
    def test_noop_conversion_preserves_buffer(self, device, dtype, conjugate):
        theta = torch.tensor([0.3], device=device, dtype=dtype, requires_grad=True)
        data = torch.complex(theta.cos(), theta.sin())
        if conjugate:
            data = data.conj()
        rotation = So2(data)
        assert rotation.to(device=device)._z is data
        assert rotation.to(dtype=dtype)._z is data
        assert rotation.to(dtype=data.dtype)._z is data
        assert self._cast(rotation, dtype, "convenience")._z is data

    def test_explicit_complex_dtype(self, device, dtype):
        rotation = So2.exp(torch.tensor([0.3], device=device, dtype=self._source_dtype(device, dtype)))
        before = rotation.z.detach().clone()
        complex_dtype = torch.complex(torch.empty((), dtype=dtype), torch.empty((), dtype=dtype)).dtype
        rotation.to(dtype=complex_dtype)
        assert rotation.z.dtype == complex_dtype
        self.assert_close(rotation.z.real, before.real.to(dtype))
        self.assert_close(rotation.z.imag, before.imag.to(dtype))

    def test_checkpoint_compatibility(self, device, dtype):
        theta = torch.tensor([0.3, -1.2], device=device, dtype=self._source_dtype(device, dtype))
        legacy_state = {"_z": torch.complex(theta.cos(), theta.sin())}
        target = So2.exp((-theta).requires_grad_()).to(dtype=dtype)
        assert target._z.data_ptr() != legacy_state["_z"].data_ptr()
        result = target.load_state_dict(legacy_state)
        assert not result.missing_keys and not result.unexpected_keys
        assert list(target.state_dict()) == ["_z"]
        assert target.state_dict()["_z"].shape == legacy_state["_z"].shape
        assert target.z.is_complex() and target.z.real.dtype == dtype
        self.assert_close(target.z.real, legacy_state["_z"].real.to(dtype))
        self.assert_close(target.z.imag, legacy_state["_z"].imag.to(dtype))
        restored = So2.identity(2, device, dtype)
        restored.load_state_dict(target.state_dict())
        self.assert_close(restored.matrix(), target.matrix())
        point = torch.tensor([[1.0, 2.0], [-3.0, 1.0]], device=device, dtype=dtype)
        c, s = theta.cos().to(dtype), theta.sin().to(dtype)
        x, y = point.unbind(-1)
        self.assert_close(restored * point, torch.stack((c * x - s * y, s * x + c * y), -1))

    @pytest.mark.parametrize("as_parameter", [False, True])
    @pytest.mark.parametrize("change_dtype", [False, True])
    def test_parent_device_migration(self, device, dtype, as_parameter, change_dtype):
        source_dtype = self._source_dtype(device, dtype) if change_dtype else dtype
        angle = torch.tensor([0.3, -1.2], dtype=source_dtype, requires_grad=not as_parameter)
        data = torch.complex(angle.cos(), angle.sin())
        rotation = So2(torch.nn.Parameter(data) if as_parameter else data)
        parent = torch.nn.ModuleDict({"rotation": rotation})
        assert rotation.z.device.type == "cpu"
        if change_dtype:
            parent.to(device=device, dtype=dtype)
        else:
            parent.to(device=device)
        assert rotation.z.device == device
        assert rotation.z.is_complex() and rotation.z.real.dtype == dtype
        assert parent.state_dict()["rotation._z"].device == device
        assert isinstance(rotation._z, torch.nn.Parameter) == as_parameter
        c, s = angle.cos().to(device=device, dtype=dtype), angle.sin().to(device=device, dtype=dtype)
        point = torch.tensor([[1.0, 2.0], [-3.0, 1.0]], device=device, dtype=dtype)
        x, y = point.unbind(-1)
        self.assert_close(rotation * point, torch.stack((c * x - s * y, s * x + c * y), -1))
        restored = So2.identity(2, device, dtype)
        restored.load_state_dict(rotation.state_dict())
        self.assert_close(restored.matrix(), rotation.matrix())

    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    def test_dynamo(self, device, dtype, torch_optimizer):
        rotation = So2.exp(torch.tensor([0.3, -1.2], device=device, dtype=self._source_dtype(device, dtype)))
        model = _So2PointTransform(rotation).to(dtype=dtype)
        point = torch.tensor([[1.0, 2.0], [-3.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(torch_optimizer(model, fullgraph=True)(point), model(point))
