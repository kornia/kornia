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

from kornia.geometry.vector import Scalar, Vector2, Vector3

from testing.base import BaseTester


class TestVector3(BaseTester):
    def test_smoke(self, device, dtype):
        vec = Vector3.random(device=device, dtype=dtype)
        assert vec.shape == (3,)
        assert vec.x is not None
        assert vec.y is not None
        assert vec.z is not None

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_getitem(self, device, dtype, batch_size):
        xyz = torch.rand((batch_size, 3), device=device, dtype=dtype)
        vec = Vector3(xyz)
        for i in range(batch_size):
            v = vec[i]
            self.assert_close(v.data, xyz[i, ...])

    @pytest.mark.parametrize("shape", [(), (1,), (2, 4)])
    def test_cardinality(self, device, dtype, shape):
        vec = Vector3.random(shape, device, dtype)
        assert vec.shape[:-1] == shape
        assert vec.x.shape == shape
        assert vec.y.shape == shape
        assert vec.z.shape == shape

    def test_from_coords(self):
        vec = Vector3.from_coords(0.0, 1.0, 0.0)
        assert vec.shape == (3,)
        assert vec.x == 0.0
        assert vec.y == 1.0
        assert vec.z == 0.0

    @pytest.mark.parametrize("shape", [(), (1,), (2, 4)])
    def test_from_coords_tensor(self, device, dtype, shape):
        xyz = torch.rand((*shape, 3), device=device, dtype=dtype)
        vec = Vector3.from_coords(xyz[..., 0], xyz[..., 1], xyz[..., 2])
        assert vec.shape[:-1] == shape
        assert vec.x.shape == shape
        assert vec.y.shape == shape
        assert vec.z.shape == shape

    @pytest.mark.parametrize("shape", [None, (1,), (2, 1)])
    def test_dot(self, device, dtype, shape):
        p0 = Vector3.random(shape, device, dtype)
        n0 = Vector3.random(shape, device, dtype).normalized()
        res: Scalar = p0.dot(n0)
        assert res.shape == () if shape is None else shape
        expected = torch.ones(shape or (), device=device, dtype=dtype)
        self.assert_close(n0.dot(n0), expected)

    @pytest.mark.parametrize("shape", [None, (1,), (2, 1)])
    def test_squared_norm(self, device, dtype, shape):
        p0 = Vector3.random(shape, device, dtype)
        res: Scalar = p0.squared_norm()
        assert res.shape == () if shape is None else shape

    def test_normalized(self, device, dtype):
        # A zero row normalizes to zero in every dtype (#5062: in float16 it used to be 0 / 0 = NaN, because
        # F.normalize's eps=1e-12 floor rounds to 0 there), next to an ordinary row.
        vec = Vector3(torch.tensor([[0.0, 3.0, -4.0], [0.0, 0.0, 0.0]], device=device, dtype=dtype))
        out = vec.normalized()
        assert isinstance(out, Vector3)
        assert out.data.dtype == dtype
        self.assert_close(out.data[0], torch.tensor([0.0, 0.6, -0.8], device=device, dtype=dtype))
        assert torch.equal(out.data[1], torch.zeros(3, device=device, dtype=dtype))

    def test_normalized_matches_functional_normalize(self, device, dtype):
        # Away from the zero vector the values are those of F.normalize(p=2, dim=-1), bit for bit, at every scale.
        # The row of norm 3e-6 (float16 subnormals) is a unit vector only while the norm floor stays below it.
        vec = Vector3(
            torch.tensor(
                [
                    [3.0, 0.0, 4.0],
                    [1e-3, -2e-3, 2e-3],
                    [100.0, 100.0, -100.0],
                    [0.5, 0.25, 0.125],
                    [1e-6, -2e-6, 2e-6],
                ],
                device=device,
                dtype=dtype,
            )
        )
        assert torch.equal(vec.normalized().data, torch.nn.functional.normalize(vec.data, p=2, dim=-1))

    def test_normalized_zero_vector_gradient_is_finite_5062(self, device, dtype):
        # The gradient at the zero vector is I / eps with eps = 1e-12, as with F.normalize, except in float16, where
        # I / eps overflows and the gradient is zero instead of inf.
        zero = torch.zeros(2, 3, device=device, dtype=dtype, requires_grad=True)
        Vector3(zero).normalized().data.sum().backward()
        assert torch.isfinite(zero.grad).all()
        expected = torch.zeros_like(zero) if dtype == torch.float16 else torch.full_like(zero, 1e12)
        self.assert_close(zero.grad, expected)

    @pytest.mark.skip(reason="not implemented yet")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_module(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_gradcheck(self, device):
        pass


class TestVector2(BaseTester):
    def test_smoke(self, device, dtype):
        vec = Vector2.random(device=device, dtype=dtype)
        assert vec.shape == (2,)
        assert vec.x is not None
        assert vec.y is not None

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_getitem(self, device, dtype, batch_size):
        xy = torch.rand((batch_size, 2), device=device, dtype=dtype)
        vec = Vector2(xy)
        for i in range(batch_size):
            v = vec[i]
            self.assert_close(v.data, xy[i, ...])

    @pytest.mark.parametrize("shape", [(), (1,), (2, 4)])
    def test_cardinality(self, device, dtype, shape):
        vec = Vector2.random(shape, device, dtype)
        assert vec.shape[:-1] == shape
        assert vec.x.shape == shape
        assert vec.y.shape == shape

    def test_from_coords(self):
        vec = Vector2.from_coords(0.0, 1.0)
        assert vec.shape == (2,)
        assert vec.x == 0.0
        assert vec.y == 1.0

    @pytest.mark.parametrize("shape", [(), (1,), (2, 4)])
    def test_from_coords_tensor(self, device, dtype, shape):
        xy = torch.rand((*shape, 2), device=device, dtype=dtype)
        vec = Vector2.from_coords(xy[..., 0], xy[..., 1])
        assert vec.shape[:-1] == shape
        assert vec.x.shape == shape
        assert vec.y.shape == shape

    @pytest.mark.parametrize("shape", [None, (1,), (2, 1)])
    def test_dot(self, device, dtype, shape):
        p0 = Vector2.random(shape, device, dtype)
        n0 = Vector2.random(shape, device, dtype).normalized()
        res: Scalar = p0.dot(n0)
        assert res.shape == () if shape is None else shape
        expected = torch.ones(shape or (), device=device, dtype=dtype)
        self.assert_close(n0.dot(n0), expected)

    @pytest.mark.parametrize("shape", [None, (1,), (2, 1)])
    def test_squared_norm(self, device, dtype, shape):
        p0 = Vector2.random(shape, device, dtype)
        res: Scalar = p0.squared_norm()
        assert res.shape == () if shape is None else shape

    def test_normalized(self, device, dtype):
        # A zero row normalizes to zero in every dtype (#5062: NaN in float16 before), next to an ordinary row.
        vec = Vector2(torch.tensor([[-3.0, 4.0], [0.0, 0.0]], device=device, dtype=dtype))
        out = vec.normalized()
        assert isinstance(out, Vector2)
        assert out.data.dtype == dtype
        self.assert_close(out.data[0], torch.tensor([-0.6, 0.8], device=device, dtype=dtype))
        assert torch.equal(out.data[1], torch.zeros(2, device=device, dtype=dtype))

    def test_normalized_matches_functional_normalize(self, device, dtype):
        # As for Vector3: the values of F.normalize(p=2, dim=-1) away from the zero vector, the 5e-6 row included.
        vec = Vector2(
            torch.tensor([[-3.0, 4.0], [1e-3, 2e-3], [100.0, -100.0], [3e-6, -4e-6]], device=device, dtype=dtype)
        )
        assert torch.equal(vec.normalized().data, torch.nn.functional.normalize(vec.data, p=2, dim=-1))

    def test_normalized_zero_vector_gradient_is_finite_5062(self, device, dtype):
        zero = torch.zeros(2, 2, device=device, dtype=dtype, requires_grad=True)
        Vector2(zero).normalized().data.sum().backward()
        assert torch.isfinite(zero.grad).all()
        expected = torch.zeros_like(zero) if dtype == torch.float16 else torch.full_like(zero, 1e12)
        self.assert_close(zero.grad, expected)

    @pytest.mark.skip(reason="not implemented yet")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_module(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_gradcheck(self, device):
        pass
