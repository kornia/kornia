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

import copy
import pickle

import pytest
import torch

from kornia.geometry.plane import Hyperplane
from kornia.geometry.vector import Scalar, Vector2, Vector3

from testing.base import BaseTester, dynamo_is_available


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


@pytest.mark.usefixtures("restore_torch_rng")
class TestConventionsVector(BaseTester):
    """Pins for the value conventions and known defects of :class:`Vector2` and :class:`Vector3`."""

    @pytest.mark.parametrize("vector_type, dim", [(Vector2, 2), (Vector3, 3)])
    def test_convention_vector_random_is_unit_box(self, vector_type, dim, device, dtype):
        # Vector.random draws every component uniformly in the unit box from torch's global generator. It is not a
        # random direction. torch.manual_seed reproduces a draw and another seed changes it.
        torch.manual_seed(0)
        vectors = vector_type.random((10000,), device=device, dtype=dtype)
        assert isinstance(vectors, vector_type)
        assert vectors.data.shape == (10000, dim)
        assert vectors.data.dtype == dtype
        assert vectors.data.device.type == device.type
        torch.manual_seed(0)
        assert torch.equal(vector_type.random((10000,), device=device, dtype=dtype).data, vectors.data)
        torch.manual_seed(1)
        assert not torch.equal(vector_type.random((10000,), device=device, dtype=dtype).data, vectors.data)
        values = vectors.data.cpu().double()
        assert float(values.min()) >= 0.0
        assert float(values.max()) <= 1.0
        norms = values.norm(dim=-1)
        assert float(norms.min()) < 0.5
        assert float(norms.max()) > 1.1
        self.assert_close(values.mean(0), torch.full((dim,), 0.5, dtype=torch.float64), rtol=0.0, atol=0.02)

    @pytest.mark.parametrize("vector_type, dim", [(Vector2, 2), (Vector3, 3)])
    def test_convention_vector_dot_returns_scalar_of_leading_shape(self, vector_type, dim, device, dtype):
        # dot and squared_norm reduce the last axis without keepdim and wrap the result as a Scalar of the leading
        # shape: two (2, dim) vectors give a (2,) Scalar; broadcastable (2, 1, dim) and (1, 4, dim) give (2, 4).
        a = vector_type(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device, dtype=dtype)[..., :dim])
        b = vector_type(torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], device=device, dtype=dtype)[..., :dim])

        dot = a.dot(b)
        assert isinstance(dot, Scalar)
        assert dot.data.shape == (2,)
        expected_dot = [8.0, 50.0] if dim == 2 else [20.0, 92.0]
        self.assert_close(dot.data, torch.tensor(expected_dot, device=device, dtype=dtype))

        broadcast_a = vector_type(
            torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], device=device, dtype=dtype)[..., :dim]
        )
        broadcast_b = vector_type(
            torch.tensor(
                [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]], device=device, dtype=dtype
            )[..., :dim]
        )
        broadcast_dot = broadcast_a.dot(broadcast_b)
        assert isinstance(broadcast_dot, Scalar)
        assert broadcast_dot.data.shape == (2, 4)
        expected_broadcast = (
            [[1.0, 2.0, 0.0, 3.0], [4.0, 5.0, 0.0, 9.0]]
            if dim == 2
            else [
                [1.0, 2.0, 3.0, 6.0],
                [4.0, 5.0, 6.0, 15.0],
            ]
        )
        self.assert_close(broadcast_dot.data, torch.tensor(expected_broadcast, device=device, dtype=dtype))

        squared_norm = a.squared_norm()
        assert isinstance(squared_norm, Scalar)
        assert squared_norm.data.shape == (2,)
        expected_squared_norm = [5.0, 41.0] if dim == 2 else [14.0, 77.0]
        self.assert_close(squared_norm.data, torch.tensor(expected_squared_norm, device=device, dtype=dtype))

    def test_convention_vector_deepcopy_keeps_the_class_5022(self, device, dtype):
        # copy.deepcopy, copy.copy and pickle all return the wrapper's own class, so a deep-copied Hyperplane keeps
        # a Vector3 normal and a Scalar offset and measures the same distances.
        wrapped = (
            Vector3(torch.tensor([[0.3, -1.2, 2.5]], device=device, dtype=dtype)),
            Vector2(torch.tensor([[0.3, -1.2]], device=device, dtype=dtype)),
            Scalar(torch.tensor([2.5], device=device, dtype=dtype)),
        )
        for obj in wrapped:
            assert type(copy.copy(obj)) is type(obj)
            deep = copy.deepcopy(obj)
            assert type(deep) is type(obj)
            self.assert_close(deep.data, obj.data, rtol=0, atol=0)
            assert deep.data.data_ptr() != obj.data.data_ptr()
            assert type(pickle.loads(pickle.dumps(obj))) is type(obj)  # noqa: S301

        normal = Vector3(torch.tensor([2.0, 1.0, -2.0], device=device, dtype=dtype) / 3.0)
        point = Vector3(torch.tensor([1.0, 2.0, 0.5], device=device, dtype=dtype))
        plane = Hyperplane.from_vector(normal, point)
        copied = copy.deepcopy(plane)
        assert type(copied.normal) is Vector3
        assert type(copied.offset) is Scalar
        self.assert_close(copied.normal.x, plane.normal.x, rtol=0, atol=0)
        query = Vector3(torch.tensor([[0.0, 0.0, 1.0], [2.0, -1.0, 0.5]], device=device, dtype=dtype))
        self.assert_close(copied.signed_distance(query).data, plane.signed_distance(query).data, rtol=0, atol=0)

    def test_convention_vector_operators_keep_the_class(self, device, dtype):
        # Reflected and in-place operators follow TensorWrapper's rule: the result is wrapped in the class of the
        # operand that handled it, and an in-place operator updates the wrapped tensor, so an alias sees it.
        data = torch.tensor([[0.5, 1.5, 2.0], [1.0, 3.0, 0.75]], device=device, dtype=dtype)
        v = Vector3(data.clone())
        reflected = 2 / v
        assert type(reflected) is Vector3
        self.assert_close(reflected.data, 2 / data, rtol=0, atol=0)
        alias = v
        v += 1
        assert v is alias
        assert type(v) is Vector3
        self.assert_close(alias.data, data + 1, rtol=0, atol=0)

    @pytest.mark.skipif(not dynamo_is_available(), reason="no Dynamo on this torch/python pair")
    def test_eager_backend_traces_vector_arithmetic(self, device, dtype):
        def fn(t):
            v = Vector3(t.clone())
            v += 1
            return (2 / v + v**2 + Vector3(t)).data

        torch._dynamo.reset()
        data = torch.tensor([[0.5, 1.5, 2.0], [1.0, 3.0, 0.75]], device=device, dtype=dtype)
        self.assert_close(torch.compile(fn, backend="eager", fullgraph=True)(data), fn(data))

    @pytest.mark.parametrize("vector_type, dim", [(Vector2, 2), (Vector3, 3)])
    def test_convention_vector_result_type_by_one_rule_5022(self, vector_type, dim, device, dtype):
        # A torch function, a tensor method, a tensor attribute and an operator give the same type: the vector class
        # while the result still holds vectors, else a plain tensor. Removing an axis returns a plain tensor even
        # when the result ends in dim by chance, as the norms of dim vectors do. The rule reads shapes only, so the
        # transpose of a square (dim, dim) batch would keep the class: four vectors keep v.T unambiguous here.
        data = torch.tensor(
            [[0.3, -1.2, 2.5], [1.0, 2.0, 3.0], [-0.5, 0.25, 1.5], [2.0, -1.0, 0.5]], device=device, dtype=dtype
        )[..., :dim]
        v = vector_type(data)
        for out in (v.clone(), torch.clone(v), v.mean(0, keepdim=True), torch.cat([v, v]), v + torch.ones_like(data)):
            assert type(out) is vector_type
        batched = v + torch.zeros(5, 1, dim, device=device, dtype=dtype)
        assert type(batched) is vector_type
        assert batched.data.shape == (5, 4, dim)

        norms = torch.linalg.norm(v, dim=-1)
        assert type(norms) is torch.Tensor
        self.assert_close(norms, torch.linalg.norm(data, dim=-1), rtol=0, atol=0)
        for out in (v.norm(dim=-1), v.T, v.sum(-1), v.reshape(-1), v.unbind(0)[0]):
            assert type(out) is torch.Tensor
        square = vector_type(torch.ones(dim, dim, device=device, dtype=dtype))
        for out in (torch.linalg.norm(square, dim=-1), square.sum(0), square.norm(dim=-1)):
            assert type(out) is torch.Tensor
            assert out.shape == (dim,)

        # A tensor attribute follows the same rule: the gradient of vectors is vectors.
        leaf = vector_type(data.clone().requires_grad_())
        (leaf.data**2).sum().backward()
        assert type(leaf.grad) is vector_type
        self.assert_close(leaf.grad.data, 2 * data)

        scalar = Scalar(torch.tensor([2.5], device=device, dtype=dtype))
        assert type(torch.clone(scalar)) is Scalar
        assert type(scalar.clone()) is Scalar

    @pytest.mark.skipif(not dynamo_is_available(), reason="no Dynamo on this torch/python pair")
    def test_eager_backend_traces_the_result_type_rule(self, device, dtype):
        # The rule runs inside a compiled function: a Scalar on the left gives a Vector3, a method keeps it, and a
        # norm drops it.
        def fn(t):
            v = Scalar(t[..., :1]) * Vector3(t)
            kept = v.clone()
            norms = kept.norm(dim=-1)
            return kept.data, norms, isinstance(kept, Vector3) and isinstance(norms, torch.Tensor)

        torch._dynamo.reset()
        data = torch.tensor([[0.5, 1.5, 2.0], [1.0, 3.0, 0.75], [2.0, -1.0, 0.25]], device=device, dtype=dtype)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)(data)
        expected = fn(data)
        assert compiled[2] is expected[2] is True
        self.assert_close(compiled[0], expected[0])
        self.assert_close(compiled[1], expected[1])

    def test_convention_vector3_scalar_on_either_side_5022(self, device, dtype):
        # An operation between a Scalar and a Vector3 is a Vector3 whichever side the Scalar is on.
        data = torch.tensor([[0.3, -1.2, 2.5], [1.0, 2.0, 3.0]], device=device, dtype=dtype)
        scale = torch.tensor([[2.0], [3.0]], device=device, dtype=dtype)
        v, s = Vector3(data), Scalar(scale)
        for out, expected in [
            (s * v, scale * data),
            (v * s, data * scale),
            (s / v, scale / data),
            (s + v, scale + data),
            (torch.mul(s, v), scale * data),
        ]:
            assert type(out) is Vector3
            self.assert_close(out.data, expected, rtol=0, atol=0)
        # A result that does not hold vectors falls back to the Scalar.
        assert type(torch.cat([v, s], -1)) is Scalar

    def test_wart_vector3_normalized_scales_below_eps_3952(self, device, dtype):
        # Wart pin (#3952): normalized() divides by max(norm, 1e-12), so a vector shorter than 1e-12 is scaled by 1e12
        # instead of normalized: (1e-13, 0, 0) comes back with length 0.1. A fix that raises or returns a unit
        # vector flips it. float16 cannot hold 1e-13 (it underflows to the zero vector).
        if dtype == torch.float16:
            pytest.skip("1e-13 underflows to 0 in float16")
        short = Vector3(torch.tensor([1e-13, 0.0, 0.0], device=device, dtype=dtype))
        self.assert_close(short.normalized().data, torch.tensor([0.1, 0.0, 0.0], device=device, dtype=dtype))
        self.assert_close(
            Vector2(short.data[:2]).normalized().data, torch.tensor([0.1, 0.0], device=device, dtype=dtype)
        )

    @pytest.mark.parametrize("vector_type, dim", [(Vector2, 2), (Vector3, 3)])
    def test_convention_vector_indexing_5022(self, vector_type, dim, device, dtype):
        # Indexing returns what the same key returns on the wrapped tensor: in the vector class while the key leaves
        # the coordinate axis last and whole, else as a plain tensor.
        data = torch.tensor([[0.3, -1.2, 2.5], [4.0, 0.5, -0.7]], device=device, dtype=dtype)[..., :dim]
        a = vector_type(data)
        mask = torch.tensor([True, False], device=device)
        index = torch.tensor([1, 0, 1], device=device)
        for key in (1, slice(1, None), mask, index, [1, 0], (Ellipsis,), (slice(None), None), None, (0, slice(None))):
            out = a[key]
            assert type(out) is vector_type, key
            self.assert_close(out.data, data[key], rtol=0, atol=0)
        for key in ((Ellipsis, 0), (slice(None), 0), (1, 1), (Ellipsis, None), (Ellipsis, slice(0, 1)), (0, index)):
            out = a[key]
            assert type(out) is torch.Tensor, key
            self.assert_close(out, data[key], rtol=0, atol=0)

        # A mask over both axes selects dim elements here, so only the key tells they are not one vector.
        full_mask = torch.zeros(2, dim, dtype=torch.bool, device=device)
        full_mask[0, :] = True
        assert type(a[full_mask]) is torch.Tensor
        self.assert_close(a[full_mask], data[full_mask], rtol=0, atol=0)

        single = vector_type(data[0])
        assert type(single[0]) is torch.Tensor
        assert single[0].shape == ()
        assert type(single[None]) is vector_type
