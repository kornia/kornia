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

import pytest
import torch

from kornia.core.check import BaseError
from kornia.geometry.plane import Hyperplane
from kornia.geometry.vector import Scalar, Vector2, Vector3

from testing.base import BaseTester


class TestVector3(BaseTester):
    def test_smoke(self, device, dtype):
        vec = Vector3.random(device=device, dtype=dtype)
        assert vec.shape == (3,)
        assert vec.x is not None
        assert vec.y is not None
        assert vec.z is not None

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_getitem(self, device, dtype, batch_size):
        xyz = torch.rand((batch_size, 3), device=device, dtype=dtype)
        vec = Vector3(xyz)
        for i in range(batch_size):
            v = vec[i]
            self.assert_close(v.data, xyz[i, ...])

    @pytest.mark.parametrize("shape", ((), (1,), (2, 4)))
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

    @pytest.mark.parametrize("shape", ((), (1,), (2, 4)))
    def test_from_coords_tensor(self, device, dtype, shape):
        xyz = torch.rand((*shape, 3), device=device, dtype=dtype)
        vec = Vector3.from_coords(xyz[..., 0], xyz[..., 1], xyz[..., 2])
        assert vec.shape[:-1] == shape
        assert vec.x.shape == shape
        assert vec.y.shape == shape
        assert vec.z.shape == shape

    @pytest.mark.parametrize("shape", (None, (1,), (2, 1)))
    def test_dot(self, device, dtype, shape):
        p0 = Vector3.random(shape, device, dtype)
        n0 = Vector3.random(shape, device, dtype).normalized()
        res: Scalar = p0.dot(n0)
        assert res.shape == () if shape is None else shape
        expected = torch.ones(shape or (), device=device, dtype=dtype)
        self.assert_close(n0.dot(n0), expected)

    @pytest.mark.parametrize("shape", (None, (1,), (2, 1)))
    def test_squared_norm(self, device, dtype, shape):
        p0 = Vector3.random(shape, device, dtype)
        res: Scalar = p0.squared_norm()
        assert res.shape == () if shape is None else shape

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

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_getitem(self, device, dtype, batch_size):
        xy = torch.rand((batch_size, 2), device=device, dtype=dtype)
        vec = Vector2(xy)
        for i in range(batch_size):
            v = vec[i]
            self.assert_close(v.data, xy[i, ...])

    @pytest.mark.parametrize("shape", ((), (1,), (2, 4)))
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

    @pytest.mark.parametrize("shape", ((), (1,), (2, 4)))
    def test_from_coords_tensor(self, device, dtype, shape):
        xy = torch.rand((*shape, 2), device=device, dtype=dtype)
        vec = Vector2.from_coords(xy[..., 0], xy[..., 1])
        assert vec.shape[:-1] == shape
        assert vec.x.shape == shape
        assert vec.y.shape == shape

    @pytest.mark.parametrize("shape", (None, (1,), (2, 1)))
    def test_dot(self, device, dtype, shape):
        p0 = Vector2.random(shape, device, dtype)
        n0 = Vector2.random(shape, device, dtype).normalized()
        res: Scalar = p0.dot(n0)
        assert res.shape == () if shape is None else shape
        expected = torch.ones(shape or (), device=device, dtype=dtype)
        self.assert_close(n0.dot(n0), expected)

    @pytest.mark.parametrize("shape", (None, (1,), (2, 1)))
    def test_squared_norm(self, device, dtype, shape):
        p0 = Vector2.random(shape, device, dtype)
        res: Scalar = p0.squared_norm()
        assert res.shape == () if shape is None else shape

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

    def test_wart_vector3_call_path_type_split_5022(self, device, dtype):
        # Wart pin (#5022): deepcopy and tensor methods lose the wrapper, while torch functions rewrap their result.
        # A shape-changing torch function then fails Vector3 validation. Each part flips when its path is fixed.
        wrapped = (
            Vector3(torch.tensor([[0.3, -1.2, 2.5]], device=device, dtype=dtype)),
            Vector2(torch.tensor([[0.3, -1.2]], device=device, dtype=dtype)),
            Scalar(torch.tensor([2.5], device=device, dtype=dtype)),
        )
        for obj in wrapped:
            assert type(copy.copy(obj)) is type(obj)
            deep = copy.deepcopy(obj)
            assert type(deep) is torch.Tensor
            self.assert_close(deep, obj.data)
            assert type(obj.clone()) is torch.Tensor
            assert type(torch.clone(obj)) is type(obj)

        with pytest.raises(BaseError):
            torch.linalg.norm(wrapped[0], dim=-1)
        # A reduced result with three elements instead passes shape validation and is miswrapped as a Vector3.
        lucky_shape = Vector3(torch.ones(3, 3, device=device, dtype=dtype))
        reduced = torch.linalg.norm(lucky_shape, dim=-1)
        assert type(reduced) is Vector3
        assert reduced.data.shape == (3,)

        normal = Vector3(torch.tensor([2.0, 1.0, -2.0], device=device, dtype=dtype) / 3.0)
        point = Vector3(torch.tensor([1.0, 2.0, 0.5], device=device, dtype=dtype))
        plane = Hyperplane.from_vector(normal, point)
        assert isinstance(plane.normal, Vector3)
        assert isinstance(plane.offset, Scalar)
        copied = copy.deepcopy(plane)
        assert type(copied.normal) is torch.Tensor
        assert type(copied.offset) is torch.Tensor
        with pytest.raises(AttributeError):
            _ = copied.normal.x

    def test_wart_vector3_normalized_scales_below_eps_3952(self, device, dtype):
        # Wart pin (#3952): normalized() divides by max(norm, 1e-12), so a vector shorter than 1e-12 is scaled by 1e12
        # instead of normalized: (1e-13, 0, 0) comes back with length 0.1. A fix that raises or returns a unit
        # vector flips it. float16 cannot hold 1e-13 (it underflows to the zero vector, #5062).
        if dtype == torch.float16:
            pytest.skip("1e-13 underflows to 0 in float16")
        short = Vector3(torch.tensor([1e-13, 0.0, 0.0], device=device, dtype=dtype))
        self.assert_close(short.normalized().data, torch.tensor([0.1, 0.0, 0.0], device=device, dtype=dtype))
        self.assert_close(
            Vector2(short.data[:2]).normalized().data, torch.tensor([0.1, 0.0], device=device, dtype=dtype)
        )

    def test_wart_vector3_normalized_zero_is_nan_in_float16_5062(self, device, dtype):
        # Wart pin (#5062): the 1e-12 norm floor underflows to 0 in float16, so a zero vector normalizes to 0 / 0 = NaN
        # there and to zero in every other dtype. A floor at the dtype's smallest subnormal (the #4162 guard) flips
        # the float16 case to zero.
        for zero in (
            Vector3(torch.zeros(2, 3, device=device, dtype=dtype)),
            Vector2(torch.zeros(2, 2, device=device, dtype=dtype)),
        ):
            out = zero.normalized().data
            if dtype == torch.float16:
                assert bool(out.isnan().all())
            else:
                assert bool((out == 0).all())

    @pytest.mark.parametrize("vector_type, dim", [(Vector2, 2), (Vector3, 3)])
    def test_wart_vector_tuple_index_raises_5022(self, vector_type, dim, device, dtype):
        # Wart pin (#5022): Vector.__getitem__ indexes data[idx, ...], so a tuple key is turned into an index
        # tensor and raises RuntimeError (v[..., 0], v[:, 0]), and an int index of an unbatched vector leaves a 0-d
        # tensor that fails the last-dimension check (IndexError). A fix (index data[idx] and return a Tensor when
        # the result is not (..., dim)) flips all three; indexing the batch, as in a[1], already works.
        data = torch.tensor([[0.3, -1.2, 2.5], [4.0, 0.5, -0.7]], device=device, dtype=dtype)[..., :dim]
        a = vector_type(data)
        with pytest.raises(RuntimeError):
            _ = a[..., 0]
        with pytest.raises(RuntimeError):
            _ = a[:, 0]
        single = vector_type(data[0])
        with pytest.raises(IndexError):
            _ = single[0]

        row = a[1]
        assert isinstance(row, vector_type)
        self.assert_close(row.data, data[1])
