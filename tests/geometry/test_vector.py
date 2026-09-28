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
    """Pins for the value conventions and known defects of the :class:`Vector3` family."""

    def test_convention_vector3_random_is_unit_cube(self, device, dtype):
        # Vector3.random draws every component uniformly from [0, 1): the points fill the unit cube in the first
        # octant, so no component is negative and the norms spread over (0, sqrt(3)). It is not a random direction.
        torch.manual_seed(0)
        vectors = Vector3.random((10000,), device=device, dtype=dtype)
        assert isinstance(vectors, Vector3)
        assert vectors.data.shape == (10000, 3)
        assert vectors.data.dtype == dtype
        assert vectors.data.device.type == device.type
        values = vectors.data.cpu().double()
        assert float(values.min()) >= 0.0
        if device.type == "cpu" or dtype in (torch.float32, torch.float64):
            assert float(values.max()) < 1.0
        else:
            # torch's CPU generator draws a float16 / bfloat16 uniform in the target precision, which stays below 1.
            # The MPS generator can return exactly 1.0 there: on torch 2.14 it rounds a float32 draw to the dtype.
            assert float(values.max()) <= 1.0
        norms = values.norm(dim=-1)
        assert float(norms.min()) < 0.5
        assert float(norms.max()) > 1.5
        self.assert_close(values.mean(0), torch.full((3,), 0.5, dtype=torch.float64), rtol=0.0, atol=0.02)

    def test_convention_vector3_dot_returns_scalar_of_leading_shape(self, device, dtype):
        # dot and squared_norm reduce the last axis without keepdim and wrap the result as a Scalar of the leading
        # shape: two (2, 3) vectors give a (2,) Scalar.
        a = Vector3(torch.tensor([[0.3, -1.2, 2.5], [4.0, 0.5, -0.7]], device=device, dtype=dtype))
        b = Vector3(torch.tensor([[1.1, 0.2, -0.4], [-2.0, 3.0, 0.6]], device=device, dtype=dtype))

        dot = a.dot(b)
        assert isinstance(dot, Scalar)
        assert dot.data.shape == (2,)
        self.assert_close(dot.data, torch.tensor([-0.91, -6.92], device=device, dtype=dtype))

        squared_norm = a.squared_norm()
        assert isinstance(squared_norm, Scalar)
        assert squared_norm.data.shape == (2,)
        self.assert_close(squared_norm.data, torch.tensor([7.78, 16.74], device=device, dtype=dtype))

    def test_wart_vector3_deepcopy_returns_tensor_5022(self, device, dtype):
        # Wart pin (#5022): copy.deepcopy of a Vector3, Vector2 or Scalar returns a plain torch.Tensor (the instance
        # __getattr__ hands the lookup of __deepcopy__ to the wrapped tensor), while copy.copy keeps the class. A
        # module holding one degrades with it: a deep-copied Hyperplane stores Tensors, so .normal.x raises. A fix
        # (a __deepcopy__ on TensorWrapper) flips the `is torch.Tensor` assertions and the AttributeError below.
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

    def test_wart_vector3_tuple_index_raises_5022(self, device, dtype):
        # Wart pin (#5022): Vector3.__getitem__ indexes data[idx, ...], so a tuple key is turned into an index
        # tensor and raises RuntimeError (v[..., 0], v[:, 0]), and an int index of an unbatched vector leaves a 0-d
        # tensor that fails the last-dimension check (IndexError). A fix (index data[idx] and return a Tensor when
        # the result is not (..., 3)) flips all three; indexing the batch, as in a[1], already works.
        a = Vector3(torch.tensor([[0.3, -1.2, 2.5], [4.0, 0.5, -0.7]], device=device, dtype=dtype))
        with pytest.raises(RuntimeError):
            _ = a[..., 0]
        with pytest.raises(RuntimeError):
            _ = a[:, 0]
        single = Vector3(torch.tensor([0.3, -1.2, 2.5], device=device, dtype=dtype))
        with pytest.raises(IndexError):
            _ = single[0]

        row = a[1]
        assert isinstance(row, Vector3)
        self.assert_close(row.data, torch.tensor([4.0, 0.5, -0.7], device=device, dtype=dtype))
