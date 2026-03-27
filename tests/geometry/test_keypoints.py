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

from kornia.geometry.keypoints import Keypoints, Keypoints3D, VideoKeypoints

from testing.base import BaseTester


class TestKeypoints(BaseTester):
    def test_smoke(self, device, dtype):
        data = torch.rand(10, 2, device=device, dtype=dtype)
        kp = Keypoints(data)
        assert isinstance(kp, Keypoints)

    def test_cardinality(self, device, dtype):
        data = torch.rand(10, 2, device=device, dtype=dtype)
        kp = Keypoints(data)
        assert kp.shape == (10, 2)

    def test_batched(self, device, dtype):
        data = torch.rand(3, 10, 2, device=device, dtype=dtype)
        kp = Keypoints(data)
        assert kp.shape == (3, 10, 2)
        assert kp._is_batched is True

    def test_unbatched(self, device, dtype):
        data = torch.rand(10, 2, device=device, dtype=dtype)
        kp = Keypoints(data)
        assert kp._is_batched is False

    def test_device_dtype(self, device, dtype):
        data = torch.rand(5, 2, device=device, dtype=dtype)
        kp = Keypoints(data)
        assert kp.device == device
        assert kp.dtype == dtype

    def test_from_tensor(self, device, dtype):
        data = torch.rand(5, 2, device=device, dtype=dtype)
        kp = Keypoints.from_tensor(data)
        assert kp.shape == data.shape

    def test_to_tensor(self, device, dtype):
        data = torch.rand(5, 2, device=device, dtype=dtype)
        kp = Keypoints(data)
        out = kp.to_tensor()
        assert out.shape == data.shape
        self.assert_close(out, data)

    def test_clone(self, device, dtype):
        data = torch.rand(5, 2, device=device, dtype=dtype)
        kp = Keypoints(data)
        kp2 = kp.clone()
        self.assert_close(kp.data, kp2.data)
        kp2._data[0, 0] = 999.0
        assert not torch.allclose(kp.data, kp2.data)

    def test_getitem(self, device, dtype):
        data = torch.rand(10, 2, device=device, dtype=dtype)
        kp = Keypoints(data)
        kp2 = kp[:5]
        assert kp2.shape == (5, 2)

    def test_setitem(self, device, dtype):
        data = torch.rand(10, 2, device=device, dtype=dtype)
        kp = Keypoints(data)
        new_data = torch.zeros(5, 2, device=device, dtype=dtype)
        new_kp = Keypoints(new_data)
        kp[:5] = new_kp
        self.assert_close(kp.data[:5], new_data)

    def test_transform_keypoints(self, device, dtype):
        # Use batched keypoints (B, N, 2) with batched M (B, 3, 3)
        data = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], device=device, dtype=dtype)  # (1, 2, 2)
        kp = Keypoints(data)
        M = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)  # (1, 3, 3)
        M[0, 0, 2] = 2.0  # translate x by 2
        M[0, 1, 2] = 3.0  # translate y by 3
        kp_t = kp.transform_keypoints(M)
        expected = torch.tensor([[[3.0, 3.0], [2.0, 4.0]]], device=device, dtype=dtype)
        self.assert_close(kp_t.data, expected)

    def test_transform_keypoints_inplace(self, device, dtype):
        data = torch.tensor([[[1.0, 0.0]]], device=device, dtype=dtype)  # (1, 1, 2)
        kp = Keypoints(data)
        M = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)  # (1, 3, 3)
        M[0, 0, 2] = 1.0
        kp.transform_keypoints_(M)
        expected = torch.tensor([[[2.0, 0.0]]], device=device, dtype=dtype)
        self.assert_close(kp.data, expected)

    def test_transform_keypoints_batched(self, device, dtype):
        data = torch.ones(2, 4, 2, device=device, dtype=dtype)
        kp = Keypoints(data)
        M = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(2, -1, -1).clone()
        M[:, 0, 2] = 5.0
        kp_t = kp.transform_keypoints(M)
        assert kp_t.shape == (2, 4, 2)
        self.assert_close(kp_t.data[..., 0], torch.full((2, 4), 6.0, device=device, dtype=dtype))

    def test_pad(self, device, dtype):
        data = torch.zeros(2, 4, 2, device=device, dtype=dtype)
        kp = Keypoints(data)
        padding = torch.tensor([[1.0, 0.0, 2.0, 0.0], [0.0, 0.0, 3.0, 0.0]], device=device, dtype=dtype)
        kp.pad(padding)
        # x += left_pad, y += top_pad
        self.assert_close(kp.data[0, :, 0], torch.full((4,), 1.0, device=device, dtype=dtype))
        self.assert_close(kp.data[1, :, 0], torch.zeros(4, device=device, dtype=dtype))
        self.assert_close(kp.data[0, :, 1], torch.full((4,), 2.0, device=device, dtype=dtype))

    def test_unpad(self, device, dtype):
        data = torch.ones(2, 4, 2, device=device, dtype=dtype) * 5.0
        kp = Keypoints(data)
        padding = torch.tensor([[1.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
        kp.unpad(padding)
        self.assert_close(kp.data[0, :, 0], torch.full((4,), 4.0, device=device, dtype=dtype))
        self.assert_close(kp.data[0, :, 1], torch.full((4,), 3.0, device=device, dtype=dtype))

    def test_index_put(self, device, dtype):
        data = torch.zeros(10, 2, device=device, dtype=dtype)
        kp = Keypoints(data)
        new_vals = torch.ones(3, 2, device=device, dtype=dtype)
        idx = (torch.tensor([0, 1, 2], device=device),)
        kp2 = kp.index_put(idx, new_vals)
        self.assert_close(kp2.data[:3], new_vals)

    def test_index_put_inplace(self, device, dtype):
        data = torch.zeros(10, 2, device=device, dtype=dtype)
        kp = Keypoints(data)
        new_vals = torch.ones(3, 2, device=device, dtype=dtype)
        idx = (torch.tensor([0, 1, 2], device=device),)
        kp.index_put(idx, new_vals, inplace=True)
        self.assert_close(kp.data[:3], new_vals)

    def test_type(self, device, dtype):
        if device.type == "mps":
            pytest.skip("MPS does not support float64")
        data = torch.rand(5, 2, device=device, dtype=torch.float32)
        kp = Keypoints(data)
        kp.type(torch.float64)
        assert kp.dtype == torch.float64

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            Keypoints("not a tensor")

        with pytest.raises(ValueError):
            Keypoints(torch.tensor([1, 2, 3], dtype=torch.int32))

        with pytest.raises(ValueError):
            Keypoints(torch.rand(3, 3, device=device, dtype=dtype))

        with pytest.raises(ValueError):
            Keypoints(torch.rand(3, 4, 2, 2, device=device, dtype=dtype))

    def test_transform_exception(self, device, dtype):
        kp = Keypoints(torch.rand(5, 2, device=device, dtype=dtype))
        with pytest.raises(ValueError):
            kp.transform_keypoints(torch.eye(4, device=device, dtype=dtype))

    def test_pad_exception(self, device, dtype):
        kp = Keypoints(torch.rand(2, 4, 2, device=device, dtype=dtype))
        with pytest.raises(RuntimeError):
            kp.pad(torch.zeros(2, 3, device=device, dtype=dtype))

    def test_int_input_raises_by_default(self, device, dtype):
        with pytest.raises(ValueError):
            Keypoints(torch.ones(5, 2, device=device, dtype=torch.int32))

    def test_int_input_converted_when_not_raising(self, device, dtype):
        data = torch.ones(5, 2, device=device, dtype=torch.int32)
        kp = Keypoints(data, raise_if_not_floating_point=False)
        assert kp.dtype == torch.float32

    def test_gradcheck(self, device):
        data = torch.rand(1, 5, 2, device=device, dtype=torch.float64, requires_grad=True)
        M = torch.eye(3, device=device, dtype=torch.float64).unsqueeze(0)
        M[0, 0, 2] = 1.0

        def fn(x):
            return Keypoints(x).transform_keypoints(M).data

        self.gradcheck(fn, (data,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        data = torch.rand(1, 5, 2, device=device, dtype=dtype)
        M = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)

        def fn(x):
            return Keypoints(x).transform_keypoints(M).data

        op = torch_optimizer(fn)
        self.assert_close(op(data), fn(data))

    def test_smoke_jit(self, device, dtype):
        pass  # Keypoints is not a nn.Module, jit test not applicable

    def test_module(self, device, dtype):
        pass  # Keypoints is not a nn.Module


class TestVideoKeypoints(BaseTester):
    def test_smoke(self, device, dtype):
        data = torch.rand(2, 5, 10, 2, device=device, dtype=dtype)
        vkp = VideoKeypoints.from_tensor(data)
        assert isinstance(vkp, VideoKeypoints)

    def test_cardinality(self, device, dtype):
        B, T, N = 2, 5, 10
        data = torch.rand(B, T, N, 2, device=device, dtype=dtype)
        vkp = VideoKeypoints.from_tensor(data)
        assert vkp.temporal_channel_size == T
        out = vkp.to_tensor()
        assert out.shape == (B, T, N, 2)

    def test_to_tensor_roundtrip(self, device, dtype):
        data = torch.rand(2, 4, 8, 2, device=device, dtype=dtype)
        vkp = VideoKeypoints.from_tensor(data)
        out = vkp.to_tensor()
        self.assert_close(out, data)

    def test_clone(self, device, dtype):
        data = torch.rand(2, 4, 8, 2, device=device, dtype=dtype)
        vkp = VideoKeypoints.from_tensor(data)
        vkp2 = vkp.clone()
        self.assert_close(vkp.to_tensor(), vkp2.to_tensor())
        assert vkp2.temporal_channel_size == vkp.temporal_channel_size

    def test_transform_keypoints(self, device, dtype):
        B, T, N = 1, 3, 5
        data = torch.ones(B, T, N, 2, device=device, dtype=dtype)
        vkp = VideoKeypoints.from_tensor(data)
        # After from_tensor, internal shape is (B*T, N, 2); need M with batch size B*T or 1
        M = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)  # (1, 3, 3) broadcasts
        out = vkp.transform_keypoints(M)
        assert isinstance(out, VideoKeypoints)
        assert out.temporal_channel_size == T

    def test_exception(self, device, dtype):
        with pytest.raises(ValueError):
            VideoKeypoints.from_tensor(torch.rand(5, 2, device=device, dtype=dtype))

        with pytest.raises(ValueError):
            VideoKeypoints.from_tensor(torch.rand(2, 5, 10, 3, device=device, dtype=dtype))

    def test_gradcheck(self, device):
        pass  # VideoKeypoints ops not differentiable through from_tensor reshape

    def test_dynamo(self, device, dtype, torch_optimizer):
        pass  # VideoKeypoints uses reshape; not straightforward to dynamo

    def test_smoke_jit(self, device, dtype):
        pass

    def test_module(self, device, dtype):
        pass

    def test_exception_in_base(self, device, dtype):
        pass


class TestKeypoints3D(BaseTester):
    def test_smoke(self, device, dtype):
        data = torch.rand(10, 3, device=device, dtype=dtype)
        kp = Keypoints3D(data)
        assert isinstance(kp, Keypoints3D)

    def test_cardinality(self, device, dtype):
        data = torch.rand(10, 3, device=device, dtype=dtype)
        kp = Keypoints3D(data)
        assert kp.shape == (10, 3)

    def test_batched(self, device, dtype):
        data = torch.rand(3, 10, 3, device=device, dtype=dtype)
        kp = Keypoints3D(data)
        assert kp.shape == (3, 10, 3)
        assert kp._is_batched is True

    def test_unbatched(self, device, dtype):
        data = torch.rand(10, 3, device=device, dtype=dtype)
        kp = Keypoints3D(data)
        assert kp._is_batched is False

    def test_from_tensor(self, device, dtype):
        data = torch.rand(5, 3, device=device, dtype=dtype)
        kp = Keypoints3D.from_tensor(data)
        assert kp.shape == data.shape

    def test_to_tensor(self, device, dtype):
        data = torch.rand(5, 3, device=device, dtype=dtype)
        kp = Keypoints3D(data)
        out = kp.to_tensor()
        self.assert_close(out, data)

    def test_clone(self, device, dtype):
        data = torch.rand(5, 3, device=device, dtype=dtype)
        kp = Keypoints3D(data)
        kp2 = kp.clone()
        self.assert_close(kp.data, kp2.data)
        kp2._data[0, 0] = 999.0
        assert not torch.allclose(kp.data, kp2.data)

    def test_getitem(self, device, dtype):
        data = torch.rand(10, 3, device=device, dtype=dtype)
        kp = Keypoints3D(data)
        kp2 = kp[:5]
        assert kp2.shape == (5, 3)

    def test_setitem(self, device, dtype):
        data = torch.rand(10, 3, device=device, dtype=dtype)
        kp = Keypoints3D(data)
        new_data = torch.zeros(5, 3, device=device, dtype=dtype)
        new_kp = Keypoints3D(new_data)
        kp[:5] = new_kp
        self.assert_close(kp.data[:5], new_data)

    def test_not_implemented(self, device, dtype):
        kp = Keypoints3D(torch.rand(5, 3, device=device, dtype=dtype))
        with pytest.raises(NotImplementedError):
            kp.pad(torch.zeros(1, 6, device=device, dtype=dtype))
        with pytest.raises(NotImplementedError):
            kp.unpad(torch.zeros(1, 6, device=device, dtype=dtype))
        with pytest.raises(NotImplementedError):
            kp.transform_keypoints(torch.eye(4, device=device, dtype=dtype))

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            Keypoints3D("not a tensor")

        with pytest.raises(ValueError):
            Keypoints3D(torch.tensor([1, 2, 3], dtype=torch.int32))

        with pytest.raises(ValueError):
            Keypoints3D(torch.rand(3, 2, device=device, dtype=dtype))

    def test_int_input_converted_when_not_raising(self, device, dtype):
        data = torch.ones(5, 3, device=device, dtype=torch.int32)
        kp = Keypoints3D(data, raise_if_not_floating_point=False)
        assert kp.data.dtype == torch.float32

    def test_gradcheck(self, device):
        pass  # Keypoints3D transform ops are NotImplemented

    def test_dynamo(self, device, dtype, torch_optimizer):
        pass

    def test_smoke_jit(self, device, dtype):
        pass

    def test_module(self, device, dtype):
        pass
