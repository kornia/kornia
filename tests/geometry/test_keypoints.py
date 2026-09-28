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

import kornia.augmentation as K
from kornia.geometry.keypoints import Keypoints, Keypoints3D, VideoKeypoints

from testing.base import BaseTester, supports_bilinear_2d_grid_sample


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

    def test_pad_unpad_unbatched_5021(self, device, dtype):
        # kornia#5021: an unbatched (N, 2) container indexes as (N,), so the (1, 1) padding column
        # broadcast the in-place update to (1, N) and raised "output with shape [2] doesn't match the
        # broadcast shape [1, 2]". The result must match the singleton-batch container.
        data = torch.tensor([[8.0, 2.0], [3.0, 5.0]], device=device, dtype=dtype)
        padding = torch.tensor([[3.0, 100.0, 7.0, 1000.0]], device=device, dtype=dtype)

        padded = Keypoints(data.clone()).pad(padding)
        self.assert_close(padded.data, torch.tensor([[11.0, 9.0], [6.0, 12.0]], device=device, dtype=dtype))
        self.assert_close(padded.data, Keypoints(data[None].clone()).pad(padding).data[0])

        unpadded = Keypoints(data.clone()).unpad(padding)
        self.assert_close(unpadded.data, torch.tensor([[5.0, -5.0], [0.0, -2.0]], device=device, dtype=dtype))
        self.assert_close(unpadded.data, Keypoints(data[None].clone()).unpad(padding).data[0])

        self.assert_close(Keypoints(data.clone()).pad(padding).unpad(padding).data, data)

    def test_pad_padding_on_cpu(self, device, dtype):
        # kornia#5021: like Boxes.pad, the padding is moved to the keypoints' device, so a CPU
        # padding_size works for keypoints on an accelerator.
        padding = torch.tensor([[1.0, 0.0, 2.0, 0.0]], dtype=dtype)
        for data in (torch.zeros(3, 2, device=device, dtype=dtype), torch.zeros(1, 3, 2, device=device, dtype=dtype)):
            kp = Keypoints(data.clone()).pad(padding)
            assert kp.device == device
            self.assert_close(kp.data[..., 0], torch.full(data.shape[:-1], 1.0, device=device, dtype=dtype))
            self.assert_close(kp.data[..., 1], torch.full(data.shape[:-1], 2.0, device=device, dtype=dtype))

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

        # an unbatched container carries a single image, so it takes exactly one padding row
        unbatched = Keypoints(torch.rand(4, 2, device=device, dtype=dtype))
        with pytest.raises(RuntimeError, match="one row"):
            unbatched.pad(torch.zeros(2, 4, device=device, dtype=dtype))
        with pytest.raises(RuntimeError, match="one row"):
            unbatched.unpad(torch.zeros(2, 4, device=device, dtype=dtype))
        # zero rows as well: a `> 1` check would let it through to an IndexError instead
        with pytest.raises(RuntimeError, match="one row"):
            unbatched.pad(torch.zeros(0, 4, device=device, dtype=dtype))

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


@pytest.mark.usefixtures("restore_torch_rng")
class TestConventionsKeypoints(BaseTester):
    """Pins for the coordinate, transform and aliasing conventions of :class:`Keypoints`."""

    @staticmethod
    def _has_cross_kernel(device, dtype):
        # The affine warp inverts its 3x3 matrix with kornia's closed-form inverse, built from torch.linalg.cross;
        # some torch builds have no cross kernel for a dtype (torch 2.5.1 on MPS raises for bfloat16).
        try:
            probe = torch.ones(1, 3, device=device, dtype=dtype)
            torch.linalg.cross(probe, probe, dim=-1)
        except RuntimeError:
            return False
        return True

    @staticmethod
    def _value_at(image, xy):
        # The image value at the pixel nearest to an (x, y) keypoint, or None when that pixel is outside the image.
        x, y = (int(v) for v in xy.round().tolist())
        if 0 <= x < image.shape[-1] and 0 <= y < image.shape[-2]:
            return float(image[0, 0, y, x])
        return None

    @pytest.mark.parametrize(
        "make_augmentation",
        [
            pytest.param(lambda: K.RandomAffine(degrees=(90.0, 90.0), p=1.0), id="rotate90"),
            pytest.param(lambda: K.RandomAffine(degrees=0.0, translate=(0.2, 0.0), p=1.0), id="translate_x"),
        ],
    )
    def test_convention_keypoints_are_xy_pixel_coordinates(self, make_augmentation, device, dtype):
        if not supports_bilinear_2d_grid_sample(device, dtype):
            pytest.skip(f"this torch build has no bilinear 2D grid_sample kernel for {dtype} on {device.type}")
        if not self._has_cross_kernel(device, dtype):
            pytest.skip(f"this torch build has no torch.linalg.cross kernel for {dtype} on {device.type}")
        # A keypoint is (x, y) in pixels: x indexes the columns (W) and y the rows (H). Image-content oracle: the
        # single bright pixel at row 2, column 8 of a 7 x 11 image is the keypoint (8, 2), and wherever the
        # augmentation moves that pixel, the transformed keypoint lands on it.
        height, width, row, col = 7, 11, 2, 8
        image = torch.zeros(1, 1, height, width, device=device, dtype=dtype)
        image[0, 0, row, col] = 1.0
        keypoints = torch.tensor([[[col, row]]], device=device, dtype=dtype)

        torch.manual_seed(0)
        augmentation = K.AugmentationSequential(make_augmentation(), data_keys=["input", "keypoints"])
        out_image, out_keypoints = augmentation(image, keypoints)
        landed = tuple(int(v) for v in out_keypoints[0, 0].round().tolist())
        brightest = int(out_image[0, 0].flatten().argmax())
        # rotate90 moves the pixel to (6, 6) and the seeded x-translation by +1.18 px to (9, 2)
        assert (brightest % width, brightest // width) == landed
        assert landed != (col, row)
        value = self._value_at(out_image, out_keypoints[0, 0])
        assert value is not None and value > 0.5

        # Relabel control: the same pixel read as (row, col) = (2, 8) leaves the image or lands on a dark pixel
        # ((0, 0) after rotate90, row 8 of a 7-row image after the translation).
        torch.manual_seed(0)
        augmentation = K.AugmentationSequential(make_augmentation(), data_keys=["input", "keypoints"])
        _, out_relabelled = augmentation(image, keypoints.flip(-1))
        value = self._value_at(out_image, out_relabelled[0, 0])
        assert value is None or value < 0.5

    def test_convention_keypoints_transform_is_column_vector_then_divided(self, device, dtype):
        # transform_keypoints maps a point p to M @ [x, y, 1]^T (a column vector, M on the left) and divides by the
        # third component. M != M^T and the projective row is non-zero, so for the first point the row-vector
        # reading [x, y, 1] @ M of the affine matrix ([0.2452, -0.0065]) and the undivided projective result
        # ([6.2, 3.2]) are far from the literals.
        points = torch.tensor([[8.0, 2.0], [3.0, 5.0]], device=device, dtype=dtype)
        affine = torch.tensor([[0.9, -0.3, 4.0], [0.2, 1.1, -1.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
        projective = torch.tensor([[1.0, 0.1, -2.0], [0.05, 0.9, 1.0], [0.01, 0.02, 1.0]], device=device, dtype=dtype)

        expected_affine = torch.tensor([[10.6, 2.8], [5.2, 5.1]], device=device, dtype=dtype)
        self.assert_close(Keypoints(points.clone()).transform_keypoints(affine).data, expected_affine)

        # M @ [x, y, 1]^T = (6.2, 3.2, 1.12) and (1.5, 5.65, 1.13)
        expected_projective = torch.tensor(
            [[6.2 / 1.12, 3.2 / 1.12], [1.5 / 1.13, 5.65 / 1.13]], device=device, dtype=dtype
        )
        self.assert_close(Keypoints(points.clone()).transform_keypoints(projective).data, expected_projective)
        batched = Keypoints(points[None].clone()).transform_keypoints(projective[None])
        self.assert_close(batched.data, expected_projective[None])

    def test_convention_keypoints_transform_inplace_rebinds_copy_does_not_alias(self, device, dtype):
        # inplace=False returns a new Keypoints on new storage, so writing into the result reaches neither the
        # original container nor the caller's tensor. inplace=True, and transform_keypoints_, return self after
        # rebinding its data to the transformed tensor, so the caller's tensor keeps the old coordinates.
        original = torch.tensor([[[8.0, 2.0], [3.0, 5.0]]], device=device, dtype=dtype)
        affine = torch.tensor([[0.9, -0.3, 4.0], [0.2, 1.1, -1.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
        expected = torch.tensor([[[10.6, 2.8], [5.2, 5.1]]], device=device, dtype=dtype)

        caller = original.clone()
        kp = Keypoints(caller)
        out = kp.transform_keypoints(affine)
        assert out is not kp
        self.assert_close(out.data, expected)
        out.data.fill_(999.0)
        assert kp.data is caller
        self.assert_close(caller, original)

        caller = original.clone()
        kp = Keypoints(caller)
        assert kp.transform_keypoints(affine, inplace=True) is kp
        self.assert_close(kp.data, expected)
        self.assert_close(caller, original)

        caller = original.clone()
        kp = Keypoints(caller)
        assert kp.transform_keypoints_(affine) is kp
        self.assert_close(kp.data, expected)
        self.assert_close(caller, original)

    @pytest.mark.parametrize("shape", [(0, 2), (1, 0, 2)])
    def test_convention_keypoints_empty_transform_keeps_storage_independent(self, shape, device, dtype):
        # transform_points returns empty inputs unchanged; Keypoints still gives every transform result new storage.
        caller = torch.empty(shape, device=device, dtype=dtype)
        transform = torch.eye(3, device=device, dtype=dtype)

        kp = Keypoints(caller)
        out = kp.transform_keypoints(transform)
        assert out is not kp
        assert out.data is not caller
        assert out.data.untyped_storage() is not caller.untyped_storage()
        assert kp.data is caller

        for transform_inplace in (
            lambda obj: obj.transform_keypoints(transform, inplace=True),
            lambda obj: obj.transform_keypoints_(transform),
        ):
            kp = Keypoints(caller)
            assert transform_inplace(kp) is kp
            assert kp.data is not caller
            assert kp.data.untyped_storage() is not caller.untyped_storage()

    @pytest.mark.parametrize("layout", ["batched", "unbatched", "strided"])
    def test_convention_keypoints_wrap_without_copy_and_edits_write_through(self, layout, device, dtype):
        # The constructor wraps the caller's tensor without copying it, a non-contiguous view included, and pad /
        # unpad shift that tensor in place and return self. padding_size rows are (left, right, top, bottom):
        # x += left and y += top. The four distinct padding values make a slot mix-up visible. clone() gives
        # independent storage. Item assignment and index_put(inplace=True) also write into the caller's tensor.
        original = torch.tensor([[8.0, 2.0], [3.0, 5.0]], device=device, dtype=dtype)
        padded = torch.tensor([[11.0, 9.0], [6.0, 12.0]], device=device, dtype=dtype)
        padding = torch.tensor([[3.0, 100.0, 7.0, 1000.0]], device=device, dtype=dtype)
        if layout == "batched":
            caller = original[None].clone()
        elif layout == "unbatched":
            caller = original.clone()
        else:
            caller = original.t().contiguous().t()
            assert not caller.is_contiguous()

        kp = Keypoints(caller)
        assert kp.pad(padding) is kp
        self.assert_close(caller.reshape(2, 2), padded)
        assert kp.unpad(padding) is kp
        self.assert_close(caller.reshape(2, 2), original)

        independent = kp.clone()
        independent.pad(padding)
        self.assert_close(independent.data.reshape(2, 2), padded)
        self.assert_close(caller.reshape(2, 2), original)

        second_point = (torch.tensor([1], device=device),)
        if layout == "batched":
            second_point = (torch.tensor([0], device=device), *second_point)
        kp.index_put(second_point, torch.tensor([100.0, 200.0], device=device, dtype=dtype), inplace=True)
        kp[..., :1, :] = Keypoints(torch.tensor([[70.0, 60.0]], device=device, dtype=dtype))
        edited = torch.tensor([[70.0, 60.0], [100.0, 200.0]], device=device, dtype=dtype)
        self.assert_close(caller.reshape(2, 2), edited)

    def test_convention_keypoints_single_padding_row_broadcasts_batch(self, device, dtype):
        caller = torch.tensor([[[8.0, 2.0], [3.0, 5.0]], [[1.0, 4.0], [6.0, 7.0]]], device=device, dtype=dtype)
        original = caller.clone()
        padding = torch.tensor([[3.0, 100.0, 7.0, 1000.0]], device=device, dtype=dtype)
        keypoints = Keypoints(caller)

        assert keypoints.pad(padding) is keypoints
        expected = torch.tensor([[[11.0, 9.0], [6.0, 12.0]], [[4.0, 11.0], [9.0, 14.0]]], device=device, dtype=dtype)
        self.assert_close(caller, expected)
        assert keypoints.unpad(padding) is keypoints
        self.assert_close(caller, original)

    @pytest.mark.parametrize(
        "path",
        [
            "keypoints_list_input",
            "keypoints_from_tensor_list",
            "keypoints_to_tensor_padded_sequence",
            "keypoints3d_list_input",
            "keypoints3d_from_tensor_list",
            "keypoints3d_to_tensor_padded_sequence",
            "keypoints3d_pad",
            "keypoints3d_unpad",
            "keypoints3d_transform_keypoints",
            "keypoints3d_transform_keypoints_",
        ],
    )
    def test_wart_keypoints_documented_paths_not_implemented_5023(self, path, device, dtype):
        # Wart pin (#5023): the docstrings advertise list input, to_tensor(as_padded_sequence=True) and the
        # Keypoints3D pad / unpad / transform_keypoints methods, and every one of these paths raises
        # NotImplementedError. Implementing a path, or removing it from the API, flips its case. The message is
        # not asserted: a change that only adds messages leaves every path unimplemented.
        kp2d = torch.tensor([[[8.0, 2.0], [3.0, 5.0]]], device=device, dtype=dtype)
        kp3d = torch.tensor([[[8.0, 2.0, 4.0], [3.0, 5.0, 1.0]]], device=device, dtype=dtype)
        calls = {
            "keypoints_list_input": lambda: Keypoints([kp2d[0], kp2d[0, :1]]),
            "keypoints_from_tensor_list": lambda: Keypoints.from_tensor([kp2d[0], kp2d[0, :1]]),
            "keypoints_to_tensor_padded_sequence": lambda: Keypoints(kp2d).to_tensor(as_padded_sequence=True),
            "keypoints3d_list_input": lambda: Keypoints3D([kp3d[0], kp3d[0, :1]]),
            "keypoints3d_from_tensor_list": lambda: Keypoints3D.from_tensor([kp3d[0], kp3d[0, :1]]),
            "keypoints3d_to_tensor_padded_sequence": lambda: Keypoints3D(kp3d).to_tensor(as_padded_sequence=True),
            "keypoints3d_pad": lambda: Keypoints3D(kp3d).pad(torch.zeros(1, 6, device=device, dtype=dtype)),
            "keypoints3d_unpad": lambda: Keypoints3D(kp3d).unpad(torch.zeros(1, 6, device=device, dtype=dtype)),
            "keypoints3d_transform_keypoints": lambda: Keypoints3D(kp3d).transform_keypoints(
                torch.eye(4, device=device, dtype=dtype)[None]
            ),
            "keypoints3d_transform_keypoints_": lambda: Keypoints3D(kp3d).transform_keypoints_(
                torch.eye(4, device=device, dtype=dtype)[None]
            ),
        }
        with pytest.raises(NotImplementedError):
            calls[path]()
