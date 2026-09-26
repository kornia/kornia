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

from kornia.geometry import NamedPose
from kornia.geometry.liegroup import Se2, Se3, So2, So3

from testing.base import BaseTester


class TestNamedPose(BaseTester):
    def test_smoke(self, device, dtype):
        b_from_a = Se3.identity(device=device, dtype=dtype)
        pose = NamedPose(b_from_a, frame_src="frame_a", frame_dst="frame_b")
        assert isinstance(pose, NamedPose)
        assert isinstance(pose.pose, Se3)

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

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

    def test_mul(self, device, dtype):
        b_from_a = NamedPose(
            Se3.trans_x(torch.tensor([1.0], device=device, dtype=dtype)), frame_src="frame_a", frame_dst="frame_b"
        )
        c_from_b = NamedPose(
            Se3.trans_y(torch.tensor([1.0], device=device, dtype=dtype)), frame_src="frame_b", frame_dst="frame_c"
        )
        c_from_a = c_from_b * b_from_a
        assert isinstance(c_from_a, NamedPose)
        assert isinstance(c_from_a.pose, Se3)
        assert c_from_a.frame_src == "frame_a"
        assert c_from_a.frame_dst == "frame_c"

    def test_from_rt(self, device, dtype):
        b_from_a_rotation = So3.random(device=device, dtype=dtype)
        b_from_a_translation = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        b_from_a = NamedPose.from_rt(b_from_a_rotation, b_from_a_translation, frame_src="frame_a", frame_dst="frame_b")
        assert isinstance(b_from_a, NamedPose)
        assert isinstance(b_from_a.pose, Se3)

        b_from_a_rotation = So2.random(device=device, dtype=dtype)
        b_from_a_translation = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        b_from_a = NamedPose.from_rt(b_from_a_rotation, b_from_a_translation, frame_src="frame_a", frame_dst="frame_b")
        assert isinstance(b_from_a, NamedPose)
        assert isinstance(b_from_a.pose, Se2)

        b_from_a_rotation = torch.eye(3, device=device, dtype=dtype)
        b_from_a_translation = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        b_from_a = NamedPose.from_rt(b_from_a_rotation, b_from_a_translation, frame_src="frame_a", frame_dst="frame_b")
        assert isinstance(b_from_a, NamedPose)
        assert isinstance(b_from_a.pose, Se3)

        b_from_a_rotation = torch.eye(2, device=device, dtype=dtype)
        b_from_a_translation = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        b_from_a = NamedPose.from_rt(b_from_a_rotation, b_from_a_translation, frame_src="frame_a", frame_dst="frame_b")
        assert isinstance(b_from_a, NamedPose)
        assert isinstance(b_from_a.pose, Se2)

    @pytest.mark.parametrize("batch_size", [None, 1, 2])
    def test_from_rt_tensor_batch_3d(self, device, dtype, batch_size):
        batch_shape = () if batch_size is None else (batch_size,)
        rotation = So3.random(batch_size, device=device, dtype=dtype).matrix()
        translation = torch.rand(*batch_shape, 3, device=device, dtype=dtype)
        b_from_a = NamedPose.from_rt(rotation, translation, frame_src="frame_a", frame_dst="frame_b")
        assert isinstance(b_from_a.pose, Se3)
        assert b_from_a.pose.matrix().shape == (*batch_shape, 4, 4)
        assert b_from_a.translation.shape == (*batch_shape, 3)

        matrix = torch.eye(4, device=device, dtype=dtype).repeat(*batch_shape, 1, 1)
        matrix[..., :3, :3] = rotation
        matrix[..., :3, 3] = translation
        self.assert_close(b_from_a.pose.matrix(), NamedPose.from_matrix(matrix).pose.matrix())
        self.assert_close(
            b_from_a.pose.matrix(), NamedPose.from_rt(So3.from_matrix(rotation), translation).pose.matrix()
        )

    @pytest.mark.parametrize("batch_size", [None, 1, 2])
    def test_from_rt_tensor_batch_2d(self, device, dtype, batch_size):
        if dtype == torch.bfloat16:
            pytest.skip("torch.complex has no bfloat16 overload, so So2 cannot be built at all")
        batch_shape = () if batch_size is None else (batch_size,)
        rotation = So2.random(batch_size, device=device, dtype=dtype).matrix()
        translation = torch.rand(*batch_shape, 2, device=device, dtype=dtype)
        b_from_a = NamedPose.from_rt(rotation, translation, frame_src="frame_a", frame_dst="frame_b")
        assert isinstance(b_from_a.pose, Se2)
        assert b_from_a.pose.matrix().shape == (*batch_shape, 3, 3)
        assert b_from_a.translation.shape == (*batch_shape, 2)

        matrix = torch.eye(3, device=device, dtype=dtype).repeat(*batch_shape, 1, 1)
        matrix[..., :2, :2] = rotation
        matrix[..., :2, 2] = translation
        self.assert_close(b_from_a.pose.matrix(), NamedPose.from_matrix(matrix).pose.matrix())
        self.assert_close(
            b_from_a.pose.matrix(), NamedPose.from_rt(So2.from_matrix(rotation), translation).pose.matrix()
        )

    @pytest.mark.parametrize(
        ("rotation_shape", "translation_shape"),
        [
            ((2, 3, 3), (1, 3)),
            ((2, 3, 3), (3,)),
            ((1, 3, 3), (3,)),
            ((3, 3), (1, 3)),
            ((2, 2, 2), (2,)),
            ((2, 2), (1, 2)),
        ],
    )
    def test_from_rt_tensor_batch_translation_mismatch(self, device, dtype, rotation_shape, translation_shape):
        # The translation must carry exactly the rotation's batch shape, as with So3/So2 rotations: neither a
        # broadcast translation nor a batched translation for an unbatched rotation is accepted.
        rotation = torch.eye(rotation_shape[-1], device=device, dtype=dtype).expand(*rotation_shape)
        translation = torch.zeros(*translation_shape, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="translation must have shape"):
            NamedPose.from_rt(rotation, translation)

    def test_from_matrix(self, device, dtype):
        b_from_a_matrix = Se3.identity(device=device, dtype=dtype).matrix()
        b_from_a = NamedPose.from_matrix(b_from_a_matrix, frame_src="frame_a", frame_dst="frame_b")
        point_in_a = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        point_in_b = b_from_a.transform_points(point_in_a)
        self.assert_close(point_in_b, point_in_a)
        assert isinstance(b_from_a, NamedPose)
        assert isinstance(b_from_a.pose, Se3)

        b_from_a_matrix = torch.eye(3, device=device, dtype=dtype)
        b_from_a = NamedPose.from_matrix(b_from_a_matrix, frame_src="frame_a", frame_dst="frame_b")
        point_in_a = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        point_in_b = b_from_a.transform_points(point_in_a)
        self.assert_close(point_in_b, point_in_a)
        assert isinstance(b_from_a, NamedPose)
        assert isinstance(b_from_a.pose, Se2)

    def test_inverse(self, device, dtype):
        b_from_a = NamedPose(
            Se3.trans_x(torch.tensor([1.0], device=device, dtype=dtype)), frame_src="frame_a", frame_dst="frame_b"
        )
        a_from_b = b_from_a.inverse()
        assert isinstance(a_from_b, NamedPose)
        assert isinstance(a_from_b.pose, Se3)
        assert a_from_b.frame_src == "frame_b"
        assert a_from_b.frame_dst == "frame_a"

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def transform_points(self, device, dtype, batch_size):
        if batch_size is None:
            points_in_a = torch.randn(3, device=device, dtype=dtype)
            b_from_a_se3 = Se3.trans_x(torch.tensor(1.0, device=device, dtype=dtype))
        else:
            points_in_a = torch.randn(batch_size, 3, device=device, dtype=dtype)
            b_from_a_se3 = Se3.trans_x(torch.tensor([1.0], device=device, dtype=dtype))
        b_from_a = NamedPose(b_from_a_se3, frame_src="frame_a", frame_dst="frame_b")
        a_from_b = b_from_a.inverse()
        points_in_b = b_from_a.transform_points(points_in_a)
        assert points_in_b.shape == points_in_a.shape
        self.assert_close(a_from_b.transform_points(points_in_b), points_in_a)
