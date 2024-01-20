import pytest
import torch

from kornia.geometry import NamedPose
from kornia.geometry.liegroup import Se2, Se3, So2, So3
from kornia.testing import BaseTester


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
