import pytest
import torch

from kornia.geometry import NamedPose
from kornia.geometry.liegroup import Se2, Se3, So2, So3
from kornia.testing import BaseTester


class TestNamedPose(BaseTester):
    def test_smoke(self, device, dtype):
        b_from_a = Se3.identity(device=device, dtype=dtype)
        pose = NamedPose(b_from_a, frame1="frame_a", frame2="frame_b")
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
            Se3.trans_x(torch.tensor([1.0], device=device, dtype=dtype)), frame1="frame_a", frame2="frame_b"
        )
        c_from_b = NamedPose(
            Se3.trans_y(torch.tensor([1.0], device=device, dtype=dtype)), frame1="frame_b", frame2="frame_c"
        )
        c_from_a = b_from_a * c_from_b
        assert isinstance(c_from_a, NamedPose)
        assert isinstance(c_from_a.pose, Se3)
        assert c_from_a.frame1 == "frame_a"
        assert c_from_a.frame2 == "frame_c"

    def test_fromRT(self, device, dtype):
        b_from_a_rotation = So3.random(device=device, dtype=dtype)
        b_from_a_translation = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        b_from_a = NamedPose.from_RT(b_from_a_rotation, b_from_a_translation, frame1="frame_a", frame2="frame_b")
        assert isinstance(b_from_a, NamedPose)
        assert isinstance(b_from_a.pose, Se3)

        b_from_a_rotation = So2.random(device=device, dtype=dtype)
        b_from_a_translation = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        b_from_a = NamedPose.from_RT(b_from_a_rotation, b_from_a_translation, frame1="frame_a", frame2="frame_b")
        assert isinstance(b_from_a, NamedPose)
        assert isinstance(b_from_a.pose, Se2)

        b_from_a_rotation = torch.eye(3, device=device, dtype=dtype)
        b_from_a_translation = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        b_from_a = NamedPose.from_RT(b_from_a_rotation, b_from_a_translation, frame1="frame_a", frame2="frame_b")
        assert isinstance(b_from_a, NamedPose)
        assert isinstance(b_from_a.pose, Se3)

        b_from_a_rotation = torch.eye(2, device=device, dtype=dtype)
        b_from_a_translation = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        b_from_a = NamedPose.from_RT(b_from_a_rotation, b_from_a_translation, frame1="frame_a", frame2="frame_b")
        assert isinstance(b_from_a, NamedPose)
        assert isinstance(b_from_a.pose, Se2)

    def test_inverse(self, device, dtype):
        b_from_a = NamedPose(
            Se3.trans_x(torch.tensor([1.0], device=device, dtype=dtype)), frame1="frame_a", frame2="frame_b"
        )
        a_from_b = b_from_a.inverse()
        assert isinstance(a_from_b, NamedPose)
        assert isinstance(a_from_b.pose, Se3)
        assert a_from_b.frame1 == "frame_b"
        assert a_from_b.frame2 == "frame_a"

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_transform(self, device, dtype, batch_size):
        if batch_size is None:
            points_in_a = torch.randn(3, device=device, dtype=dtype)
            b_from_a_se3 = Se3.trans_x(torch.tensor(1.0, device=device, dtype=dtype))
        else:
            points_in_a = torch.randn(batch_size, 3, device=device, dtype=dtype)
            b_from_a_se3 = Se3.trans_x(torch.tensor([1.0], device=device, dtype=dtype))
        b_from_a = NamedPose(b_from_a_se3, frame1="frame_a", frame2="frame_b")
        a_from_b = b_from_a.inverse()
        points_in_b = b_from_a.transform(points_in_a)
        assert points_in_b.shape == points_in_a.shape
        self.assert_close(a_from_b.transform(points_in_b), points_in_a)
