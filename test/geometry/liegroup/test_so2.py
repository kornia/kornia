import pytest
import torch

from kornia.geometry.liegroup import So2
from kornia.testing import BaseTester


class TestSo2(BaseTester):
    def _make_rand_data(self, device, dtype, input_shape):
        batch_size = input_shape[0]
        shape = input_shape[1:] if batch_size is None else input_shape
        return torch.rand(shape, device=device, dtype=dtype)

    def test_smoke(self, device, dtype):
        z = torch.randn(2, 1, dtype=torch.cfloat, device=device)
        s = So2(z)
        assert isinstance(s, So2)
        self.assert_close(s.z.data, z.data)

    @pytest.mark.parametrize("input_shape", [(1, 1), (2, 1), (5, 1), (1,), (2,), (5,)])
    def test_cardinality(self, device, dtype, input_shape):
        z = torch.randn(input_shape, dtype=torch.cfloat, device=device)
        s = So2(z)
        theta = torch.rand(input_shape, dtype=dtype, device=device)
        assert s.z.shape == input_shape
        assert (s * s).z.shape == input_shape
        assert s.exp(theta).z.shape == input_shape
        assert s.log().shape == input_shape
        expected_hat_shape = (2, 2) if input_shape[0] == 1 else (input_shape[0], 2, 2)
        assert s.hat(theta).shape == expected_hat_shape
        assert s.inverse().z.shape == input_shape

    @pytest.mark.parametrize("input_shape", [(1, 2, 2), (2, 2, 2), (5, 2, 2), (2, 2)])
    def test_matrix_cardinality(self, device, dtype, input_shape):
        matrix = torch.rand(input_shape, dtype=dtype, device=device)
        s = So2.from_matrix(matrix)
        assert s.matrix().shape == input_shape

    @pytest.mark.parametrize("batch_size", (1, 2, 5))
    def test_exception(self, batch_size, device, dtype):
        # with pytest.raises(ValueError):
        #     z = torch.randn(batch_size, 2, dtype=torch.cfloat, device=device)
        #     assert So2(z)
        # with pytest.raises(TypeError):
        #     assert So2.identity(1, device, dtype) * [1.0, 2.0, 1.0]
        # with pytest.raises(ValueError):
        #     theta = torch.rand((2, 2), dtype=dtype, device=device)
        #     assert So2.exp(theta)
        # with pytest.raises(ValueError):
        #     theta = torch.rand((2, 2), dtype=dtype, device=device)
        #     assert So2.hat(theta)
        # with pytest.raises(ValueError):
        #     m = torch.rand((2, 2, 1), dtype=dtype, device=device)
        #     assert So2.from_matrix(m)
        # with pytest.raises(Exception):
        #     assert So2.identity(batch_size=0)
        pass
    # TODO: implement me
    def test_gradcheck(self, device):
        pass

    # TODO: implement me
    def test_jit(self, device, dtype):
        pass

    # TODO: implement me
    def test_module(self, device, dtype):
        pass

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_init(self, device, dtype, batch_size):
        z = self._make_rand_data(device, dtype, [batch_size, 2])
        s1 = So2(torch.complex(z[..., 0, None], z[..., 1, None]))
        s2 = So2(s1.z)
        assert isinstance(s2, So2)
        self.assert_close(s1.z.data, s2.z.data)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_getitem(self, device, dtype, batch_size):
        data = self._make_rand_data(device, dtype, [batch_size, 2])
        z = torch.complex(data[..., 0, None], data[..., 1, None])
        s = So2(z)
        n = 1 if batch_size is None else batch_size
        for i in range(n):
            s1 = s[i]
            self.assert_close(s1.z.data[0].real.squeeze(), z.data[i].real.squeeze())
            self.assert_close(s1.z.data[0].imag.squeeze(), z.data[i].imag.squeeze())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_mul(self, device, dtype, batch_size):
        s1 = So2.identity(batch_size, device, dtype)
        z = self._make_rand_data(device, dtype, [batch_size, 2])
        s2 = So2(torch.complex(z[..., 0, None], z[..., 1, None]))
        t = self._make_rand_data(device, dtype, [batch_size, 2])
        s1_pose_s2 = s1 * s2
        s2_pose_s2 = s2 * s2.inverse()
        self.assert_close(s1_pose_s2.z.real, s2.z.real)
        self.assert_close(s1_pose_s2.z.imag, s2.z.imag)
        self.assert_close(s2_pose_s2.z.real, s1.z.real)
        self.assert_close(s2_pose_s2.z.imag, s1.z.imag)
        # import pdb
        # pdb.set_trace()
        # self.assert_close((s1 * t), t.reshape(1, 2, 1)) TODO

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_exp(self, device, dtype, batch_size):
        theta = self._make_rand_data(device, dtype, [batch_size, 1])
        s = So2.exp(theta)
        self.assert_close(s.z.real, theta.cos())
        self.assert_close(s.z.imag, theta.sin())

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_log(self, device, dtype, batch_size):
        data = self._make_rand_data(device, dtype, [batch_size, 2])
        z = torch.complex(data[..., 0, None], data[..., 1, None])
        t = So2(z).log()
        self.assert_close(t, data[..., 1, None].atan2(data[..., 0, None]))

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_exp_log(self, device, dtype, batch_size):
        theta = self._make_rand_data(device, dtype, [batch_size, 1])
        self.assert_close(So2.exp(theta).log(), theta)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_hat(self, device, dtype, batch_size):
        theta = self._make_rand_data(device, dtype, [batch_size, 1])
        m = So2.hat(theta)
        o = torch.ones((1, 2, 1), device=device, dtype=dtype)
        self.assert_close(m @ o, torch.cat((theta[..., None], theta[..., None]), 1).reshape(-1, 2, 1))

    # @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    # def test_matrix(self, device, dtype, batch_size):
    #     theta = self._make_rand_data(device, dtype, [batch_size, 1])
    #     t = self._make_rand_data(device, dtype, [batch_size, 2, 1])
    #     ss = So2.exp(theta)
    #     p1 = ss * t
    #     p2 = ss.matrix() @ t
    #     self.assert_close(p1, p2)

    @pytest.mark.parametrize("batch_size", (None, 1, 2, 5))
    def test_inverse(self, device, dtype, batch_size):
        data = self._make_rand_data(device, dtype, [batch_size, 2])
        z = torch.complex(data[..., 0, None], data[..., 1, None])
        s = So2(z)
        self.assert_close(s.inverse().inverse().z.real, z.real)
        self.assert_close(s.inverse().inverse().z.imag, z.imag)
