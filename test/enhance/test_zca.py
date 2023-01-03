import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import BaseTester


class TestZCA(BaseTester):
    @pytest.mark.parametrize("unbiased", [True, False])
    def test_zca_unbiased(self, unbiased, device, dtype):
        data = torch.tensor([[0, 1], [1, 0], [-1, 0], [0, -1]], device=device, dtype=dtype)

        if unbiased:
            unbiased_val = 1.5
        else:
            unbiased_val = 2.0

        expected = torch.sqrt(unbiased_val * torch.abs(data)) * torch.sign(data)

        zca = kornia.enhance.ZCAWhitening(unbiased=unbiased).fit(data)

        actual = zca(data)

        self.assert_close(actual, expected, low_tolerance=True)

    @pytest.mark.parametrize("dim", [0, 1])
    def test_dim_args(self, dim, device, dtype):
        if 'xla' in device.type:
            pytest.skip("buggy with XLA devices.")

        if dtype == torch.float16:
            pytest.skip('not work for half-precision')

        data = torch.tensor([[0, 1], [1, 0], [-1, 0], [0, -1]], device=device, dtype=dtype)

        if dim == 1:
            expected = torch.tensor(
                [
                    [-0.35360718, 0.35360718],
                    [0.35351562, -0.35351562],
                    [-0.35353088, 0.35353088],
                    [0.35353088, -0.35353088],
                ],
                device=device,
                dtype=dtype,
            )
        elif dim == 0:
            expected = torch.tensor(
                [[0.0, 1.2247448], [1.2247448, 0.0], [-1.2247448, 0.0], [0.0, -1.2247448]], device=device, dtype=dtype
            )

        zca = kornia.enhance.ZCAWhitening(dim=dim)
        actual = zca(data, True)

        self.assert_close(actual, expected, low_tolerance=True)

    @pytest.mark.parametrize("input_shape,eps", [((15, 2, 2, 2), 1e-6), ((10, 4), 0.1), ((20, 3, 2, 2), 1e-3)])
    def test_identity(self, input_shape, eps, device, dtype):
        """Assert that data can be recovered by the inverse transform."""

        data = torch.randn(*input_shape, device=device, dtype=dtype)

        zca = kornia.enhance.ZCAWhitening(compute_inv=True, eps=eps).fit(data)

        data_w = zca(data)

        data_hat = zca.inverse_transform(data_w)

        self.assert_close(data, data_hat, low_tolerance=True)

    def test_grad_zca_individual_transforms(self, device, dtype):
        """Check if the gradients of the transforms are correct w.r.t to the input data."""

        data = torch.tensor([[2, 0], [0, 1], [-2, 0], [0, -1]], device=device, dtype=dtype)

        data = utils.tensor_to_gradcheck_var(data)

        def zca_T(x):
            return kornia.enhance.zca_mean(x)[0]

        def zca_mu(x):
            return kornia.enhance.zca_mean(x)[1]

        def zca_T_inv(x):
            return kornia.enhance.zca_mean(x, return_inverse=True)[2]

        assert gradcheck(zca_T, (data,), raise_exception=True)
        assert gradcheck(zca_mu, (data,), raise_exception=True)
        assert gradcheck(zca_T_inv, (data,), raise_exception=True)

    def test_grad_zca_with_fit(self, device, dtype):
        data = torch.tensor([[2, 0], [0, 1], [-2, 0], [0, -1]], device=device, dtype=dtype)

        data = utils.tensor_to_gradcheck_var(data)

        def zca_fit(x):
            zca = kornia.enhance.ZCAWhitening(detach_transforms=False)
            return zca(x, include_fit=True)

        assert gradcheck(zca_fit, (data,), raise_exception=True)

    def test_grad_detach_zca(self, device, dtype):
        data = torch.tensor([[1, 0], [0, 1], [-2, 0], [0, -1]], device=device, dtype=dtype)

        data = utils.tensor_to_gradcheck_var(data)
        zca = kornia.enhance.ZCAWhitening()

        zca.fit(data)

        assert gradcheck(zca, (data,), raise_exception=True)

    def test_not_fitted(self, device, dtype):
        with pytest.raises(RuntimeError):
            data = torch.rand(10, 2, device=device, dtype=dtype)

            zca = kornia.enhance.ZCAWhitening()
            zca(data)

    def test_not_fitted_inv(self, device, dtype):
        with pytest.raises(RuntimeError):
            data = torch.rand(10, 2, device=device, dtype=dtype)

            zca = kornia.enhance.ZCAWhitening()
            zca.inverse_transform(data)

    def test_jit(self, device, dtype):
        data = torch.rand(10, 3, 1, 2, device=device, dtype=dtype)
        zca = kornia.enhance.ZCAWhitening().fit(data)
        zca_jit = kornia.enhance.ZCAWhitening().fit(data)
        zca_jit = torch.jit.script(zca_jit)
        self.assert_close(zca_jit(data), zca(data))

    @pytest.mark.parametrize("unbiased", [True, False])
    def test_zca_whiten_func_unbiased(self, unbiased, device, dtype):
        data = torch.tensor([[0, 1], [1, 0], [-1, 0], [0, -1]], device=device, dtype=dtype)

        if unbiased:
            unbiased_val = 1.5
        else:
            unbiased_val = 2.0

        expected = torch.sqrt(unbiased_val * torch.abs(data)) * torch.sign(data)

        actual = kornia.enhance.zca_whiten(data, unbiased=unbiased)

        self.assert_close(actual, expected, low_tolerance=True)

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_gradcheck(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_smoke(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_module(self, device, dtype):
        pass
