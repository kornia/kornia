import pytest

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose

import kornia
from kornia.geometry.transform import elastic_transform2d


class TestElasticTransformSmoothing:
    # Test for deprecated 'smoothing' approach

    def test_smoke(self, device, dtype):
        image = torch.rand(1, 4, 5, 5, device=device, dtype=dtype)
        noise = torch.rand(1, 2, 5, 5, device=device, dtype=dtype)
        assert elastic_transform2d(image, noise, approach='smoothing') is not None

    @pytest.mark.parametrize(
        "batch, channels, height, width", [(1, 3, 3, 4), (2, 2, 2, 4), (1, 5, 4, 1)])
    def test_cardinality(self, device, dtype, batch, channels, height, width):
        shape = batch, channels, height, width
        img = torch.ones(shape, device=device, dtype=dtype)
        noise = torch.ones((batch, 2, height, width), device=device, dtype=dtype)
        assert elastic_transform2d(img, noise, approach='smoothing').shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert elastic_transform2d([0.], approach='smoothing')

        with pytest.raises(TypeError):
            assert elastic_transform2d(torch.tensor(), 1, approach='smoothing')

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, 1, device=device, dtype=dtype)
            noise = torch.ones(1, 2, 1, 1, device=device, dtype=dtype)
            assert elastic_transform2d(img, noise, approach='smoothing')

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, 1, 1, device=device, dtype=dtype)
            noise = torch.ones(1, 3, 1, 1, device=device, dtype=dtype)
            assert elastic_transform2d(img, noise, approach='smoothing')

    @pytest.mark.parametrize("kernel_size, sigma, alpha", [
        [(3, 3), (4., 4.), (32., 32.)],
        [(5, 3), (4., 8.), (16., 32.)],
        [(5, 5), torch.tensor([2., 8.]), torch.tensor([16., 64.])],
    ])
    def test_valid_paramters(self, device, dtype, kernel_size, sigma, alpha):
        image = torch.rand(1, 4, 5, 5, device=device, dtype=dtype)
        noise = torch.rand(1, 2, 5, 5, device=device, dtype=dtype)
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.to(device, dtype)
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.to(device, dtype)
        assert elastic_transform2d(image, noise, kernel_size, sigma, alpha, approach='smoothing') is not None

    def test_values(self, device, dtype):
        image = torch.tensor([[
            [[0.0018, 0.7521, 0.7550],
             [0.2053, 0.4249, 0.1369],
             [0.1027, 0.3992, 0.8773]]
        ]], device=device, dtype=dtype)

        noise = torch.ones(1, 2, 3, 3, device=device, dtype=dtype)

        expected = torch.tensor([[
            [[0.0005, 0.3795, 0.1905],
             [0.1034, 0.4235, 0.0702],
             [0.0259, 0.2007, 0.2193]]
        ]], device=device, dtype=dtype)

        actual = elastic_transform2d(image, noise, alpha=(1, 1), approach="smoothing")
        assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("requires_grad", [True, False])
    def test_gradcheck(self, device, dtype, requires_grad):
        image = torch.rand(1, 1, 3, 3, device=device, dtype=torch.float64, requires_grad=requires_grad)
        noise = torch.rand(1, 2, 3, 3, device=device, dtype=torch.float64, requires_grad=not requires_grad)
        assert gradcheck(elastic_transform2d,
                         (image, noise, (63, 63), (32., 32.), (16., 16.), False, 'bilinear', 'smoothing'),
                         raise_exception=True)

    def test_jit(self, device, dtype):
        image = torch.rand(1, 4, 5, 5, device=device, dtype=dtype)
        noise = torch.rand(1, 2, 5, 5, device=device, dtype=dtype)

        op = kornia.geometry.transform.elastic_transform2d
        op_jit = torch.jit.script(op)

        assert_allclose(op(image, noise, (63, 63), (32., 32.), (1., 1.), False, 'bilinear', 'smoothing'),
                        op_jit(image, noise, (63, 63), (32., 32.), (1., 1.), False, 'bilinear', 'smoothing'))


class TestElasticTransformCoarseNoise:
    # Test for deprecated 'coarse_noise' approach

    def test_smoke(self, device, dtype):
        image = torch.rand(1, 4, 2, 2, device=device, dtype=dtype)
        noise = torch.rand(1, 2, 8, 8, device=device, dtype=dtype)
        assert elastic_transform2d(image, noise, approach='coarse_noise') is not None

    @pytest.mark.parametrize(
        "batch, channels, height, width", [(1, 3, 3, 4), (2, 2, 2, 4), (1, 5, 4, 1)])
    def test_cardinality(self, device, dtype, batch, channels, height, width):
        shape = batch, channels, height, width
        img = torch.ones(shape, device=device, dtype=dtype)
        noise = torch.ones((batch, 2, 2, 2), device=device, dtype=dtype)
        assert elastic_transform2d(img, noise, approach='coarse_noise').shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert elastic_transform2d([0.], approach='coarse_noise')

        with pytest.raises(TypeError):
            assert elastic_transform2d(torch.tensor(), 1, approach='coarse_noise')

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, 1, device=device, dtype=dtype)
            noise = torch.ones(1, 2, 1, 1, device=device, dtype=dtype)
            assert elastic_transform2d(img, noise, approach='coarse_noise')

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, 1, 1, device=device, dtype=dtype)
            noise = torch.ones(1, 3, 1, 1, device=device, dtype=dtype)
            assert elastic_transform2d(img, noise, approach='coarse_noise')

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, 1, 1, device=device, dtype=dtype)
            noise = torch.ones(1, 3, 1, 1, device=device, dtype=dtype)
            assert elastic_transform2d(img, noise, approach='toeffi')

    @pytest.mark.parametrize("kernel_size, sigma, alpha", [
        [(3, 3), (4., 4.), (32., 32.)],
        [(5, 3), (4., 8.), (16., 32.)],
        [(5, 5), torch.tensor([2., 8.]), torch.tensor([16., 64.])],
    ])
    def test_valid_paramters(self, device, dtype, kernel_size, sigma, alpha):
        image = torch.rand(1, 4, 2, 2, device=device, dtype=dtype)
        noise = torch.rand(1, 2, 8, 8, device=device, dtype=dtype)
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.to(device, dtype)
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.to(device, dtype)
        assert elastic_transform2d(image, noise, kernel_size, sigma, alpha, approach='coarse_noise') is not None

    def test_values(self, device, dtype):
        image = torch.tensor([[
            [[0.0018, 0.7521, 0.7550, 0.1904],
             [0.2053, 0.4249, 0.1369, 0.0217],
             [0.1027, 0.3992, 0.8773, 0.6120],
             [0.3442, 0.8542, 0.1092, 0.6551]]
        ]], device=device, dtype=dtype)

        noise = torch.ones(1, 2, 2, 2, device=device, dtype=dtype)

        expected = torch.tensor([[
            [[0.0018, 0.7531, 0.3786, 0.0952],
             [0.1711, 0.4055, 0.2735, 0.1092],
             [0.2637, 0.5901, 0.5489, 0.3204],
             [0.1721, 0.3029, 0.2366, 0.1638]]
        ]], device=device, dtype=dtype)

        actual = elastic_transform2d(image, noise, alpha=(1., 1.), approach='coarse_noise')
        assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("requires_grad", [True, False])
    def test_gradcheck(self, device, dtype, requires_grad):
        image = torch.rand(1, 1, 2, 2, device=device, dtype=torch.float64, requires_grad=requires_grad)
        noise = torch.rand(1, 2, 8, 8, device=device, dtype=torch.float64, requires_grad=not requires_grad)
        assert gradcheck(elastic_transform2d,
                         (image, noise, (63, 63), (32., 32.), (16., 16.), False, 'bilinear', 'coarse_noise'),
                         raise_exception=True)

    def test_jit(self, device, dtype):
        image = torch.rand(1, 4, 2, 2, device=device, dtype=dtype)
        noise = torch.rand(1, 2, 8, 8, device=device, dtype=dtype)

        op = kornia.geometry.transform.elastic_transform2d
        op_jit = torch.jit.script(op)

        assert_allclose(op(image, noise, (0, 0), (0, 0), (1., 1.), False, 'bilinear', 'coarse_noise'),
                        op_jit(image, noise, (0, 0), (0, 0), (1., 1.), False, 'bilinear', 'coarse_noise'))
