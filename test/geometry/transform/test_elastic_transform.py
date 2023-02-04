import pytest
import torch
from torch.autograd import gradcheck

from kornia.geometry.transform import elastic_transform2d
from kornia.testing import assert_close


class TestElasticTransform:
    def test_smoke(self, device, dtype):
        image = torch.rand(1, 4, 5, 5, device=device, dtype=dtype)
        noise = torch.rand(1, 2, 5, 5, device=device, dtype=dtype)
        assert elastic_transform2d(image, noise) is not None

    @pytest.mark.parametrize("batch, channels, height, width", [(1, 3, 3, 4), (2, 2, 2, 4), (1, 5, 4, 1)])
    def test_cardinality(self, device, dtype, batch, channels, height, width):
        shape = batch, channels, height, width
        img = torch.ones(shape, device=device, dtype=dtype)
        noise = torch.ones((batch, 2, height, width), device=device, dtype=dtype)
        assert elastic_transform2d(img, noise).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert elastic_transform2d([0.0])

        with pytest.raises(TypeError):
            assert elastic_transform2d(torch.tensor(), 1)

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, 1, device=device, dtype=dtype)
            noise = torch.ones(1, 2, 1, 1, device=device, dtype=dtype)
            assert elastic_transform2d(img, noise)

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, 1, 1, device=device, dtype=dtype)
            noise = torch.ones(1, 3, 1, 1, device=device, dtype=dtype)
            assert elastic_transform2d(img, noise)

    @pytest.mark.parametrize(
        "kernel_size, sigma, alpha",
        [
            [(3, 3), (4.0, 4.0), (32.0, 32.0)],
            [(5, 3), (4.0, 8.0), (16.0, 32.0)],
            [(5, 5), torch.tensor([2.0, 8.0]), torch.tensor([16.0, 64.0])],
        ],
    )
    def test_valid_paramters(self, device, dtype, kernel_size, sigma, alpha):
        image = torch.rand(1, 4, 5, 5, device=device, dtype=dtype)
        noise = torch.rand(1, 2, 5, 5, device=device, dtype=dtype)
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.to(device, dtype)
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.to(device, dtype)
        assert elastic_transform2d(image, noise, kernel_size, sigma, alpha) is not None

    def test_values(self, device, dtype):
        image = torch.tensor(
            [[[[0.0018, 0.7521, 0.7550], [0.2053, 0.4249, 0.1369], [0.1027, 0.3992, 0.8773]]]],
            device=device,
            dtype=dtype,
        )

        noise = torch.ones(1, 2, 3, 3, device=device, dtype=dtype)

        expected = torch.tensor(
            [[[[0.0005, 0.3795, 0.1905], [0.1034, 0.4235, 0.0702], [0.0259, 0.2007, 0.2193]]]],
            device=device,
            dtype=dtype,
        )

        actual = elastic_transform2d(image, noise)
        assert_close(actual, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("requires_grad", [True, False])
    def test_gradcheck(self, device, dtype, requires_grad):
        image = torch.rand(1, 1, 3, 3, device=device, dtype=torch.float64, requires_grad=requires_grad)
        noise = torch.rand(1, 2, 3, 3, device=device, dtype=torch.float64, requires_grad=not requires_grad)
        assert gradcheck(elastic_transform2d, (image, noise), raise_exception=True, fast_mode=True)
