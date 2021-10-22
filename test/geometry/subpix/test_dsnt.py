import pytest
import torch

import kornia
from kornia.testing import assert_close


class TestRenderGaussian2d:
    @pytest.fixture
    def gaussian(self, device, dtype):
        return torch.tensor(
            [
                [0.002969, 0.013306, 0.021938, 0.013306, 0.002969],
                [0.013306, 0.059634, 0.098320, 0.059634, 0.013306],
                [0.021938, 0.098320, 0.162103, 0.098320, 0.021938],
                [0.013306, 0.059634, 0.098320, 0.059634, 0.013306],
                [0.002969, 0.013306, 0.021938, 0.013306, 0.002969],
            ],
            dtype=dtype,
            device=device,
        )

    def test_pixel_coordinates(self, gaussian, device, dtype):
        mean = torch.tensor([2.0, 2.0], dtype=dtype, device=device)
        std = torch.tensor([1.0, 1.0], dtype=dtype, device=device)
        actual = kornia.geometry.subpix.render_gaussian2d(mean, std, (5, 5), False)
        assert_close(actual, gaussian, rtol=0, atol=1e-4)

    def test_normalized_coordinates(self, gaussian, device, dtype):
        mean = torch.tensor([0.0, 0.0], dtype=dtype, device=device)
        std = torch.tensor([0.25, 0.25], dtype=dtype, device=device)
        actual = kornia.geometry.subpix.render_gaussian2d(mean, std, (5, 5), True)
        assert_close(actual, gaussian, rtol=0, atol=1e-4)

    def test_jit(self, device, dtype):
        mean = torch.tensor([0.0, 0.0], dtype=dtype, device=device)
        std = torch.tensor([0.25, 0.25], dtype=dtype, device=device)
        args = (mean, std, (5, 5), True)
        op = kornia.geometry.subpix.render_gaussian2d
        op_jit = torch.jit.script(op)
        assert_close(op(*args), op_jit(*args), rtol=0, atol=1e-5)

    @pytest.mark.skip(reason="it works but raises some warnings.")
    def test_jit_trace(self, device, dtype):
        def op(mean, std):
            return kornia.geometry.subpix.render_gaussian2d(mean, std, (5, 5), True)

        mean = torch.tensor([0.0, 0.0], dtype=dtype, device=device)
        std = torch.tensor([0.25, 0.25], dtype=dtype, device=device)
        args = (mean, std)
        op_jit = torch.jit.trace(op, args)
        assert_close(op(*args), op_jit(*args), rtol=0, atol=1e-5)


class TestSpatialSoftmax2d:
    @pytest.fixture(params=[torch.ones(1, 1, 5, 7), torch.randn(2, 3, 16, 16)])
    def input(self, request, device, dtype):
        return request.param.to(device, dtype)

    def test_forward(self, input):
        actual = kornia.geometry.subpix.spatial_softmax2d(input)
        assert actual.lt(0).sum().item() == 0, 'expected no negative values'
        sums = actual.sum(-1).sum(-1)
        assert_close(sums, torch.ones_like(sums))

    def test_jit(self, input):
        op = kornia.geometry.subpix.spatial_softmax2d
        op_jit = torch.jit.script(op)
        assert_close(op(input), op_jit(input), rtol=0, atol=1e-5)

    @pytest.mark.skip(reason="it works but raises some warnings.")
    def test_jit_trace(self, input):
        op = kornia.geometry.subpix.spatial_softmax2d
        op_jit = torch.jit.trace(op, (input,))
        assert_close(op(input), op_jit(input), rtol=0, atol=1e-5)


class TestSpatialExpectation2d:
    @pytest.fixture(
        params=[
            (
                torch.tensor([[[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]]]),
                torch.tensor([[[1.0, -1.0]]]),
                torch.tensor([[[2.0, 0.0]]]),
            )
        ]
    )
    def example(self, request, device, dtype):
        input, expected_norm, expected_px = request.param
        return input.to(device, dtype), expected_norm.to(device, dtype), expected_px.to(device, dtype)

    def test_forward(self, example):
        input, expected_norm, expected_px = example
        actual_norm = kornia.geometry.subpix.spatial_expectation2d(input, True)
        assert_close(actual_norm, expected_norm)
        actual_px = kornia.geometry.subpix.spatial_expectation2d(input, False)
        assert_close(actual_px, expected_px)

    def test_jit(self, example):
        input = example[0]
        op = kornia.geometry.subpix.spatial_expectation2d
        op_jit = torch.jit.script(op)
        assert_close(op(input), op_jit(input), rtol=0, atol=1e-5)

    @pytest.mark.skip(reason="it works but raises some warnings.")
    def test_jit_trace(self, example):
        input = example[0]
        op = kornia.geometry.subpix.spatial_expectation2d
        op_jit = torch.jit.trace(op, (input,))
        assert_close(op(input), op_jit(input), rtol=0, atol=1e-5)
