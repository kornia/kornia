import pytest
import torch
from torch.autograd import gradcheck

from kornia import enhance
from kornia.testing import assert_close, BaseTester, tensor_to_gradcheck_var


class TestEqualization(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 1, 10, 20
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        res = enhance.equalize_clahe(img)
        assert isinstance(res, torch.Tensor)
        assert res.shape == img.shape
        assert res.device == img.device
        assert res.dtype == img.dtype

    @pytest.mark.parametrize("B, C", [(None, 1), (None, 3), (1, 1), (1, 3), (4, 1), (4, 3)])
    def test_cardinality(self, B, C, device, dtype):
        H, W = 10, 20
        if B is None:
            img = torch.rand(C, H, W, device=device, dtype=dtype)
        else:
            img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        res = enhance.equalize_clahe(img)
        assert res.shape == img.shape

    @pytest.mark.parametrize("clip, grid", [(0.0, None), (None, (2, 2)), (2.0, (2, 2))])
    def test_optional_params(self, clip, grid, device, dtype):
        C, H, W = 1, 10, 20
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        if clip is None:
            res = enhance.equalize_clahe(img, grid_size=grid)
        elif grid is None:
            res = enhance.equalize_clahe(img, clip_limit=clip)
        else:
            res = enhance.equalize_clahe(img, clip, grid)
        assert isinstance(res, torch.Tensor)
        assert res.shape == img.shape

    @pytest.mark.parametrize(
        "B, clip, grid, exception_type",
        [
            (0, 1.0, (2, 2), ValueError),
            (1, 1, (2, 2), TypeError),
            (1, 2.0, 2, TypeError),
            (1, 2.0, (2, 2, 2), TypeError),
            (1, 2.0, (2, 2.0), TypeError),
            (1, 2.0, (2, 0), ValueError),
        ],
    )
    def test_exception(self, B, clip, grid, exception_type):
        C, H, W = 1, 10, 20
        img = torch.rand(B, C, H, W)
        with pytest.raises(exception_type):
            enhance.equalize_clahe(img, clip, grid)

    @pytest.mark.parametrize("dims", [(1, 1, 1, 1, 1), (1, 1)])
    def test_exception_tensor_dims(self, dims):
        img = torch.rand(dims)
        with pytest.raises(ValueError):
            enhance.equalize_clahe(img)

    def test_exception_tensor_type(self):
        with pytest.raises(TypeError):
            enhance.equalize_clahe([1, 2, 3])

    @pytest.mark.xfail(
        raises=RuntimeError,
        reason="Sometimes generates the error: 'Numerical gradient for function expected to be zero'",
    )
    def test_gradcheck(self, device, dtype):
        bs, channels, height, width = 1, 1, 6, 6
        inputs = torch.rand(bs, channels, height, width, device=device, dtype=dtype)
        inputs = tensor_to_gradcheck_var(inputs)
        assert gradcheck(enhance.equalize_clahe, (inputs, 40.0, (2, 2)), raise_exception=True)

    @pytest.mark.skip(reason="args and kwargs in decorator")
    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 10, 20
        inp = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        op = enhance.equalize_clahe
        op_script = torch.jit.script(op)
        assert_close(op(inp), op_script(inp))

    def test_module(self):
        # equalize_clahe is only a function
        pass

    @pytest.fixture()
    def img(self, device, dtype):
        height, width = 20, 20
        # TODO: test with a more realistic pattern
        img = torch.arange(width, device=device).div(float(width - 1))[None].expand(height, width)[None][None]
        return img

    def test_he(self, img):
        # should be similar to enhance.equalize but slower. Similar because the lut is computed in a different way.
        clip_limit: float = 0.0
        grid_size: Tuple = (1, 1)
        res = enhance.equalize_clahe(img, clip_limit=clip_limit, grid_size=grid_size)
        # NOTE: for next versions we need to improve the computation of the LUT
        # and test with a better image
        assert torch.allclose(
            res[..., 0, :],
            torch.tensor(
                [
                    [
                        [
                            0.0471,
                            0.0980,
                            0.1490,
                            0.2000,
                            0.2471,
                            0.2980,
                            0.3490,
                            0.3490,
                            0.4471,
                            0.4471,
                            0.5490,
                            0.5490,
                            0.6471,
                            0.6471,
                            0.6980,
                            0.7490,
                            0.8000,
                            0.8471,
                            0.8980,
                            1.0000,
                        ]
                    ]
                ],
                dtype=res.dtype,
                device=res.device,
            ),
            atol=1e-04,
            rtol=1e-04,
        )

    def test_ahe(self, img):
        clip_limit: float = 0.0
        grid_size: Tuple = (8, 8)
        res = enhance.equalize_clahe(img, clip_limit=clip_limit, grid_size=grid_size)
        # NOTE: for next versions we need to improve the computation of the LUT
        # and test with a better image
        assert torch.allclose(
            res[..., 0, :],
            torch.tensor(
                [
                    [
                        [
                            0.2471,
                            0.4980,
                            0.7490,
                            0.6667,
                            0.4980,
                            0.4980,
                            0.7490,
                            0.4993,
                            0.4980,
                            0.2471,
                            0.7490,
                            0.4993,
                            0.4980,
                            0.2471,
                            0.4980,
                            0.4993,
                            0.3333,
                            0.2471,
                            0.4980,
                            1.0000,
                        ]
                    ]
                ],
                dtype=res.dtype,
                device=res.device,
            ),
            atol=1e-04,
            rtol=1e-04,
        )

    def test_clahe(self, img):
        clip_limit: float = 2.0
        grid_size: Tuple = (8, 8)
        res = enhance.equalize_clahe(img, clip_limit=clip_limit, grid_size=grid_size)
        # NOTE: for next versions we need to improve the computation of the LUT
        # and test with a better image
        assert torch.allclose(
            res[..., 0, :],
            torch.tensor(
                [
                    [
                        [
                            0.1216,
                            0.8745,
                            0.9373,
                            0.9163,
                            0.8745,
                            0.8745,
                            0.9373,
                            0.8745,
                            0.8745,
                            0.8118,
                            0.9373,
                            0.8745,
                            0.8745,
                            0.8118,
                            0.8745,
                            0.8745,
                            0.8327,
                            0.8118,
                            0.8745,
                            1.0000,
                        ]
                    ]
                ],
                dtype=res.dtype,
                device=res.device,
            ),
            atol=1e-04,
            rtol=1e-04,
        )
