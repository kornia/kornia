import pytest

from kornia import enhance
from kornia.testing import tensor_to_gradcheck_var, BaseTester
import kornia.testing as utils

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


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

    @pytest.mark.parametrize("clip, grid", [(0., None), (None, (2, 2)), (2., (2, 2))])
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

    @pytest.mark.parametrize("B, clip, grid, exception_type", [
        (0, 1., (2, 2), ValueError),
        (1, 1, (2, 2), TypeError),
        (1, 2., 2, TypeError),
        (1, 2., (2, 2, 2), TypeError),
        (1, 2., (2, 2.), TypeError),
        (1, 2., (2, 0), ValueError)])
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

    def test_gradcheck(self, device, dtype):
        pass

    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 10, 20
        inp = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        op = enhance.equalize_clahe
        op_script = torch.jit.script(op)
        assert_allclose(op(inp), op_script(inp))

    def test_module(self):
        # equalize_clahe is only a function
        pass
