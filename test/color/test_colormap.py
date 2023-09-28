import pytest
import torch
from torch.autograd import gradcheck

from kornia.color import AUTUMN, ApplyColorMap, apply_colormap
from kornia.core import tensor
from kornia.testing import BaseTester, assert_close, tensor_to_gradcheck_var


def test_autumn(device, dtype):
    cm = AUTUMN(num_colors=254, device=device, dtype=dtype)
    colors = cm.colors

    actual = colors[..., 0]

    expected = tensor([1, 0, 0], device=device, dtype=dtype)

    assert_close(actual, expected)

    actual = colors[..., 127]
    expected = tensor([1, 0.5019997358322144, 0], device=device, dtype=dtype)
    assert_close(actual, expected)

    actual = colors[..., -1]
    expected = tensor([1, 1, 0], device=device, dtype=dtype)
    assert_close(actual, expected)


class TestApplyColorMap(BaseTester):
    def test_smoke(self, device, dtype):
        input_tensor = tensor([[[0, 1, 3], [25, 50, 63]]], device=device, dtype=dtype)

        expected_tensor = tensor(
            [
                [[1, 1, 1], [1, 1, 1]],
                [[0, 0.01587301587301587, 0.04761904761904762], [0.3968253968253968, 0.7936507936507936, 1]],
                [[0, 0, 0], [0, 0, 0]],
            ],
            device=device,
            dtype=dtype,
        )
        cm = AUTUMN(device=device, dtype=dtype)
        actual = apply_colormap(input_tensor, cm)

        self.assert_close(actual, expected_tensor)

    def test_eye(self, device, dtype):
        input_tensor = torch.stack(
            [torch.eye(2, dtype=dtype, device=device) * 255, torch.eye(2, dtype=dtype, device=device) * 150]
        ).view(2, -1, 2, 2)

        expected_tensor = tensor(
            [
                [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]],
            ],
            device=device,
            dtype=dtype,
        )

        actual = apply_colormap(input_tensor, AUTUMN(device=device, dtype=dtype))
        self.assert_close(actual, expected_tensor)

    def test_exception(self, device, dtype):
        cm = AUTUMN(device=device, dtype=dtype)
        with pytest.raises(TypeError):
            apply_colormap(torch.rand(size=(5, 1, 1), dtype=dtype, device=device), cm)

    @pytest.mark.parametrize("shape", [(2, 1, 4, 4), (1, 4, 4), (4, 4)])
    def test_cardinality(self, shape, device, dtype):
        cm = AUTUMN(device=device, dtype=dtype)
        input_tensor = torch.randint(0, 63, shape, device=device, dtype=dtype)
        actual = apply_colormap(input_tensor, cm)

        if len(shape) == 4:
            expected_shape = (shape[0], 3, shape[-2], shape[-1])
        else:
            expected_shape = (3, shape[-2], shape[-1])

        assert actual.shape == expected_shape

    @pytest.mark.skip(reason='jacobian mismatch')
    def test_gradcheck(self, device, dtype):
        # TODO: implement differentiability
        cm = AUTUMN(device=device, dtype=dtype)
        input_tensor = torch.randint(0, 63, (1, 2, 1), device=device, dtype=dtype)

        input_tensor = tensor_to_gradcheck_var(input_tensor)
        cm.colors = tensor_to_gradcheck_var(cm.colors)

        assert gradcheck(apply_colormap, (input_tensor, cm), raise_exception=True, fast_mode=True)

    def test_dynamo(self, device, dtype, torch_optimizer):
        op = apply_colormap
        op_script = torch_optimizer(op)

        cm = AUTUMN(device=device, dtype=dtype)
        img = torch.ones(1, 3, 3, device=device, dtype=dtype)

        self.assert_close(op(img, cm), op_script(img, cm))

    def test_module(self, device, dtype):
        op = apply_colormap
        cm = AUTUMN(device=device, dtype=dtype)
        op_module = ApplyColorMap(cm)

        img = torch.ones(1, 3, 3, device=device, dtype=dtype)

        self.assert_close(op(img, cm), op_module(img))
