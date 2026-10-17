# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import torch

import kornia

from testing.base import BaseTester


class TestNormalize(BaseTester):
    @pytest.mark.parametrize("shape", [(2, 3, 4, 5), (2, 3, 2, 4, 5)])
    def test_noncontiguous(self, shape, device, dtype):
        data = torch.rand(shape, device=device, dtype=dtype).transpose(-1, -2)
        mean = torch.tensor([0.25, 0.5, 0.75], device=device, dtype=dtype)
        std = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=dtype)
        broadcast_shape = (1, 3) + (1,) * (data.ndim - 2)
        expected = (data - mean.reshape(broadcast_shape)) / std.reshape(broadcast_shape)

        assert not data.is_contiguous()
        self.assert_close(kornia.enhance.normalize(data, mean, std), expected)
        self.assert_close(kornia.enhance.Normalize(mean, std)(data), expected)

    def test_noncontiguous_gradcheck(self, device):
        data = torch.rand(1, 2, 3, 4, device=device, dtype=torch.float64).transpose(-1, -2)
        mean = torch.tensor([0.25, 0.5], device=device, dtype=torch.float64)
        std = torch.tensor([0.5, 2.0], device=device, dtype=torch.float64)
        self.gradcheck(kornia.enhance.Normalize(mean, std), (data,))

    def test_smoke(self, device, dtype):
        mean = [0.5]
        std = [0.1]
        repr = "Normalize(mean=[0.5], std=[0.1])"
        assert str(kornia.enhance.Normalize(mean, std)) == repr

    def test_normalize(self, device, dtype):
        # prepare input data
        data = torch.ones(1, 2, 2, device=device, dtype=dtype)
        mean = torch.tensor([0.5], device=device, dtype=dtype)
        std = torch.tensor([2.0], device=device, dtype=dtype)

        # expected output
        expected = torch.tensor([0.25], device=device, dtype=dtype).repeat(1, 2, 2).view_as(data)

        f = kornia.enhance.Normalize(mean, std)
        self.assert_close(f(data), expected)

    def test_empty_batch(self, device, dtype):
        data = torch.rand(0, 3, 6, 8, device=device, dtype=dtype, requires_grad=True)
        mean = torch.tensor([0.5], device=device, dtype=dtype, requires_grad=True)
        std = torch.tensor([0.5], device=device, dtype=dtype, requires_grad=True)

        output = kornia.enhance.normalize(data, mean, std)
        augmentation_output = kornia.augmentation.Normalize(mean, std, p=1.0)(data)

        assert output.shape == data.shape
        assert output.device == data.device
        assert output.dtype == data.dtype
        assert augmentation_output.shape == data.shape

        output.sum().backward()

        assert data.grad is not None
        assert mean.grad is not None
        assert std.grad is not None
        assert data.grad.abs().sum() == 0
        assert mean.grad.abs().sum() == 0
        assert std.grad.abs().sum() == 0

    def test_rank2_normalize(self, device, dtype):
        data = torch.ones(2, 3, device=device, dtype=dtype)
        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=dtype)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype)

        expected = (data - mean) / std

        self.assert_close(kornia.enhance.normalize(data, mean, std), expected)

    def test_empty_rank2_normalize(self, device, dtype):
        data = torch.empty(0, 3, device=device, dtype=dtype)
        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=dtype)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype)

        output = kornia.enhance.normalize(data, mean, std)

        assert output.shape == data.shape
        self.assert_close(output, data)

    def test_broadcast_normalize(self, device, dtype):
        # prepare input data
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        data += 2

        mean = torch.tensor([2.0], device=device, dtype=dtype)
        std = torch.tensor([0.5], device=device, dtype=dtype)

        # expected output
        expected = torch.ones_like(data) + 1

        f = kornia.enhance.Normalize(mean, std)
        self.assert_close(f(data), expected)

    def test_int_input(self, device, dtype):
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        data += 2

        mean: int = 2
        std: int = 1

        # expected output
        expected = torch.ones_like(data)

        f = kornia.enhance.Normalize(mean, std)
        self.assert_close(f(data), expected)

    def test_float_input(self, device, dtype):
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        data += 2

        mean: float = 2.0
        std: float = 0.5

        # expected output
        expected = torch.ones_like(data) + 1

        f = kornia.enhance.Normalize(mean, std)
        self.assert_close(f(data), expected)

    def test_batch_normalize(self, device, dtype):
        # prepare input data
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        data += 2

        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype).repeat(2, 1)

        # expected output
        expected = torch.tensor([1.25, 1, 0.5], device=device, dtype=dtype).repeat(2, 1, 1).view_as(data)

        f = kornia.enhance.Normalize(mean, std)
        self.assert_close(f(data), expected)

    @pytest.mark.skip(reason="union type not supported")
    def test_jit(self, device, dtype):
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        inputs = (data, mean, std)

        op = kornia.enhance.normalize
        op_script = torch.jit.script(op)

        self.assert_close(op(*inputs), op_script(*inputs))

    def test_gradcheck(self, device):
        # prepare input data
        data = torch.ones(2, 3, 1, 1, device=device, dtype=torch.float64)
        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=torch.float64).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=torch.float64).repeat(2, 1)

        self.gradcheck(kornia.enhance.Normalize(mean, std), (data,))

    def test_single_value(self, device, dtype):
        # prepare input data
        mean = torch.tensor(2, device=device, dtype=dtype)
        std = torch.tensor(3, device=device, dtype=dtype)
        data = torch.ones(2, 3, 16, 17, device=device, dtype=dtype)

        # expected output
        expected = (data - mean) / std

        self.assert_close(kornia.enhance.normalize(data, mean, std), expected)

    def test_module(self, device, dtype):
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        inputs = (data, mean, std)

        op = kornia.enhance.normalize
        op_module = kornia.enhance.Normalize(mean, std)

        self.assert_close(op(*inputs), op_module(data))

    @pytest.mark.parametrize(
        "mean, std", [((1.0, 1.0, 1.0), (0.5, 0.5, 0.5)), (1.0, 0.5), (torch.tensor([1.0]), torch.tensor([0.5]))]
    )
    def test_random_normalize_different_parameter_types(self, mean, std):
        f = kornia.enhance.Normalize(mean=mean, std=std)
        data = torch.ones(2, 3, 16, 17)
        if isinstance(mean, float):
            expected = (data - torch.as_tensor(mean)) / torch.as_tensor(std)
        else:
            expected = (data - torch.as_tensor(mean[0])) / torch.as_tensor(std[0])
        self.assert_close(f(data), expected)

    @pytest.mark.parametrize("mean, std", [((1.0, 1.0, 1.0, 1.0), (0.5, 0.5, 0.5, 0.5)), ((1.0, 1.0), (0.5, 0.5))])
    def test_random_normalize_invalid_parameter_shape(self, mean, std):
        f = kornia.enhance.Normalize(mean=mean, std=std)
        inputs = torch.arange(0.0, 16.0, step=1).reshape(1, 4, 4).unsqueeze(0)
        with pytest.raises((ValueError, RuntimeError)):
            f(inputs)

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass


class TestNormalizeConstantsAreBuffers(BaseTester):
    """Tensor arguments move with the module; Python constants follow the input."""

    @staticmethod
    def _modules():
        return {
            "Normalize": kornia.enhance.Normalize(torch.zeros(3), torch.ones(3)),
            "Denormalize": kornia.enhance.Denormalize(torch.zeros(3), torch.ones(3)),
            "Rescale": kornia.enhance.Rescale(torch.tensor(2.0)),
        }

    def test_constants_are_registered_buffers(self, device, dtype):
        for name, module in self._modules().items():
            assert list(module.buffers()), f"{name} registered no buffers"

    @pytest.mark.parametrize(
        "name,attrs",
        [
            ("Normalize", ("mean", "std")),
            ("Denormalize", ("mean", "std")),
            ("Rescale", ("factor",)),
        ],
    )
    def test_to_moves_the_constants(self, name, attrs, device, dtype):
        """dtype stands in for device, so the check runs on CPU-only CI.

        Asserted on the attributes by name rather than by iterating buffers:
        iterating would pass vacuously on a module that registered none, which
        is exactly the bug.
        """
        moved = self._modules()[name].to(torch.float64)
        for attr in attrs:
            value = getattr(moved, attr)
            assert isinstance(value, torch.Tensor), f"{name}.{attr} is not a tensor"
            assert value.dtype == torch.float64, f"{name}.{attr} stayed {value.dtype} after .to(float64)"

    def test_the_constants_stay_out_of_the_state_dict(self, device, dtype):
        """They are constructor arguments, not learned state.

        Registering them persistently would make every existing checkpoint
        report unexpected keys, so they are non-persistent buffers.
        """
        for name, module in self._modules().items():
            assert module.state_dict() == {}, f"{name} state_dict is not empty"

    def test_denormalize_coerces_scalars(self, device, dtype):
        """Scalar arguments stay as Python values and are materialized per input."""
        module = kornia.enhance.Denormalize(0.0, 255.0)
        assert module.mean == 0.0
        assert module.std == 255.0
        assert not list(module.buffers())

    def test_python_constants_preserve_float64_precision(self):
        x = torch.full((1, 3, 1, 1), 0.3, dtype=torch.float64)
        normalize = kornia.enhance.Normalize(0.1, 0.3).double()
        denormalize = kornia.enhance.Denormalize(0.1, 0.3).double()
        rescale = kornia.enhance.Rescale(0.1).double()

        assert torch.equal(normalize(x), (x - 0.1) / 0.3)
        assert torch.equal(denormalize(x), x * 0.3 + 0.1)
        assert torch.equal(rescale(x), x * 0.1)

    def test_normalize_sequence_constants_preserve_float64_precision(self):
        x = torch.full((1, 2, 1, 1), 0.3, dtype=torch.float64)
        module = kornia.enhance.Normalize((0.1, 0.2), (0.3, 0.4))
        mean = torch.tensor([[0.1, 0.2]], dtype=torch.float64)
        std = torch.tensor([[0.3, 0.4]], dtype=torch.float64)

        assert torch.equal(module(x), (x - mean.reshape(1, 2, 1, 1)) / std.reshape(1, 2, 1, 1))

    def test_denormalize_sequence_is_checked_against_the_channels(self):
        """A (C,) list must not broadcast a 1-channel input to C channels."""
        module = kornia.enhance.Denormalize([0.1, 0.2, 0.3], [1.0, 1.0, 1.0])
        with pytest.raises(ValueError):
            module(torch.ones(2, 1, 4, 4))

    def test_denormalize_nested_sequence_gives_per_sample_statistics(self):
        x = torch.ones(2, 3, 1, 1)
        out = kornia.enhance.Denormalize([[0.0] * 3, [1.0] * 3], [[1.0] * 3, [2.0] * 3])(x)
        expected = torch.tensor([1.0, 3.0]).view(2, 1, 1, 1).expand(2, 3, 1, 1)
        assert torch.equal(out, expected)

    def test_python_constants_take_the_onnx_branch(self, monkeypatch):
        """The ONNX branch indexes ``mean.shape[0]``, so a 0-d constant would raise IndexError."""
        monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: True)
        x = torch.full((1, 3, 2, 2), 0.5)
        self.assert_close(kornia.enhance.Normalize(0.1, 0.4)(x), torch.full_like(x, 1.0))
        self.assert_close(kornia.enhance.Denormalize(0.1, 0.4)(x), torch.full_like(x, 0.3))

    def test_forward_is_unchanged(self, device, dtype):
        x = torch.rand(1, 3, 4, 4, device=device, dtype=dtype)
        mean = torch.zeros(3, device=device, dtype=dtype)
        std = 2.0 * torch.ones(3, device=device, dtype=dtype)
        self.assert_close(
            kornia.enhance.Normalize(mean, std)(x),
            kornia.enhance.normalize(x, mean, std),
        )
        self.assert_close(
            kornia.enhance.Denormalize(mean, std)(x),
            kornia.enhance.denormalize(x, mean, std),
        )
        self.assert_close(kornia.enhance.Rescale(2.0)(x), x * 2.0)

    def test_smoke(self, device, dtype):
        pass

    def test_cardinality(self, device, dtype):
        pass

    def test_exception(self, device, dtype):
        pass

    def test_gradcheck(self, device):
        pass

    def test_module(self, device, dtype):
        pass

    def test_dynamo(self, device, dtype, torch_optimizer):
        pass


class TestDenormalize(BaseTester):
    def test_smoke(self, device, dtype):
        mean = [0.5]
        std = [0.1]
        repr = "Denormalize(mean=[0.5], std=[0.1])"
        assert str(kornia.enhance.Denormalize(mean, std)) == repr

    def test_denormalize(self, device, dtype):
        # prepare input data
        data = torch.ones(1, 2, 2, device=device, dtype=dtype)
        mean = torch.tensor([0.5])
        std = torch.tensor([2.0])

        # expected output
        expected = torch.tensor([2.5], device=device, dtype=dtype).repeat(1, 2, 2).view_as(data)

        f = kornia.enhance.Denormalize(mean, std)
        self.assert_close(f(data), expected)

    def test_broadcast_denormalize(self, device, dtype):
        # prepare input data
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        data += 2

        mean = torch.tensor([2.0], device=device, dtype=dtype)
        std = torch.tensor([0.5], device=device, dtype=dtype)

        # expected output
        expected = torch.ones_like(data) + 2.5

        f = kornia.enhance.Denormalize(mean, std)
        self.assert_close(f(data), expected)

    def test_float_input(self, device, dtype):
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        data += 2

        mean: float = 2.0
        std: float = 0.5

        # expected output
        expected = torch.ones_like(data) + 2.5

        f = kornia.enhance.Denormalize(mean, std)
        self.assert_close(f(data), expected)

    def test_batch_denormalize(self, device, dtype):
        # prepare input data
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        data += 2

        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype).repeat(2, 1)

        # expected output
        expected = torch.tensor([6.5, 7, 8], device=device, dtype=dtype).repeat(2, 1, 1).view_as(data)

        f = kornia.enhance.Denormalize(mean, std)
        self.assert_close(f(data), expected)

    @pytest.mark.skip(reason="union type not supported")
    def test_jit(self, device, dtype):
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        inputs = (data, mean, std)

        op = kornia.enhance.denormalize
        op_script = torch.jit.script(op)

        self.assert_close(op(*inputs), op_script(*inputs))

    def test_gradcheck(self, device):
        # prepare input data
        data = torch.ones(2, 3, 1, 1, device=device, dtype=torch.float64)
        data += 2
        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=torch.float64)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=torch.float64)

        self.gradcheck(kornia.enhance.Denormalize(mean, std), (data,))

    def test_single_value(self, device, dtype):
        # prepare input data
        mean = torch.tensor(2, device=device, dtype=dtype)
        std = torch.tensor(3, device=device, dtype=dtype)
        data = torch.ones(2, 3, 256, 313, device=device, dtype=dtype)

        # expected output
        expected = (data * std) + mean

        self.assert_close(kornia.enhance.denormalize(data, mean, std), expected)

    def test_module(self, device, dtype):
        data = torch.ones(2, 3, 1, 1, device=device, dtype=dtype)
        mean = torch.tensor([0.5, 1.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype).repeat(2, 1)
        inputs = (data, mean, std)

        op = kornia.enhance.denormalize
        op_module = kornia.enhance.Denormalize(mean, std)

        self.assert_close(op(*inputs), op_module(data))

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass


class TestNormalizeIntegerInput(BaseTester):
    """`normalize` and `denormalize` used to cast mean and std to an integer input's dtype (#5403).

    A fractional statistic then truncated to an integer (0.5 -> 0) and the arithmetic ran in the integer dtype, so
    `normalize` divided by zero and `denormalize` wrapped. Integer and bool inputs are now promoted to torch's
    default floating dtype first, as `x * 0.5` does, and both functions return that dtype.
    """

    INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool)

    @staticmethod
    def _image(device, input_dtype):
        g = torch.Generator().manual_seed(0)
        values = torch.randint(0, 2 if input_dtype is torch.bool else 128, (2, 3, 4, 5), generator=g)
        return values.to(device, input_dtype)

    @pytest.mark.parametrize("input_dtype", INTEGER_DTYPES)
    @pytest.mark.parametrize("fn", ["normalize", "denormalize"])
    @pytest.mark.parametrize("stats", ["float", "tensor"])
    def test_integer_input_matches_the_float_path_5403(self, device, input_dtype, fn, stats):
        # The same values held as float32 are the reference: nothing is truncated, wrapped or non-finite.
        op = getattr(kornia.enhance, fn)
        img = self._image(device, input_dtype)
        if stats == "float":
            mean, std = 0.5, 0.25
        else:
            mean = torch.tensor([0.5, 0.25, 0.75], device=device)
            std = torch.tensor([0.25, 0.5, 2.0], device=device)

        out = op(img, mean, std)

        assert out.dtype == torch.get_default_dtype()
        assert out.shape == img.shape
        assert torch.isfinite(out).all()
        self.assert_close(out, op(img.to(torch.get_default_dtype()), mean, std))

    def test_issue_values_5403(self, device):
        x = torch.tensor([[[[0, 128, 255]]]], device=device, dtype=torch.uint8)

        # 0.5 used to truncate to 0: [nan, inf, inf]
        self.assert_close(
            kornia.enhance.normalize(x, 0.5, 0.5),
            torch.tensor([[[[-1.0, 255.0, 509.0]]]], device=device),
        )
        # 0 - 100 used to wrap to 156 in uint8: [3.12, 0.56, 3.10]
        self.assert_close(
            kornia.enhance.normalize(x, 100.0, 50.0),
            torch.tensor([[[[-2.0, 0.56, 3.1]]]], device=device),
        )
        # used to return [0, 0, 0] uint8
        self.assert_close(
            kornia.enhance.denormalize(x, 0.5, 0.5),
            torch.tensor([[[[0.5, 64.5, 128.0]]]], device=device),
        )
        # 128 * 2 + 10 and 255 * 2 + 10 used to wrap in uint8: [10, 10, 8]
        self.assert_close(
            kornia.enhance.denormalize(x, 10.0, 2.0),
            torch.tensor([[[[10.0, 266.0, 520.0]]]], device=device),
        )

    @pytest.mark.parametrize("module", ["Normalize", "Denormalize"])
    def test_modules_promote_integer_input_5403(self, device, module):
        img = self._image(device, torch.uint8)
        op = getattr(kornia.enhance, module)(0.5, 0.25).to(device)

        out = op(img)

        assert out.dtype == torch.get_default_dtype()
        assert torch.isfinite(out).all()
        self.assert_close(out, op(img.to(torch.get_default_dtype())))

    def test_uint8_pixel_scale_round_trip_5403(self, device):
        # Statistics in the 0-255 pixel scale are what a caller holding a uint8 image passes.
        mean = torch.tensor([123.675, 116.28, 103.53], device=device)
        std = torch.tensor([58.395, 57.12, 57.375], device=device)
        img = torch.randint(0, 256, (2, 3, 6, 7), device=device, dtype=torch.uint8)

        normalized = kornia.enhance.normalize(img, mean, std)

        assert normalized.dtype == torch.get_default_dtype()
        assert normalized.abs().max() < 3.0
        self.assert_close(kornia.enhance.denormalize(normalized, mean, std), img.to(normalized.dtype))

    def test_integer_input_follows_the_default_dtype_5403(self, device):
        # The promotion target is torch's default floating dtype, the same as `x * 0.5`, not a hardcoded float32.
        if device.type == "mps":
            pytest.skip("MPS does not support float64")
        img = self._image(device, torch.uint8)
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            assert (img * 0.5).dtype == torch.float64
            assert kornia.enhance.normalize(img, 0.5, 0.25).dtype == torch.float64
            assert kornia.enhance.denormalize(img, 0.5, 0.25).dtype == torch.float64
        finally:
            torch.set_default_dtype(previous)

    def test_float_input_keeps_its_dtype_5403(self, device, dtype):
        # Only integer and bool inputs are promoted; a floating input is returned in its own dtype.
        img = torch.rand(1, 3, 4, 4, device=device, dtype=dtype)
        assert kornia.enhance.normalize(img, 0.5, 0.25).dtype == dtype
        assert kornia.enhance.denormalize(img, 0.5, 0.25).dtype == dtype

    @pytest.mark.parametrize("fn", ["normalize", "denormalize"])
    def test_complex_input_is_not_cast_5403(self, device, fn):
        # Promoting a complex tensor to a real dtype would drop its imaginary part, so it must stay complex.
        op = getattr(kornia.enhance, fn)
        img = torch.complex(torch.rand(1, 3, 2, 2), torch.rand(1, 3, 2, 2)).to(device)

        out = op(img, 0.5, 0.25)

        assert out.dtype == img.dtype
        assert out.imag.abs().sum() > 0


class TestNormalizeMinMax(BaseTester):
    @pytest.mark.parametrize("shape", [(2, 3, 6, 8), (0, 3, 6, 8), (2, 0, 3, 6, 8)])
    def test_dynamo(self, shape, device, dtype, torch_optimizer):
        data = torch.rand(shape, device=device, dtype=dtype)
        op = kornia.enhance.normalize_min_max
        self.assert_close(torch_optimizer(op)(data), op(data))

    @pytest.mark.parametrize("shape", [(0, 3, 6, 8), (2, 0, 3, 6, 8), (0, 2, 3, 6, 8), (2, 1, 0, 3, 6, 8)])
    def test_empty_leading_dimensions(self, shape, device, dtype):
        data = torch.empty(shape, device=device, dtype=dtype, requires_grad=True)
        actual = kornia.enhance.normalize_min_max(input=data, min_val=-2.0, max_val=3.0)
        assert actual.shape == data.shape
        assert actual.dtype == data.dtype
        assert actual.device == data.device
        assert actual.numel() == 0
        actual.sum().backward()
        assert data.grad is not None
        assert data.grad.shape == data.shape

    @pytest.mark.parametrize("shape", [(0, 8), (3, 0, 8), (0, 6, 8), (2, 0, 6, 8), (0, 3, 0, 8), (0, 3, 6, 0)])
    def test_empty_image_planes_rejected(self, shape, device, dtype):
        with pytest.raises(ValueError):
            kornia.enhance.normalize_min_max(torch.empty(shape, device=device, dtype=dtype))

    @pytest.mark.parametrize("shape", [(), (3,), (0,)])
    def test_invalid_rank(self, shape, device, dtype):
        with pytest.raises(ValueError, match="at least two dimensions"):
            kornia.enhance.normalize_min_max(torch.empty(shape, device=device, dtype=dtype))

    @pytest.mark.parametrize("kwargs", [{"min_val": 0}, {"max_val": 1}])
    def test_empty_batch_validates_range_types(self, kwargs, device, dtype):
        with pytest.raises(TypeError):
            kornia.enhance.normalize_min_max(torch.empty(0, 3, 6, 8, device=device, dtype=dtype), **kwargs)

    @pytest.mark.parametrize("shape", [(2, 3), (2, 2, 3), (2, 2, 2, 3), (2, 2, 2, 2, 3)])
    def test_tied_extrema_gradients(self, shape, device, dtype):
        # Indexed extrema must retain the first-tie gradient behavior of the BCHW implementation.
        data = torch.tensor([0.0, 0.0, 1.0, 2.0, 2.0, 1.0], device=device, dtype=dtype)
        data = data.repeat(torch.Size(shape).numel() // 6).reshape(shape).requires_grad_()
        reference = data.detach().clone().requires_grad_()
        planes = reference.reshape(-1, 6)
        low = planes.min(-1, keepdim=True)[0]
        high = planes.max(-1, keepdim=True)[0]
        expected = (3.0 * (planes - low) / (high - low + 1e-6) - 1.0).reshape(shape)
        actual = kornia.enhance.normalize_min_max(data, min_val=-1.0, max_val=2.0)
        self.assert_close(actual, expected)
        weights = torch.arange(data.numel(), device=device, dtype=dtype).reshape(shape)
        (actual * weights).sum().backward()
        (expected * weights).sum().backward()
        self.assert_close(data.grad, reference.grad)

    @pytest.mark.parametrize("shape", [(4, 5), (3, 4, 5), (2, 3, 4, 5), (2, 2, 3, 4, 5)])
    def test_noncontiguous(self, shape, device, dtype):
        data = torch.rand(shape, device=device, dtype=dtype).transpose(-1, -2)
        low = data.amin(dim=(-2, -1), keepdim=True)
        high = data.amax(dim=(-2, -1), keepdim=True)
        expected = 3.0 * (data - low) / (high - low + 1e-6) - 1.0

        assert not data.is_contiguous()
        actual = kornia.enhance.normalize_min_max(data, min_val=-1.0, max_val=2.0)
        self.assert_close(actual, expected)

    def test_noncontiguous_gradcheck(self, device):
        data = torch.arange(12, device=device, dtype=torch.float64).reshape(1, 1, 3, 4).transpose(-1, -2)
        self.gradcheck(kornia.enhance.normalize_min_max, (data,))

    def test_smoke(self, device, dtype):
        x = torch.ones(1, 1, 1, 1, device=device, dtype=dtype)
        assert kornia.enhance.normalize_min_max(x) is not None
        assert kornia.enhance.normalize_min_max(x) is not None

    def test_exception(self, device, dtype):
        x = torch.ones(1, 1, 3, 4, device=device, dtype=dtype)
        with pytest.raises(TypeError):
            assert kornia.enhance.normalize_min_max(0.0)

        with pytest.raises(TypeError):
            assert kornia.enhance.normalize_min_max(x, "", "")

        with pytest.raises(TypeError):
            assert kornia.enhance.normalize_min_max(x, 2.0, "")

    @pytest.mark.parametrize("input_shape", [(1, 2, 3, 4), (2, 1, 4, 3), (1, 3, 2, 1)])
    def test_cardinality(self, device, dtype, input_shape):
        x = torch.rand(input_shape, device=device, dtype=dtype)
        assert kornia.enhance.normalize_min_max(x).shape == input_shape

    @pytest.mark.parametrize("min_val, max_val", [(1.0, 2.0), (2.0, 3.0), (5.0, 20.0), (40.0, 1000.0)])
    def test_range(self, device, dtype, min_val, max_val):
        x = torch.rand(1, 2, 4, 5, device=device, dtype=dtype)
        out = kornia.enhance.normalize_min_max(x, min_val=min_val, max_val=max_val)
        self.assert_close(out.min(), torch.tensor(min_val, device=device, dtype=dtype), low_tolerance=True)
        self.assert_close(out.max(), torch.tensor(max_val, device=device, dtype=dtype), low_tolerance=True)

    def test_values(self, device, dtype):
        x = torch.tensor([[[[0.0, 1.0, 3.0], [-1.0, 4.0, 3.0], [9.0, 5.0, 2.0]]]], device=device, dtype=dtype)

        expected = torch.tensor(
            [[[[-0.8, -0.6, -0.2], [-1.0, 0.0, -0.2], [1.0, 0.2, -0.4]]]], device=device, dtype=dtype
        )

        actual = kornia.enhance.normalize_min_max(x, min_val=-1.0, max_val=1.0)
        self.assert_close(actual, expected, low_tolerance=True)

    @pytest.mark.parametrize(
        "shape", [(1, 1, 1, 1), (6, 8), (3, 6, 8), (2, 3, 6, 8), (2, 2, 3, 6, 8), (0, 3, 6, 8), (2, 0, 3, 6, 8)]
    )
    def test_jit(self, shape, device, dtype):
        # Non-constant planes of every rank: the scripted function must normalise the same (*, C, H, W) planes as eager.
        x = torch.arange(torch.Size(shape).numel(), device=device, dtype=dtype).reshape(shape)
        op = kornia.enhance.normalize_min_max
        op_jit = torch.jit.script(op)
        self.assert_close(op(x), op_jit(x))

    def test_gradcheck(self, device):
        x = torch.ones(1, 1, 1, 1, device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(kornia.enhance.normalize_min_max, (x,))

    def test_3d_tensor(self, device, dtype):
        # Test with 3D tensor (C, H, W) - the main bug fix
        x = torch.tensor([[[0.0, 1.0, 3.0], [-1.0, 4.0, 3.0], [9.0, 5.0, 2.0]]], device=device, dtype=dtype)

        # Expected: normalized to [-1, 1] range
        expected = torch.tensor([[[-0.8, -0.6, -0.2], [-1.0, 0.0, -0.2], [1.0, 0.2, -0.4]]], device=device, dtype=dtype)

        actual = kornia.enhance.normalize_min_max(x, min_val=-1.0, max_val=1.0)

        # Verify shape is preserved
        assert actual.shape == x.shape
        self.assert_close(actual, expected, low_tolerance=True)

    def test_3d_tensor_multiple_channels(self, device, dtype):
        # Test with 3D tensor with multiple channels (C, H, W)
        x = torch.rand(3, 4, 5, device=device, dtype=dtype)
        out = kornia.enhance.normalize_min_max(x, min_val=0.0, max_val=1.0)

        # Verify shape is preserved
        assert out.shape == x.shape

        # Verify per-channel normalization
        for c in range(x.shape[0]):
            channel_out = out[c]
            self.assert_close(channel_out.min(), torch.tensor(0.0, device=device, dtype=dtype), low_tolerance=True)
            self.assert_close(channel_out.max(), torch.tensor(1.0, device=device, dtype=dtype), low_tolerance=True)

    def test_2d_tensor(self, device, dtype):
        # Test with 2D tensor (H, W)
        x = torch.tensor([[0.0, 5.0], [10.0, 15.0]], device=device, dtype=dtype)
        out = kornia.enhance.normalize_min_max(x, min_val=0.0, max_val=1.0)

        # Verify shape is preserved
        assert out.shape == x.shape

        # Verify normalization
        expected = torch.tensor([[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]], device=device, dtype=dtype)
        self.assert_close(out, expected, low_tolerance=True)

    @pytest.mark.parametrize("input_shape", [(3, 4, 5), (1, 32, 32), (4, 8, 8)])
    def test_3d_shapes(self, device, dtype, input_shape):
        # Test various 3D tensor shapes
        x = torch.rand(input_shape, device=device, dtype=dtype)
        out = kornia.enhance.normalize_min_max(x, min_val=-1.0, max_val=1.0)

        # Verify shape is preserved
        assert out.shape == input_shape

    def test_keyword_argument(self, device, dtype):
        # Regression test for #3745: the image keyword must match the documented `input` signature.
        x = torch.rand(1, 2, 4, 5, device=device, dtype=dtype)
        out_kwarg = kornia.enhance.normalize_min_max(input=x, min_val=-1.0, max_val=1.0)
        out_positional = kornia.enhance.normalize_min_max(x, min_val=-1.0, max_val=1.0)
        self.assert_close(out_kwarg, out_positional)
