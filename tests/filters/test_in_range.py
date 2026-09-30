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

from kornia.core._compat import torch_version
from kornia.filters import InRange, in_range

from testing.base import BaseTester, assert_close


def test_in_range(device, dtype):
    torch.manual_seed(1)
    # Generate on CPU first so the expected mask is device-independent, then move to target device.
    input_tensor = torch.rand(1, 3, 3, 3).to(dtype=dtype).to(device=device)
    expected = torch.tensor([[[[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]]], device=device, dtype=dtype)
    lower = (0.2, 0.3, 0.4)
    upper = (0.8, 0.9, 1.0)
    result = in_range(input_tensor, lower, upper, return_mask=True)

    assert_close(result, expected, atol=1e-4, rtol=1e-4)


class TestInRange(BaseTester):
    def _get_expected(self, device, dtype):
        return torch.tensor(
            [[[[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]]],
            device=device,
            dtype=dtype,
        )

    def test_smoke(self, device, dtype):
        torch.manual_seed(1)
        # Generate on CPU first so the expected mask is device-independent, then move to target device.
        input_tensor = torch.rand(1, 3, 3, 3).to(dtype=dtype).to(device=device)
        expected = self._get_expected(device=device, dtype=dtype)
        res = InRange(lower=(0.2, 0.3, 0.4), upper=(0.8, 0.9, 1.0), return_mask=True)(input_tensor)
        assert expected.shape == res.shape
        self.assert_close(res, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "input_shape, lower, upper",
        [
            ((1, 3, 3, 3), (0.2, 0.2, 0.2), (0.6, 0.6, 0.6)),
            ((2, 3, 3, 3), (0.2, 0.2, 0.2), (0.6, 0.6, 0.6)),
            ((5, 5, 3, 3), (0.2, 0.2, 0.2, 0.2, 0.2), (0.6, 0.6, 0.6, 0.6, 0.6)),
            ((3, 3), (0.2,), (0.6,)),
            ((2, 3, 3), (0.2, 0.2), (0.6, 0.6)),
        ],
    )
    def test_cardinality(self, input_shape, lower, upper, device, dtype):
        input_tensor = torch.rand(input_shape, device=device, dtype=dtype)
        res = InRange(lower=lower, upper=upper, return_mask=True)(input_tensor)

        if len(input_tensor.shape) == 2:
            assert res.shape == (res.shape[-2], res.shape[-1])
        elif len(input_tensor.shape) == 3:
            assert res.shape == (1, res.shape[-2], res.shape[-1])
        else:
            assert res.shape == (res.shape[0], 1, res.shape[-2], res.shape[-1])

    def test_exception(self, device, dtype):
        input_tensor = torch.rand(1, 3, 3, 3, device=device, dtype=dtype)
        with pytest.raises(Exception, match=r"Invalid `lower` and `upper` format. Should be tuple or torch\.Tensor\."):
            InRange(lower=3, upper=3)(input_tensor)

        with pytest.raises(Exception, match=r"Invalid `lower` and `upper` format. Should be tuple or torch\.Tensor\."):
            InRange(lower=[0.2, 0.2], upper=[0.2, 0.2])(input_tensor)

        with pytest.raises(Exception, match=r"Invalid `lower` and `upper` format. Should be tuple or torch\.Tensor\."):
            InRange(lower=(0.2), upper=(0.2))(input_tensor)

        with pytest.raises(
            ValueError, match=r"Shape of `lower`, `upper` and `input` image channels must have same shape."
        ):
            InRange(lower=(0.2,), upper=(0.2,))(input_tensor)

        # A 1-D Tensor bound must have one element per channel: C = 3 here.
        lower = torch.tensor([0.2, 0.2, 0.2, 0.2])
        upper = torch.tensor([0.6, 0.6, 0.6, 0.6])
        with pytest.raises(ValueError, match=r"`lower` as a Tensor must be"):
            InRange(lower=lower, upper=upper)(input_tensor)

        lower = torch.tensor([0.2, 0.2, 0.2])
        upper = torch.tensor([0.6, 0.6, 0.6])
        with pytest.raises(Exception, match=r"Invalid `return_mask` format. Should be boolean."):
            InRange(lower=lower, upper=upper, return_mask=2)(input_tensor)

    @staticmethod
    def _per_channel_bounds(channels, device, dtype):
        lower = torch.linspace(0.0, 0.3, channels, device=device, dtype=torch.float32).to(dtype)
        upper = torch.linspace(0.7, 0.95, channels, device=device, dtype=torch.float32).to(dtype)
        return lower, upper

    @staticmethod
    def _bounds_in_form(bound, form, batch_size):
        # `bound` is the per-channel (C,) vector; build the same bound in every accepted tensor layout.
        channels = bound.shape[0]
        if form == "c":
            return bound
        if form == "c11":
            return bound.reshape(channels, 1, 1)
        if form == "chw":
            return bound.reshape(channels, 1, 1).repeat(1, 4, 5)
        if form == "1c11":
            return bound.reshape(1, channels, 1, 1)
        if form == "1chw":
            return bound.reshape(1, channels, 1, 1).repeat(1, 1, 4, 5)
        if form == "bc11":
            return bound.reshape(1, channels, 1, 1).repeat(batch_size, 1, 1, 1)
        if form == "bchw":
            return bound.reshape(1, channels, 1, 1).repeat(batch_size, 1, 4, 5)
        raise AssertionError(form)

    @pytest.mark.parametrize("batch_size", [1, 3])
    @pytest.mark.parametrize("lower_form", ["c", "c11", "chw", "1c11", "1chw", "bc11", "bchw"])
    @pytest.mark.parametrize("upper_form", ["c", "c11", "chw", "1c11", "1chw", "bc11", "bchw"])
    @pytest.mark.parametrize("return_mask", [True, False])
    def test_tensor_bound_forms_match_bchw(self, batch_size, lower_form, upper_form, return_mask, device, dtype):
        # Every accepted tensor layout of a bound gives the same result as the (B, C, 1, 1) bound, in any pairing.
        torch.manual_seed(0)
        channels = 3
        img = torch.rand(batch_size, channels, 4, 5).to(device=device, dtype=dtype)
        lower_c, upper_c = self._per_channel_bounds(channels, device, dtype)
        expected = in_range(
            img,
            self._bounds_in_form(lower_c, "bc11", batch_size),
            self._bounds_in_form(upper_c, "bc11", batch_size),
            return_mask=return_mask,
        )
        lower = self._bounds_in_form(lower_c, lower_form, batch_size)
        upper = self._bounds_in_form(upper_c, upper_form, batch_size)

        actual = in_range(img, lower, upper, return_mask=return_mask)
        module_actual = InRange(lower, upper, return_mask=return_mask)(img)

        assert actual.dtype == img.dtype
        assert actual.device == img.device
        assert torch.equal(actual, expected)
        assert torch.equal(module_actual, expected)
        # Guard against a vacuous comparison: the bounds keep some pixels and reject others.
        mask = in_range(img, lower, upper, return_mask=True)
        assert 0 < mask.sum() < mask.numel()

    @staticmethod
    def _reference_mask(img, lower, upper):
        """Compute the mask with explicit loops, on the CPU.

        A bound is aligned with (B, C, H, W) from its last dimension, a dimension of size 1 is repeated, and a 1-D
        bound is read per channel.
        """
        batch, channels, height, width = img.shape

        def at(bound, b, c, y, x):
            if bound.dim() == 1:
                return float(bound[0 if bound.shape[0] == 1 else c])
            shape = (1,) * (4 - bound.dim()) + tuple(bound.shape)
            index = tuple(0 if size == 1 else i for size, i in zip(shape, (b, c, y, x)))
            return float(bound.reshape(shape)[index])

        mask = torch.zeros(batch, 1, height, width)
        for b in range(batch):
            for y in range(height):
                for x in range(width):
                    keep = all(
                        at(lower, b, c, y, x) <= float(img[b, c, y, x]) <= at(upper, b, c, y, x)
                        for c in range(channels)
                    )
                    mask[b, 0, y, x] = float(keep)
        return mask

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("pairing", ["lower", "upper", "both"])
    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param(lambda b: (), id="0d"),
            pytest.param(lambda b: (1,), id="1d-one-element"),
            pytest.param(lambda b: (3,), id="c"),
            pytest.param(lambda b: (3, 1, 1), id="c11"),
            pytest.param(lambda b: (1, 3, 1, 1), id="1c11"),
            pytest.param(lambda b: (b, 3, 1, 1), id="bc11"),
            pytest.param(lambda b: (3, 4, 5), id="chw"),
            pytest.param(lambda b: (1, 3, 4, 5), id="1chw"),
            pytest.param(lambda b: (b, 3, 4, 5), id="bchw"),
            pytest.param(lambda b: (4, 5), id="hw"),
            pytest.param(lambda b: (1, 1, 4, 5), id="11hw"),
            pytest.param(lambda b: (b, 1, 1, 1), id="b111"),
            pytest.param(lambda b: (1, 5), id="1w"),
        ],
    )
    def test_tensor_bound_layouts_match_explicit_loops(self, batch_size, pairing, shape, device, dtype):
        # H != W != C, so a bound applied along the wrong axis changes the mask. `lower` / `upper` pair the layout
        # with a (B, C, 1, 1) bound; `both` uses the layout for both bounds.
        gen = torch.Generator().manual_seed(0)
        img = torch.rand(batch_size, 3, 4, 5, generator=gen).to(dtype)
        layout = shape(batch_size)
        lower_t = (torch.rand(layout, generator=gen) * 0.3).to(dtype)
        upper_t = (0.7 + torch.rand(layout, generator=gen) * 0.3).to(dtype)
        lower_b = (torch.rand(batch_size, 3, 1, 1, generator=gen) * 0.3).to(dtype)
        upper_b = (0.7 + torch.rand(batch_size, 3, 1, 1, generator=gen) * 0.3).to(dtype)
        lower, upper = {"lower": (lower_t, upper_b), "upper": (lower_b, upper_t), "both": (lower_t, upper_t)}[pairing]

        expected = self._reference_mask(img, lower, upper).to(device=device, dtype=dtype)
        assert 0 < expected.sum() < expected.numel()  # not a vacuous comparison

        actual = in_range(img.to(device), lower.to(device), upper.to(device), return_mask=True)
        assert actual.dtype == dtype
        assert actual.device == img.to(device).device
        assert torch.equal(actual, expected)
        assert torch.equal(InRange(lower.to(device), upper.to(device), return_mask=True)(img.to(device)), expected)

    def test_one_dim_bound_is_per_channel_not_per_column(self, device, dtype):
        # (C,) must apply along the channel axis: a bound of 0 on channel 2 only clears the pixels through channel 2.
        img = torch.full((2, 3, 4, 5), 0.5, device=device, dtype=dtype)
        lower = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)
        upper = torch.tensor([1.0, 1.0, 0.0], device=device, dtype=dtype)
        assert not in_range(img, lower, upper, return_mask=True).any()
        upper = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)
        assert in_range(img, lower, upper, return_mask=True).all()

    def test_one_dim_bound_as_long_as_channels_and_width_is_per_channel(self, device, dtype):
        # C == W == 5: the shape cannot tell a per-channel bound from a per-column one, and a 1-D bound is per channel.
        # Channel 4 has an upper bound of 0, so every pixel is cleared; a per-column reading would keep columns 0-3.
        img = torch.full((2, 5, 4, 5), 0.5, device=device, dtype=dtype)
        lower = torch.zeros(5, device=device, dtype=dtype)
        upper = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0], device=device, dtype=dtype)
        assert not in_range(img, lower, upper, return_mask=True).any()
        assert not InRange(lower, upper, return_mask=True)(img).any()
        assert not in_range(img, lower, upper, return_mask=False).any()

    @pytest.mark.parametrize("input_shape", [(3, 4, 5), (2, 3, 4, 5), (2, 2, 3, 4, 5)])
    def test_one_dim_bound_leading_dims(self, input_shape, device, dtype):
        # `in_range` flattens leading dims into the batch; a (C,) bound is shared by every flattened image.
        torch.manual_seed(0)
        img = torch.rand(input_shape, device=device, dtype=dtype)
        lower_c, upper_c = self._per_channel_bounds(3, device, dtype)
        expected = in_range(img, tuple(lower_c.tolist()), tuple(upper_c.tolist()), return_mask=True)
        actual = in_range(img, lower_c, upper_c, return_mask=True)
        assert actual.shape == expected.shape
        self.assert_close(actual, expected)

    def test_bounds_are_inclusive(self, device, dtype):
        # lower <= input <= upper: values equal to a bound are kept (these values are exact in every dtype).
        img = torch.tensor([0.125, 0.25, 0.5, 0.75, 0.875], device=device, dtype=dtype).reshape(1, 1, 1, 5)
        expected = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0], device=device, dtype=dtype).reshape(1, 1, 1, 5)
        by_tuple = in_range(img, (0.25,), (0.75,), return_mask=True)
        by_tensor = in_range(
            img, torch.tensor([0.25]).reshape(1, 1, 1, 1), torch.tensor([0.75]).reshape(1, 1, 1, 1), return_mask=True
        )
        self.assert_close(by_tuple, expected)
        self.assert_close(by_tensor, expected)

    def test_multi_channel_mask_requires_every_channel(self, device, dtype):
        # The conjunction runs over the C channels: one channel out of range clears the pixel.
        img = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.95], device=device, dtype=dtype).reshape(1, 3, 1, 2)
        mask = in_range(img, (0.0, 0.0, 0.0), (1.0, 1.0, 0.9), return_mask=True)
        self.assert_close(mask, torch.tensor([[[[1.0, 0.0]]]], device=device, dtype=dtype))

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize(
        "bad_shape",
        [
            pytest.param(lambda b: (4,), id="1d-C+1"),
            pytest.param(lambda b: (5,), id="1d-W"),  # a 1-D bound is per channel, so W elements fit no axis
            pytest.param(lambda b: (3, 1), id="2d-C1"),
            pytest.param(lambda b: (5, 4), id="2d-WH"),
            pytest.param(lambda b: (4, 4, 5), id="3d-C+1HW"),
            pytest.param(lambda b: (3, 5, 4), id="3d-CWH"),
            pytest.param(lambda b: (1, b, 3, 1, 1), id="5d"),
            pytest.param(lambda b: (1, 1, 1, 1, 1), id="5d-all-ones"),  # every dimension fits, only the rank is wrong
            pytest.param(lambda b: (b, 4, 1, 1), id="4d-C+1"),
            pytest.param(lambda b: (b, 1, 3, 1), id="4d-channels-on-height-axis"),
            pytest.param(lambda b: (b + 1, 3, 1, 1), id="4d-B+1"),
            pytest.param(lambda b: (b, 3, 7, 1), id="4d-height-neither-1-nor-H"),
        ],
    )
    @pytest.mark.parametrize("bad_bound", ["lower", "upper"])
    def test_mis_shaped_tensor_bound_raises(self, batch_size, bad_shape, bad_bound, device, dtype):
        # Every bound is validated on its own: a mis-shaped one raises, whatever the other bound is.
        img = torch.rand(batch_size, 3, 4, 5, device=device, dtype=dtype)
        bad = torch.full(bad_shape(batch_size), 0.5, device=device, dtype=dtype)
        match = rf"`{bad_bound}` as a Tensor must be"

        # The other bound is a well-formed (B, C, 1, 1) Tensor, then a well-formed (C,) Tensor.
        for good_shape in [(batch_size, 3, 1, 1), (3,)]:
            good = torch.full(good_shape, 0.5, device=device, dtype=dtype)
            lower, upper = (bad, good) if bad_bound == "lower" else (good, bad)
            with pytest.raises(ValueError, match=match):
                in_range(img, lower, upper)
            with pytest.raises(ValueError, match=match):
                InRange(lower, upper)(img)

    @pytest.mark.parametrize(
        "input_shape, input_view",
        [
            ((2, 3, 4, 5), (2, 3, 4, 5)),
            ((3, 4, 5), (1, 3, 4, 5)),
            ((4, 5), (1, 1, 4, 5)),
            ((2, 2, 3, 4, 5), (4, 3, 4, 5)),
        ],
    )
    @pytest.mark.parametrize("bad_bound", ["lower", "upper"])
    def test_error_message_quotes_the_bound_shape_and_the_input_view(
        self, input_shape, input_view, bad_bound, device, dtype
    ):
        # The error quotes the shape of the offending bound and the (B, C, H, W) view of the input it was checked
        # against.
        img = torch.rand(input_shape, device=device, dtype=dtype)
        bad = torch.zeros(7, device=device, dtype=dtype)
        good = torch.zeros(input_view[1], device=device, dtype=dtype)
        lower, upper = (bad, good) if bad_bound == "lower" else (good, bad)
        with pytest.raises(ValueError) as error:
            in_range(img, lower, upper)
        message = str(error.value)
        assert message.startswith(f"`{bad_bound}` as a Tensor must be")
        assert f"Got (7,) for an input viewed as (B, C, H, W) = {input_view}." in message

    @pytest.mark.parametrize("tensor_bound", ["lower", "upper"])
    def test_tensor_bound_with_tuple_bound_raises(self, tensor_bound, device, dtype):
        # A Tensor bound paired with a tuple bound raises a TypeError that names the rule, whatever the Tensor's shape.
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        for shape in [(5,), (3,), (2, 3, 1, 1)]:
            bound = torch.full(shape, 0.5, device=device, dtype=dtype)
            lower, upper = (bound, (0.9, 0.9, 0.9)) if tensor_bound == "lower" else ((0.1, 0.1, 0.1), bound)
            with pytest.raises(TypeError, match=r"Both should be tuples or both torch\.Tensor"):
                in_range(img, lower, upper)
            with pytest.raises(TypeError, match=r"Both should be tuples or both torch\.Tensor"):
                InRange(lower, upper)(img)

    def test_tensor_bounds_follow_input_dtype_and_device(self, device, dtype):
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        lower = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)  # created on the CPU, not on `device`
        upper = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
        out = in_range(img, lower, upper, return_mask=False)
        mask = in_range(img, lower, upper, return_mask=True)
        assert out.dtype == dtype
        assert out.device == img.device
        assert mask.dtype == dtype
        assert mask.device == img.device

    @pytest.mark.parametrize("bound_shape", [(1,), (1, 1, 1, 1)])
    def test_tensor_bounds_are_cast_to_the_input_dtype(self, bound_shape, device, dtype):
        # 0.1 is not exact in any dtype. The input holds 0.1 rounded to `dtype`; a bound of a wider dtype holds a
        # different number and only equals the input once it is cast to the input's dtype, so the pixel is kept only
        # if it is. The bound is created on the CPU, which also covers devices without float64.
        bound_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else torch.float64
        img = torch.full((1, 1, 1, 1), 0.1, dtype=torch.float64).to(dtype).to(device)
        bound = torch.full(bound_shape, 0.1, dtype=bound_dtype)
        assert bound.item() != img.item() or dtype == torch.float64  # the bound really is a different number
        mask = in_range(img, bound, bound, return_mask=True)
        assert mask.dtype == dtype
        assert mask.all()
        self.assert_close(in_range(img, bound, bound, return_mask=False), img)

    def test_tensor_bounds_return_masked_input(self, device, dtype):
        # Exercises the Tensor-bounds branch with return_mask=False
        inp = torch.ones(1, 3, 4, 4, device=device, dtype=dtype) * 0.5
        lower = torch.tensor([0.2, 0.2, 0.2], device=device, dtype=dtype).reshape(1, 3, 1, 1)
        upper = torch.tensor([0.8, 0.8, 0.8], device=device, dtype=dtype).reshape(1, 3, 1, 1)
        out = in_range(inp, lower, upper, return_mask=False)
        # All pixels are in range, so output == input
        self.assert_close(out, inp)

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(1, 3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)
        actual = InRange((0.2, 0.2, 0.2), (0.6, 0.6, 0.6), return_mask=True)(inp)
        assert actual.is_contiguous()

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 3, 5, 5
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(in_range, (img, (0.2, 0.2, 0.2), (0.6, 0.6, 0.6), True))

    @pytest.mark.parametrize(
        "input_shape, lower, upper",
        [
            ((1, 3, 3, 3), (0.2, 0.2, 0.2), (0.6, 0.6, 0.6)),
            ((2, 3, 3, 3), (0.2, 0.2, 0.2), (0.6, 0.6, 0.6)),
            ((3, 3), (0.2,), (0.6,)),
        ],
    )
    def test_module(self, input_shape, lower, upper, device, dtype):
        img = torch.rand(input_shape, device=device, dtype=dtype)
        op = in_range
        op_module = InRange(lower=lower, upper=upper, return_mask=True)
        actual = op_module(img)
        expected = op(img, lower, upper, True)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_dynamo(self, batch_size, device, dtype, torch_optimizer):
        if device == torch.device("cpu") and torch_version() in {"2.3.0", "2.3.1"}:
            pytest.skip("Failing to compile on CPU see pytorch/pytorch#126619")
        data = torch.rand(batch_size, 3, 5, 5, device=device, dtype=dtype)
        op = InRange(lower=(0.2, 0.2, 0.2), upper=(0.6, 0.6, 0.6), return_mask=True)
        op_optimized = torch_optimizer(op, fullgraph=True)
        self.assert_close(op(data), op_optimized(data))

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("form", ["c", "c11", "1c11", "chw", "bc11"])
    def test_dynamo_tensor_bounds(self, batch_size, form, device, dtype, torch_optimizer):
        if device == torch.device("cpu") and torch_version() in {"2.3.0", "2.3.1"}:
            pytest.skip("Failing to compile on CPU see pytorch/pytorch#126619")
        data = torch.rand(batch_size, 3, 4, 5, device=device, dtype=dtype)
        lower_c, upper_c = self._per_channel_bounds(3, device, dtype)
        lower = self._bounds_in_form(lower_c, form, batch_size)
        upper = self._bounds_in_form(upper_c, form, batch_size)
        op = InRange(lower=lower, upper=upper, return_mask=True)
        op_optimized = torch_optimizer(op, fullgraph=True)
        self.assert_close(op(data), op_optimized(data))


class TestConventionsInRange(BaseTester):
    def test_convention_in_range_bounds_inclusive_and_all_channels_must_pass(self, device, dtype):
        # built in the test dtype so that a value equal to a bound is equal after rounding
        row = [0.2, 0.5, 0.8, 0.9]
        img = torch.tensor([[row, row]] * 3, device=device, dtype=dtype)[None]  # (1, 3, 2, 4), H != W
        img[0, 2, 1, 1] = 0.95  # above channel 2's upper bound 0.9 at one pixel only
        img[0, 0, 1, 0] = float("nan")  # at a pixel that is otherwise in range
        lower, upper = (0.2, 0.2, 0.2), (0.8, 0.8, 0.9)
        mask = in_range(img, lower, upper, return_mask=True)
        # both bounds inclusive (0.2 and 0.8 pass), a pixel passes only if every channel does, NaN fails;
        # the mask is (B, 1, H, W) in the input dtype with 1 (not 255) for a pass
        expected = torch.tensor([[[[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]], device=device, dtype=dtype)
        assert mask.dtype == dtype
        self.assert_close(mask, expected)
        # return_mask=False zeroes every channel of a failing pixel
        self.assert_close(in_range(img, lower, upper).nan_to_num(), (img * expected).nan_to_num())
        # relabel: transposing the image transposes the mask
        self.assert_close(in_range(img.transpose(-1, -2), lower, upper, return_mask=True), expected.transpose(-1, -2))
        # lower > upper is not rejected; it selects nothing
        assert in_range(img, upper, lower, return_mask=True).sum().item() == 0

    def test_convention_in_range_bounds_are_cast_to_input_dtype(self, device, dtype):
        # on an integer image a fractional bound truncates: 100.7 -> 100 admits the value 100, a float image does not
        values = torch.tensor([100, 101, 200, 201], device=device, dtype=torch.uint8).view(1, 1, 1, 4)
        as_uint8 = in_range(values, (100.7,), (200.2,), return_mask=True)
        assert as_uint8.dtype == torch.uint8
        assert as_uint8.flatten().tolist() == [1, 1, 1, 0]
        as_float = in_range(values.to(dtype), (100.7,), (200.2,), return_mask=True)
        assert as_float.flatten().tolist() == [0, 1, 1, 0]

    def test_wart_in_range_checks_one_tensor_bound_shape_5176(self, device, dtype):
        """#5176: a (W,) upper bound is accepted when the lower bound is (B, C, 1, 1), and applied per column."""
        generator = torch.Generator().manual_seed(0)
        img = torch.rand(2, 3, 4, 5, generator=generator).to(device=device, dtype=dtype)
        lower = torch.zeros(2, 3, 1, 1, device=device, dtype=dtype)
        upper = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0], device=device, dtype=dtype)
        mask = in_range(img, lower, upper, return_mask=True)
        assert mask.sum(dim=(0, 1, 2)).tolist() == [8, 8, 8, 8, 0]
