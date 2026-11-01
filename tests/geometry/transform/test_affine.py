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

import itertools
import math
import warnings

import numpy as np
import pytest
import torch

import kornia
from kornia.core.exceptions import DeviceError, TypeCheckError

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, dynamo_is_available, supports_reflect_padding


class TestResize(BaseTester):
    @pytest.mark.parametrize(
        "shape, divisible_factor, expected",
        [
            ((10, 14), 24, (24, 24)),
            ((1, 3, 10, 14), 32, (1, 3, 32, 32)),
            ((3, 10, 14), 4, (3, 8, 16)),
            ((3, 14, 10), 4, (3, 16, 8)),
            ((3, 10, 14), 24, (3, 24, 24)),
            ((1, 3, 10, 14), 24, (1, 3, 24, 24)),
            ((2, 1, 3, 10, 14), 24, (2, 1, 3, 24, 24)),
        ],
    )
    def test_resize_to_be_divisible_short_sides(self, shape, divisible_factor, expected, device, dtype):
        inp = torch.rand(*shape, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize_to_be_divisible(inp, divisible_factor)
        assert out.shape == expected

    @pytest.mark.parametrize("kwargs", [{}, {"side": None}])
    def test_resize_to_be_divisible_default_side(self, kwargs, device, dtype):
        inp = torch.rand(1, 1, 6, 12, device=device, dtype=dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            out = kornia.geometry.transform.resize_to_be_divisible(inp, 5, **kwargs)
        assert out.shape == (1, 1, 5, 10)

    @pytest.mark.parametrize("shape", [(1, 1, 6, 12), (2, 3, 17, 9), (1, 1, 5, 10), (1, 1, 0, 12)])
    @pytest.mark.parametrize("side", ["short", "long", "vert", "horz", "bogus"])
    def test_resize_to_be_divisible_deprecated_side(self, shape, side, device, dtype):
        inp = torch.rand(*shape, device=device, dtype=dtype)
        expected = kornia.geometry.transform.resize_to_be_divisible(inp, 5)
        with pytest.warns(DeprecationWarning, match="side.*has no effect") as records:
            out = kornia.geometry.transform.resize_to_be_divisible(inp, 5, side=side)
        assert len(records) == 1
        assert records[0].filename == __file__
        self.assert_close(out, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("side", ["short", "long", "vert", "horz", "bogus"])
    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    def test_resize_to_be_divisible_compile_fullgraph(self, side, device, dtype, torch_optimizer, optimizer_backend):
        if not optimizer_backend or optimizer_backend == "jit":
            pytest.skip("Fullgraph requires a torch.compile backend, not TorchScript or the no-op optimizer.")
        inp = torch.rand(1, 1, 6, 12, device=device, dtype=dtype)
        expected = kornia.geometry.transform.resize_to_be_divisible(inp, 5)
        op = torch_optimizer(kornia.geometry.transform.resize_to_be_divisible, fullgraph=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=r".*resize_to_be_divisible.*side", category=DeprecationWarning)
            out = op(inp, 5, side=side)
        self.assert_close(out, expected)

    def test_resize_to_be_divisible_numpy_factor_and_empty_side(self, device, dtype):
        resize_to_be_divisible = kornia.geometry.transform.resize_to_be_divisible
        inp = torch.rand(1, 3, 10, 14, device=device, dtype=dtype)
        assert resize_to_be_divisible(inp, np.int64(4)).shape == (1, 3, 8, 16)
        # An empty side is already divisible and stays empty, as in resize.
        empty = torch.zeros(1, 3, 0, 14, device=device, dtype=dtype)
        assert resize_to_be_divisible(empty, 4).shape == (1, 3, 0, 16)
        assert resize_to_be_divisible(empty.transpose(-1, -2), 4).shape == (1, 3, 16, 0)

    def test_resize_to_be_divisible_rejects_invalid_inputs(self, device, dtype):
        resize_to_be_divisible = kornia.geometry.transform.resize_to_be_divisible
        with pytest.raises(TypeError, match="is not a torch"):
            resize_to_be_divisible([[0.0, 1.0]], 4)
        with pytest.raises(ValueError, match="at least two dimensions"):
            resize_to_be_divisible(torch.rand(4, device=device, dtype=dtype), 4)
        inp = torch.rand(4, 4, device=device, dtype=dtype)
        for divisible_factor in (0, -1, 2.5):
            with pytest.raises(ValueError, match="positive integer"):
                resize_to_be_divisible(inp, divisible_factor)

    def test_smoke(self, device, dtype):
        inp = torch.rand(1, 3, 3, 4, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (3, 4), align_corners=False)
        self.assert_close(inp, out, atol=1e-4, rtol=1e-4)

        # 2D
        inp = torch.rand(3, 4, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (3, 4), align_corners=False)
        self.assert_close(inp, out, atol=1e-4, rtol=1e-4)

        # 3D
        inp = torch.rand(3, 3, 4, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (3, 4), align_corners=False)
        self.assert_close(inp, out, atol=1e-4, rtol=1e-4)

        # arbitrary dim
        inp = torch.rand(1, 2, 3, 2, 1, 3, 3, 4, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (3, 4), align_corners=False)
        self.assert_close(inp, out, atol=1e-4, rtol=1e-4)

    def test_upsize(self, device, dtype):
        inp = torch.rand(1, 3, 3, 4, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (6, 8), align_corners=False)
        assert out.shape == (1, 3, 6, 8)

        # 2D
        inp = torch.rand(3, 4, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (6, 8), align_corners=False)
        assert out.shape == (6, 8)

        # 3D
        inp = torch.rand(3, 3, 4, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (6, 8), align_corners=False)
        assert out.shape == (3, 6, 8)

        # arbitrary dim
        inp = torch.rand(1, 2, 3, 2, 1, 3, 3, 4, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (6, 8), align_corners=False)
        assert out.shape == (1, 2, 3, 2, 1, 3, 6, 8)

    def test_downsize(self, device, dtype):
        inp = torch.rand(1, 3, 5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (3, 1), align_corners=False)
        assert out.shape == (1, 3, 3, 1)

        # 2D
        inp = torch.rand(5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (3, 1), align_corners=False)
        assert out.shape == (3, 1)

        # 3D
        inp = torch.rand(3, 5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (3, 1), align_corners=False)
        assert out.shape == (3, 3, 1)

        # arbitrary dim
        inp = torch.rand(1, 2, 3, 2, 1, 3, 5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (3, 1), align_corners=False)
        assert out.shape == (1, 2, 3, 2, 1, 3, 3, 1)

    def test_zero_output_size(self, device, dtype):
        # A zero output side gives an empty image, matching warp_affine,
        # warp_perspective and center_crop on the same argument.
        inp = torch.rand(1, 3, 4, 4, device=device, dtype=dtype)

        assert kornia.geometry.transform.resize(inp, (0, 4)).shape == (1, 3, 0, 4)
        assert kornia.geometry.transform.resize(inp, (4, 0)).shape == (1, 3, 4, 0)
        assert kornia.geometry.transform.resize(inp, (0, 0)).shape == (1, 3, 0, 0)
        assert kornia.geometry.transform.resize(inp, 0).shape == (1, 3, 0, 0)
        assert kornia.geometry.transform.Resize((0, 4))(inp).shape == (1, 3, 0, 4)
        assert kornia.geometry.transform.resize(inp, (0, 4), antialias=True).shape == (1, 3, 0, 4)

        eye = torch.eye(2, 3, device=device, dtype=dtype)[None]
        assert kornia.geometry.transform.warp_affine(inp, eye, (0, 4)).shape == (1, 3, 0, 4)

        # The empty result stays attached to the graph, as it does for the warping ops:
        # _empty_warp_output_2d builds a connected tensor precisely so a batch that
        # degenerates to a zero-sized output still contributes a zero gradient.
        grad_inp = torch.rand(1, 3, 4, 4, device=device, dtype=dtype, requires_grad=True)
        empty = kornia.geometry.transform.resize(grad_inp, (0, 4))
        assert empty.requires_grad
        assert empty.dtype == grad_inp.dtype
        empty.sum().backward()
        assert grad_inp.grad is not None

        # 2D
        inp_2d = torch.rand(4, 4, device=device, dtype=dtype)
        assert kornia.geometry.transform.resize(inp_2d, (0, 4)).shape == (0, 4)

        # 3D
        inp_3d = torch.rand(3, 4, 4, device=device, dtype=dtype)
        assert kornia.geometry.transform.resize(inp_3d, (0, 4)).shape == (3, 0, 4)

        # arbitrary dim
        inp_nd = torch.rand(2, 1, 3, 4, 4, device=device, dtype=dtype)
        assert kornia.geometry.transform.resize(inp_nd, (0, 4)).shape == (2, 1, 3, 0, 4)

    def test_zero_input_size(self, device, dtype):
        # A degenerate input is rejected with the message the warping ops already
        # use, instead of a bare ZeroDivisionError out of the aspect-ratio division.
        inp = torch.rand(1, 3, 0, 4, device=device, dtype=dtype)

        with pytest.raises(ValueError, match="Input image size must be positive"):
            kornia.geometry.transform.resize(inp, (2, 2))

        with pytest.raises(ValueError, match="Input image size must be positive"):
            kornia.geometry.transform.resize(inp, 2)

        # Empty in and empty out is not undefined, so it comes back empty rather than
        # raising -- the same answer warp_affine and center_crop give.
        assert kornia.geometry.transform.resize(inp, (0, 4)).shape == (1, 3, 0, 4)
        assert kornia.geometry.transform.resize(inp, (0, 2)).shape == (1, 3, 0, 2)

        eye = torch.eye(2, 3, device=device, dtype=dtype)[None]
        assert kornia.geometry.transform.warp_affine(inp, eye, (0, 4)).shape == (1, 3, 0, 4)
        assert kornia.geometry.transform.center_crop(inp, (0, 4)).shape == (1, 3, 0, 4)

    def test_downsizeAA(self, device, dtype):
        inp = torch.rand(1, 3, 10, 8, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (5, 3), align_corners=False, antialias=True)
        assert out.shape == (1, 3, 5, 3)

        inp = torch.rand(1, 1, 20, 10, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (15, 8), align_corners=False, antialias=True)
        assert out.shape == (1, 1, 15, 8)

        # 2D
        inp = torch.rand(10, 8, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (5, 3), align_corners=False, antialias=True)
        assert out.shape == (5, 3)

        # 3D
        inp = torch.rand(3, 10, 8, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (5, 3), align_corners=False, antialias=True)
        assert out.shape == (3, 5, 3)

        # arbitrary dim
        inp = torch.rand(1, 2, 3, 2, 1, 3, 10, 8, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, (5, 3), align_corners=False, antialias=True)
        assert out.shape == (1, 2, 3, 2, 1, 3, 5, 3)

    @pytest.mark.parametrize("leading_shape", [(), (2,), (2, 3), (2, 1, 3)])
    @pytest.mark.parametrize("transpose", [False, True])
    @pytest.mark.parametrize("singleton_output", [1, 3])
    def test_antialias_singleton_axis(self, leading_shape, transpose, singleton_output, device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip(f"this torch build has no reflect padding kernel for {dtype} on {device.type}")
        # Repeating a row cannot change its horizontal antialiasing. The repeated image also
        # supplies a reference that the old reflect-padding path can process.
        row = torch.arange(8, device=device, dtype=dtype).reshape(*([1] * len(leading_shape)), 1, 8)
        image = row.expand(*leading_shape, 1, 8)
        repeated = image.expand(*leading_shape, 3, 8)
        size = (singleton_output, 4)
        reference_size = (3, 4)
        if transpose:
            image = image.transpose(-1, -2)
            repeated = repeated.transpose(-1, -2)
            size = size[::-1]
            reference_size = reference_size[::-1]

        expected = kornia.geometry.transform.resize(repeated, reference_size, antialias=True)
        if transpose:
            expected = expected[..., :1].expand(*leading_shape, 4, singleton_output)
        else:
            expected = expected[..., :1, :].expand(*leading_shape, singleton_output, 4)
        actual = kornia.geometry.transform.resize(image, size, antialias=True)
        assert actual.shape == (*leading_shape, *size)
        assert actual.dtype == dtype
        assert actual.device == device
        self.assert_close(actual, expected)
        # The downscaled axis is still filtered, including at its reflected endpoints.
        unfiltered = kornia.geometry.transform.resize(image, size)
        assert not torch.allclose(actual, unfiltered)

    @pytest.mark.parametrize("transpose", [False, True])
    @pytest.mark.parametrize("rows", [2, 3])
    def test_antialias_blurs_a_two_or_three_pixel_axis(self, rows, transpose, device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip(f"this torch build has no reflect padding kernel for {dtype} on {device.type}")
        # Only a singleton axis skips the Gaussian. Two or three rows shrunk to one are blurred with sigma
        # (rows - 1) / 2 over taps -(rows - 1) .. rows - 1, reflected at the border; the columns keep their size, so
        # their kernel is the identity. Nearest keeps row 0 of the blurred rows, and row i is row 0 + 8 i.
        image = torch.arange(rows * 8, device=device, dtype=dtype).reshape(1, 1, rows, 8)
        taps = torch.arange(1 - rows, rows, dtype=torch.float64)
        weights = torch.exp(-(taps**2) / (2 * ((rows - 1) / 2) ** 2))
        offset = 8 * float((weights * taps.abs()).sum() / weights.sum())
        expected = torch.arange(8, dtype=torch.float64).add(offset).reshape(1, 1, 1, 8)
        size = (1, 8)
        if transpose:
            image, expected, size = image.transpose(-1, -2), expected.transpose(-1, -2), size[::-1]
        actual = kornia.geometry.transform.resize(image, size, interpolation="nearest", antialias=True)
        self.assert_close(actual, expected.to(device, dtype))

    @pytest.mark.parametrize("transpose", [False, True])
    def test_antialias_singleton_wrappers(self, transpose, device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip(f"this torch build has no reflect padding kernel for {dtype} on {device.type}")
        image = torch.arange(8, device=device, dtype=dtype).reshape(1, 1, 1, 8)
        size, factor = (1, 4), (1.0, 0.5)
        if transpose:
            image = image.transpose(-1, -2)
            size, factor = size[::-1], factor[::-1]
        expected = kornia.geometry.transform.resize(image, size, antialias=True)
        self.assert_close(kornia.geometry.transform.Resize(size, antialias=True)(image), expected)
        self.assert_close(kornia.geometry.transform.rescale(image, factor, antialias=True), expected)

    @pytest.mark.parametrize("transpose", [False, True])
    def test_antialias_singleton_gradcheck(self, transpose, device):
        image = torch.arange(8, device=device, dtype=torch.float64).reshape(1, 1, 1, 8).requires_grad_()
        size = (1, 4)
        if transpose:
            image, size = image.transpose(-1, -2), size[::-1]
        self.gradcheck(lambda x: kornia.geometry.transform.resize(x, size, antialias=True), (image,))

    def test_dynamo_singleton_antialias(self, device, dtype, torch_optimizer):
        if not supports_reflect_padding(device, dtype):
            pytest.skip(f"this torch build has no reflect padding kernel for {dtype} on {device.type}")
        image = torch.arange(8, device=device, dtype=dtype).reshape(1, 1, 1, 8)
        op = torch_optimizer(kornia.geometry.transform.resize, fullgraph=True)
        expected = kornia.geometry.transform.resize(image, (1, 4), antialias=True)
        self.assert_close(op(image, (1, 4), antialias=True), expected)

    def test_one_param(self, device, dtype):
        inp = torch.rand(1, 3, 5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False)
        assert out.shape == (1, 3, 25, 10)

        # 2D
        inp = torch.rand(5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False)
        assert out.shape == (25, 10)

        # 3D
        inp = torch.rand(3, 5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False)
        assert out.shape == (3, 25, 10)

        # arbitrary dim
        inp = torch.rand(1, 2, 3, 2, 1, 3, 5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False)
        assert out.shape == (1, 2, 3, 2, 1, 3, 25, 10)

    def test_one_param_long(self, device, dtype):
        inp = torch.rand(1, 3, 5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False, side="long")
        assert out.shape == (1, 3, 10, 4)

        # 2D
        inp = torch.rand(5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False, side="long")
        assert out.shape == (10, 4)

        # 3D
        inp = torch.rand(3, 5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False, side="long")
        assert out.shape == (3, 10, 4)

        # arbitrary dim
        inp = torch.rand(1, 2, 3, 2, 1, 3, 5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False, side="long")
        assert out.shape == (1, 2, 3, 2, 1, 3, 10, 4)

    def test_one_param_vert(self, device, dtype):
        inp = torch.rand(1, 3, 5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False, side="vert")
        assert out.shape == (1, 3, 10, 4)

        # 2D
        inp = torch.rand(5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False, side="vert")
        assert out.shape == (10, 4)

        # 3D
        inp = torch.rand(3, 5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False, side="vert")
        assert out.shape == (3, 10, 4)

        # arbitrary dim
        inp = torch.rand(1, 2, 3, 2, 1, 3, 5, 2, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False, side="vert")
        assert out.shape == (1, 2, 3, 2, 1, 3, 10, 4)

    def test_one_param_horz(self, device, dtype):
        inp = torch.rand(1, 3, 2, 5, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False, side="horz")
        assert out.shape == (1, 3, 4, 10)

        # 2D
        inp = torch.rand(1, 3, 2, 5, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False, side="horz")
        assert out.shape == (1, 3, 4, 10)

        # 3D
        inp = torch.rand(1, 3, 2, 5, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False, side="horz")
        assert out.shape == (1, 3, 4, 10)

        # arbitrary dim
        inp = torch.rand(1, 3, 2, 5, device=device, dtype=dtype)
        out = kornia.geometry.transform.resize(inp, 10, align_corners=False, side="horz")
        assert out.shape == (1, 3, 4, 10)

    def test_convention_align_corners_default_none(self, device, dtype):
        # resize's bare default align_corners=None behaves like align_corners=False (no test in the
        # suite exercises the bare default; every other call here passes align_corners=False explicitly).
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("hardcoded-literal pin only reliable at float32/float64 precision")
        # Snippet used to generate expected:
        # inp_x = torch.arange(20, dtype=torch.float32) / 20.0
        # inp = inp_x[None].T @ inp_x[None]
        # inp = inp[None, None]
        # expected = kornia.geometry.transform.resize(inp, (5, 5), antialias=False)  # no align_corners passed
        inp_x = torch.arange(20, device=device, dtype=dtype) / 20.0
        inp = inp_x[None].T @ inp_x[None]
        inp = inp[None, None]
        out = kornia.geometry.transform.resize(inp, (5, 5), antialias=False)
        expected = torch.tensor(
            [
                [
                    [
                        [0.0056, 0.0206, 0.0356, 0.0506, 0.0656],
                        [0.0206, 0.0756, 0.1306, 0.1856, 0.2406],
                        [0.0356, 0.1306, 0.2256, 0.3206, 0.4156],
                        [0.0506, 0.1856, 0.3206, 0.4556, 0.5906],
                        [0.0656, 0.2406, 0.4156, 0.5906, 0.7656],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(out, expected, atol=1e-3, rtol=1e-3)
        # the output above cannot distinguish None from False (they interpolate identically), so
        # the documented default is additionally pinned on the signature itself
        import inspect

        assert inspect.signature(kornia.geometry.transform.resize).parameters["align_corners"].default is None

    def test_gradcheck(self, device):
        # test parameters
        new_size = 4
        inp = torch.rand(1, 2, 3, 4, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.transform.Resize(new_size, align_corners=False), (inp,))

    @pytest.mark.parametrize("anti_alias", [True, False])
    def test_dynamo(self, device, dtype, anti_alias, torch_optimizer):
        new_size = (5, 6)
        inp = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        op = torch_optimizer(kornia.geometry.transform.resize)
        out = op(inp, new_size, align_corners=False, antialias=anti_alias)
        assert out.shape == (1, 2, 5, 6)
        expected = op(inp, new_size, align_corners=False, antialias=anti_alias)
        self.assert_close(out, expected)

    @pytest.mark.parametrize("interpolation", ["linear", "trilinear"])
    @pytest.mark.parametrize("shape", [(3, 16, 20), (2, 3, 16, 20), (2, 3, 4, 16, 20)])
    @pytest.mark.parametrize("api", ["resize", "Resize", "rescale", "Rescale"])
    def test_one_and_three_dimensional_modes_raise_6012(self, interpolation, shape, api, device, dtype):
        # Every input is resampled as a 4-D batch, which F.interpolate rejects for these modes; this used to
        # surface as its raw "Got 4D input" NotImplementedError although the docstrings listed both modes.
        image = torch.rand(shape, device=device, dtype=dtype)
        T = kornia.geometry.transform
        calls = {
            "resize": lambda: T.resize(image, (8, 10), interpolation=interpolation),
            "Resize": lambda: T.Resize((8, 10), interpolation=interpolation)(image),
            "rescale": lambda: T.rescale(image, 0.5, interpolation=interpolation),
            "Rescale": lambda: T.Rescale(0.5, interpolation=interpolation, align_corners=None)(image),
        }

        with pytest.raises(ValueError, match=f"interpolation='{interpolation}' is not supported: resize resamples"):
            calls[api]()

    @pytest.mark.parametrize("interpolation", ["nearest", "nearest-exact", "bilinear", "bicubic", "area"])
    def test_documented_modes_resize_6012(self, interpolation, device, dtype):
        # Each mode the docstrings list resamples a (B, C, H, W) batch.
        image = torch.rand(2, 3, 16, 20, device=device, dtype=dtype)

        out = kornia.geometry.transform.resize(image, (8, 10), interpolation=interpolation)

        assert out.shape == (2, 3, 8, 10)

    @pytest.mark.parametrize("interpolation", ["linear", "trilinear"])
    @pytest.mark.parametrize("api", ["resize_empty_output", "resize_to_be_divisible", "crop_by_indices"])
    def test_one_and_three_dimensional_modes_raise_in_resize_callers_6012(self, interpolation, api, device, dtype):
        # An empty output side used to skip interpolate and return an empty tensor; resize_to_be_divisible, and
        # crop_by_indices when a crop has to be resized, reach these modes through resize.
        image = torch.zeros(2, 3, 16, 20, device=device, dtype=dtype)
        box = torch.tensor([[[1, 1], [10, 1], [10, 8], [1, 8]]], device=device, dtype=dtype).repeat(2, 1, 1)
        T = kornia.geometry.transform
        calls = {
            "resize_empty_output": lambda: T.resize(image, (0, 10), interpolation=interpolation),
            "resize_to_be_divisible": lambda: T.resize_to_be_divisible(image, 7, interpolation=interpolation),
            "crop_by_indices": lambda: T.crop_by_indices(image, box, size=(5, 6), interpolation=interpolation),
        }

        with pytest.raises(ValueError, match=f"interpolation='{interpolation}' is not supported: resize resamples"):
            calls[api]()


class TestRescale(BaseTester):
    def test_smoke(self, device, dtype):
        input = torch.rand(1, 3, 3, 4, device=device, dtype=dtype)
        output = kornia.geometry.transform.rescale(input, (1.0, 1.0), align_corners=False)
        self.assert_close(input, output, atol=1e-4, rtol=1e-4)

    def test_upsize(self, device, dtype):
        input = torch.rand(1, 3, 3, 4, device=device, dtype=dtype)
        output = kornia.geometry.transform.rescale(input, (3.0, 2.0), align_corners=False)
        assert output.shape == (1, 3, 9, 8)

    def test_downsize(self, device, dtype):
        input = torch.rand(1, 3, 9, 8, device=device, dtype=dtype)
        output = kornia.geometry.transform.rescale(input, (1.0 / 3.0, 1.0 / 2.0), align_corners=False)
        assert output.shape == (1, 3, 3, 4)

    def test_zero_factor(self, device, dtype):
        # A zero factor asks for a zero-sized output, which gives an empty image
        # rather than a ZeroDivisionError.
        inp = torch.rand(1, 3, 4, 4, device=device, dtype=dtype)

        assert kornia.geometry.transform.rescale(inp, 0.0).shape == (1, 3, 0, 0)
        assert kornia.geometry.transform.rescale(inp, (0.0, 1.0)).shape == (1, 3, 0, 4)
        assert kornia.geometry.transform.Rescale(0.0)(inp).shape == (1, 3, 0, 0)

    def test_downscale_values(self, device, dtype):
        inp_x = torch.arange(20, device=device, dtype=dtype) / 20.0
        inp = inp_x[None].T @ inp_x[None]
        inp = inp[None, None]
        out = kornia.geometry.transform.rescale(inp, (0.25, 0.25), antialias=False, align_corners=False)
        expected = torch.tensor(
            [
                [
                    [
                        [0.0056, 0.0206, 0.0356, 0.0506, 0.0656],
                        [0.0206, 0.0756, 0.1306, 0.1856, 0.2406],
                        [0.0356, 0.1306, 0.2256, 0.3206, 0.4156],
                        [0.0506, 0.1856, 0.3206, 0.4556, 0.5906],
                        [0.0656, 0.2406, 0.4156, 0.5906, 0.7656],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(out, expected, atol=1e-3, rtol=1e-3)

    def test_downscale_values_AA(self, device, dtype):
        inp_x = torch.arange(20, device=device, dtype=dtype) / 20.0
        inp = inp_x[None].T @ inp_x[None]
        inp = inp[None, None]
        out = kornia.geometry.transform.rescale(inp, (0.25, 0.25), antialias=True, align_corners=False)
        expected = torch.tensor(
            [
                [
                    [
                        [0.0074, 0.0237, 0.0409, 0.0581, 0.0743],
                        [0.0237, 0.0756, 0.1306, 0.1856, 0.2376],
                        [0.0409, 0.1306, 0.2256, 0.3206, 0.4104],
                        [0.0581, 0.1856, 0.3206, 0.4556, 0.5832],
                        [0.0743, 0.2376, 0.4104, 0.5832, 0.7464],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(out, expected, atol=1e-3, rtol=1e-3)

    def test_one_param(self, device, dtype):
        input = torch.rand(1, 3, 3, 4, device=device, dtype=dtype)
        output = kornia.geometry.transform.rescale(input, 2.0, align_corners=False)
        assert output.shape == (1, 3, 6, 8)

    def test_gradcheck(self, device):
        input = torch.rand(1, 2, 3, 4, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.transform.Rescale(2.0, align_corners=False), (input,), nondet_tol=1e-8)

    @pytest.mark.parametrize("interpolation", ["nearest", "nearest-exact", "area"])
    @pytest.mark.parametrize("factor", [0.5, 2.0])
    def test_non_interpolating_modes_ignore_align_corners_5998(self, interpolation, factor, device, dtype):
        # The module's default align_corners=True used to reach F.interpolate, which rejects it for these modes.
        input = torch.rand(2, 3, 4, 6, device=device, dtype=dtype)

        out = kornia.geometry.transform.Rescale(factor, interpolation=interpolation)(input)

        expected = kornia.geometry.transform.rescale(input, factor, interpolation=interpolation)
        self.assert_close(out, expected, rtol=0.0, atol=0.0)
        explicit = kornia.geometry.transform.Rescale(factor, interpolation=interpolation, align_corners=False)(input)
        self.assert_close(explicit, expected, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("interpolation", ["bilinear", "bicubic"])
    def test_interpolating_modes_keep_align_corners_default_5998(self, interpolation, device, dtype):
        # The interpolating modes still use the module's default align_corners=True.
        input = torch.rand(2, 3, 4, 6, device=device, dtype=dtype)

        out = kornia.geometry.transform.Rescale(0.5, interpolation=interpolation)(input)

        expected = kornia.geometry.transform.rescale(input, 0.5, interpolation=interpolation, align_corners=True)
        self.assert_close(out, expected, rtol=0.0, atol=0.0)


class TestUnbatchedTransforms(BaseTester):
    @pytest.mark.parametrize("operation", ["rotate", "Rotate", "scale", "Scale", "rotate3d", "Affine"])
    @pytest.mark.parametrize("channels", [1, 3])
    @pytest.mark.parametrize("custom_center", [False, True])
    def test_matches_singleton_batch(self, device, dtype, operation, channels, custom_center):
        shape = (channels, 4, 5, 7) if operation == "rotate3d" else (channels, 5, 7)
        image = torch.arange(math.prod(shape), device=device, dtype=dtype).reshape(shape) / math.prod(shape)
        image.requires_grad_()
        value = 1.2 if operation in ("scale", "Scale") else 23.0
        parameter = torch.tensor([value], device=device, dtype=dtype, requires_grad=True)
        center = None
        if custom_center:
            dimensions = 3 if operation == "rotate3d" else 2
            center = torch.full((1, dimensions), 1.3, device=device, dtype=dtype, requires_grad=True)
        if operation == "rotate3d":

            def fn(x: torch.Tensor) -> torch.Tensor:
                return kornia.geometry.transform.rotate3d(x, parameter, parameter, parameter, center)

        elif operation in ("scale", "Scale"):
            fn = kornia.geometry.transform.Scale(parameter, center)
            if operation == "scale":

                def fn(x: torch.Tensor) -> torch.Tensor:
                    return kornia.geometry.transform.scale(x, parameter, center)

        elif operation == "Affine":
            fn = kornia.geometry.transform.Affine(angle=parameter, center=center)

        else:
            fn = kornia.geometry.transform.Rotate(parameter, center)
            if operation == "rotate":

                def fn(x: torch.Tensor) -> torch.Tensor:
                    return kornia.geometry.transform.rotate(x, parameter, center)

        actual = fn(image)
        expected = fn(image.unsqueeze(0)).squeeze(0)

        assert actual.shape == image.shape
        self.assert_close(actual, expected)
        inputs = (image, parameter) if center is None else (image, parameter, center)
        actual_grads = torch.autograd.grad(actual.sum(), inputs)
        expected_grads = torch.autograd.grad(expected.sum(), inputs)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            self.assert_close(actual_grad, expected_grad)


class TestRotate(BaseTester):
    @pytest.mark.parametrize("align_corners", [False, True])
    @pytest.mark.parametrize("center_shape", [None, (2,), (1, 2), (3, 2)])
    @pytest.mark.parametrize("train_transform", [False, True])
    def test_shared_angle_values_and_gradients(self, device, dtype, align_corners, center_shape, train_transform):
        image = torch.rand(3, 2, 5, 7, device=device, dtype=dtype, requires_grad=True)
        angle = torch.tensor([23.0], device=device, dtype=dtype, requires_grad=train_transform)
        center = None
        if center_shape is not None:
            center = torch.full(center_shape, 1.3, device=device, dtype=dtype, requires_grad=train_transform)
        actual = kornia.geometry.transform.rotate(image, angle, center, align_corners=align_corners)
        expanded_center = None if center is None else center.expand(3, -1)
        expected = kornia.geometry.transform.rotate(
            image, angle.expand(3), expanded_center, align_corners=align_corners
        )
        self.assert_close(actual, expected)
        inputs = tuple(t for t in (image, angle, center) if t is not None and t.requires_grad)
        weights = torch.rand_like(actual)
        actual_grads = torch.autograd.grad(actual, inputs, weights)
        expected_grads = torch.autograd.grad(expected, inputs, weights)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            self.assert_close(actual_grad, expected_grad)

    def test_independent_equal_angles_gradients(self, device, dtype):
        image = torch.rand(3, 1, 5, 7, device=device, dtype=dtype)
        angle = torch.full((3,), 23.0, device=device, dtype=dtype, requires_grad=True)
        actual = kornia.geometry.transform.rotate(image, angle)
        expected = torch.cat([kornia.geometry.transform.rotate(image[i : i + 1], angle[i : i + 1]) for i in range(3)])
        self.assert_close(actual, expected)
        self.assert_close(torch.autograd.grad(actual.sum(), angle)[0], torch.autograd.grad(expected.sum(), angle)[0])

    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    def test_shared_angle_export(self, device, dtype):
        image = torch.rand(3, 2, 5, 7, device=device, dtype=dtype)
        angle = torch.tensor([23.0], device=device, dtype=dtype)
        module = kornia.geometry.transform.Rotate(angle)
        exported = torch.export.export(module, (image,)).module()
        self.assert_close(exported(image), module(image))

    def test_shared_angle_trace(self, device, dtype):
        image = torch.rand(3, 2, 5, 7, device=device, dtype=dtype)
        angle = torch.tensor([23.0], device=device, dtype=dtype)
        traced = torch.jit.trace(kornia.geometry.transform.rotate, (image, angle), check_trace=False)
        if dtype in (torch.float16, torch.bfloat16):
            # Tracing uses affine_grid rather than the eager grid expression. Their rounding already
            # differs on CUDA half on the baseline, so compare with the same traced per-image path.
            def expanded_rotation(image, angle):
                return kornia.geometry.transform.rotate(image, angle.expand(image.shape[0]))

            reference = torch.jit.trace(expanded_rotation, (image, angle), check_trace=False)
            expected = reference(image, angle)
        else:
            expected = kornia.geometry.transform.rotate(image, angle)
        self.assert_close(traced(image, angle), expected)

    def test_trace_generalizes_over_image_and_angle_batches(self, device, dtype):
        image = torch.rand(1, 1, 5, 7, device=device, dtype=dtype)
        angle = torch.tensor([23.0], device=device, dtype=dtype)
        traced = torch.jit.trace(kornia.geometry.transform.rotate, (image, angle), check_trace=False)
        image = torch.rand(4, 1, 5, 7, device=device, dtype=dtype)
        angle = torch.tensor([0.0, 10.0, 20.0, 30.0], device=device, dtype=dtype)
        if dtype in (torch.float16, torch.bfloat16):
            # See test_shared_angle_trace: compare two traced affine_grid paths at reduced precision.
            def expanded_rotation(image, angle):
                return kornia.geometry.transform.rotate(image, angle.expand(image.shape[0]))

            expected = torch.jit.trace(expanded_rotation, (image[:1], angle[:1]), check_trace=False)(image, angle)
        else:
            expected = kornia.geometry.transform.rotate(image, angle)
        self.assert_close(traced(image, angle), expected)

    @pytest.mark.parametrize("op", ["rotate", "Rotate", "affine"])
    def test_shared_trace_generalizes_over_batch(self, device, dtype, op):
        # A trace (and the TorchScript ONNX export built on it) recorded at batch 1 must not bake in the batch.
        angle = torch.tensor([23.0], device=device, dtype=dtype)
        matrix = torch.tensor([[[0.9, 0.2, 1.0], [-0.1, 1.1, -0.5]]], device=device, dtype=dtype)
        fns = {
            "rotate": lambda x: kornia.geometry.transform.rotate(x, angle),
            "Rotate": kornia.geometry.transform.Rotate(angle),
            "affine": lambda x: kornia.geometry.transform.affine(x, matrix),
        }
        fn = fns[op]
        traced = torch.jit.trace(fn, torch.rand(1, 2, 5, 7, device=device, dtype=dtype), check_trace=False)
        image = torch.rand(4, 2, 5, 7, device=device, dtype=dtype)
        if dtype in (torch.float16, torch.bfloat16):
            # Tracing builds the grid with affine_grid; compare shapes only, see test_shared_angle_trace.
            assert traced(image).shape == image.shape
        else:
            self.assert_close(traced(image), fn(image))

    def test_angle90(self, device, dtype):
        # prepare input data
        inp = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]], device=device, dtype=dtype)
        expected = torch.tensor([[[0.0, 0.0], [4.0, 6.0], [3.0, 5.0], [0.0, 0.0]]], device=device, dtype=dtype)
        # prepare transformation
        angle = torch.tensor([90.0], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Rotate(angle, align_corners=True)
        self.assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    def test_angle90_batch2(self, device, dtype):
        # prepare input data
        inp = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]], device=device, dtype=dtype).repeat(
            2, 1, 1, 1
        )
        expected = torch.tensor(
            [[[[0.0, 0.0], [4.0, 6.0], [3.0, 5.0], [0.0, 0.0]]], [[[0.0, 0.0], [5.0, 3.0], [6.0, 4.0], [0.0, 0.0]]]],
            device=device,
            dtype=dtype,
        )
        # prepare transformation
        angle = torch.tensor([90.0, -90.0], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Rotate(angle, align_corners=True)
        self.assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    def test_angle90_batch2_broadcast(self, device, dtype):
        # prepare input data
        inp = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]], device=device, dtype=dtype).repeat(
            2, 1, 1, 1
        )
        expected = torch.tensor(
            [[[[0.0, 0.0], [4.0, 6.0], [3.0, 5.0], [0.0, 0.0]]], [[[0.0, 0.0], [4.0, 6.0], [3.0, 5.0], [0.0, 0.0]]]],
            device=device,
            dtype=dtype,
        )
        # prepare transformation
        angle = torch.tensor([90.0], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Rotate(angle, align_corners=True)
        self.assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    def test_convention_align_corners_default_true(self, device, dtype):
        # rotate/Rotate's bare default align_corners=True (every other test in this class passes
        # align_corners=True explicitly to Rotate; this exercises the bare default).
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("hardcoded-literal pin only reliable at float32/float64 precision")
        # Snippet used to generate expected:
        # inp = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
        # angle = torch.tensor([90.0])
        # expected = kornia.geometry.transform.Rotate(angle)(inp)  # no align_corners passed
        inp = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]], device=device, dtype=dtype)
        expected = torch.tensor([[[0.0, 0.0], [4.0, 6.0], [3.0, 5.0], [0.0, 0.0]]], device=device, dtype=dtype)
        angle = torch.tensor([90.0], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Rotate(angle)
        self.assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)
        # the function's own bare default must match (class and function defaults can diverge
        # in this module — see Rescale vs rescale — so both APIs are pinned independently)
        self.assert_close(kornia.geometry.transform.rotate(inp, angle), expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        # test parameters
        angle = torch.tensor([90.0], device=device, dtype=torch.float64)

        # evaluate function gradient
        input = torch.rand(1, 2, 3, 4, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.transform.rotate, (input, angle))

    @pytest.mark.skip("Need deep look into it since crashes everywhere.")
    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):
        angle = torch.tensor([90.0], device=device, dtype=dtype)
        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width, device=device, dtype=dtype)
        rot = kornia.geometry.transform.Rotate(angle)
        rot_traced = torch.jit.trace(kornia.geometry.transform.Rotate(angle), img)
        self.assert_close(rot(img), rot_traced(img))


class TestTranslate(BaseTester):
    def test_dxdy(self, device, dtype):
        # prepare input data
        inp = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]], device=device, dtype=dtype)
        expected = torch.tensor([[[0.0, 1.0], [0.0, 3.0], [0.0, 5.0], [0.0, 7.0]]], device=device, dtype=dtype)
        # prepare transformation
        translation = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Translate(translation, align_corners=True)
        self.assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    def test_dxdy_batch(self, device, dtype):
        # prepare input data
        inp = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]], device=device, dtype=dtype).repeat(
            2, 1, 1, 1
        )
        expected = torch.tensor(
            [[[[0.0, 1.0], [0.0, 3.0], [0.0, 5.0], [0.0, 7.0]]], [[[0.0, 0.0], [0.0, 1.0], [0.0, 3.0], [0.0, 5.0]]]],
            device=device,
            dtype=dtype,
        )
        # prepare transformation
        translation = torch.tensor([[1.0, 0.0], [1.0, 1.0]], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Translate(translation, align_corners=True)
        self.assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    def test_dxdy_batch_broadcast(self, device, dtype):
        # prepare input data
        inp = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]], device=device, dtype=dtype).repeat(
            2, 1, 1, 1
        )
        expected = torch.tensor(
            [[[[0.0, 1.0], [0.0, 3.0], [0.0, 5.0], [0.0, 7.0]]], [[[0.0, 1.0], [0.0, 3.0], [0.0, 5.0], [0.0, 7.0]]]],
            device=device,
            dtype=dtype,
        )
        # prepare transformation
        translation = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Translate(translation, align_corners=True)
        self.assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        # test parameters
        translation = torch.tensor([[1.0, 0.0]], device=device, dtype=torch.float64)

        # evaluate function gradient
        input = torch.rand(1, 2, 3, 4, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.transform.translate, (input, translation), requires_grad=(True, False))

    @pytest.mark.skip("Need deep look into it since crashes everywhere.")
    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):
        translation = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)
        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width, device=device, dtype=dtype)
        trans = kornia.geometry.transform.Translate(translation)
        trans_traced = torch.jit.trace(kornia.geometry.transform.Translate(translation), img)
        self.assert_close(trans(img), trans_traced(img), atol=1e-4, rtol=1e-4)


class TestScale(BaseTester):
    def test_scale_factor_2(self, device, dtype):
        # prepare input data
        inp = torch.tensor(
            [[[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
            device=device,
            dtype=dtype,
        )
        # prepare transformation
        scale_factor = torch.tensor([[2.0, 2.0]], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Scale(scale_factor)
        self.assert_close(transform(inp).sum().item(), 12.25, atol=1e-4, rtol=1e-4)

    def test_scale_factor_05(self, device, dtype):
        # prepare input data
        inp = torch.tensor(
            [[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
            device=device,
            dtype=dtype,
        )
        expected = torch.tensor(
            [[[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
            device=device,
            dtype=dtype,
        )
        # prepare transformation
        scale_factor = torch.tensor([[0.5, 0.5]], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Scale(scale_factor)
        self.assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    def test_scale_factor_05_batch2(self, device, dtype):
        # prepare input data
        inp = torch.tensor(
            [[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)
        expected = torch.tensor(
            [[[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)
        # prepare transformation
        scale_factor = torch.tensor([[0.5, 0.5]], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Scale(scale_factor)
        self.assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    def test_scale_factor_05_batch2_broadcast(self, device, dtype):
        # prepare input data
        inp = torch.tensor(
            [[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)
        expected = torch.tensor(
            [[[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)
        # prepare transformation
        scale_factor = torch.tensor([[0.5, 0.5]], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Scale(scale_factor)
        self.assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size, num_factors", [(1, 1), (2, 2), (3, 3), (3, 1)])
    @pytest.mark.parametrize("as_module", [False, True])
    def test_isotropic_scale_factor(self, device, dtype, batch_size, num_factors, as_module):
        image = torch.arange(batch_size * 35, device=device, dtype=dtype).reshape(batch_size, 1, 5, 7) / 35
        scale_factor = torch.tensor([0.6, 1.0, 1.4], device=device, dtype=dtype)[:num_factors]
        # Isotropic factors apply equally to x and y, independently for each image.
        expected = kornia.geometry.transform.scale(image, torch.stack([scale_factor, scale_factor], dim=-1))
        if as_module:
            actual = kornia.geometry.transform.Scale(scale_factor)(image)
        else:
            actual = kornia.geometry.transform.scale(image, scale_factor)

        self.assert_close(actual, expected)

    def test_isotropic_scale_factor_gradcheck(self, device):
        image = torch.arange(48, device=device, dtype=torch.float64).reshape(2, 1, 4, 6) / 48
        scale_factor = torch.tensor([0.8, 1.3], device=device, dtype=torch.float64)
        # CUDA's grid_sample backward accumulates with atomics, so repeated backward passes differ by roundoff.
        self.gradcheck(kornia.geometry.transform.scale, (image, scale_factor), nondet_tol=1e-8)

    def test_gradcheck(self, device):
        # test parameters
        scale_factor = torch.tensor([[0.5, 0.5]], device=device, dtype=torch.float64)

        # evaluate function gradient
        input = torch.rand(1, 2, 3, 4, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.transform.scale, (input, scale_factor), requires_grad=(True, False))

    @pytest.mark.skip("Need deep look into it since crashes everywhere.")
    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):
        scale_factor = torch.tensor([[0.5, 0.5]], device=device, dtype=dtype)
        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width, device=device, dtype=dtype)
        trans = kornia.geometry.transform.Scale(scale_factor)
        trans_traced = torch.jit.trace(kornia.Scale(scale_factor), img)
        self.assert_close(trans(img), trans_traced(img), atol=1e-4, rtol=1e-4)


class TestShear(BaseTester):
    def test_shear_x(self, device, dtype):
        # prepare input data
        inp = torch.tensor(
            [[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
            device=device,
            dtype=dtype,
        )
        # x-shear by 0.5 about the origin: dst[x, y] samples src[x - 0.5 * y, y], so row y=1
        # samples x=-0.5 (half outside, zero padding -> 0.5), row y=2 samples x=-1 (fully
        # outside -> 0), row y=3 samples x=-1.5 and x=-0.5 -> 0, 0.5.
        # Snippet used to generate expected (identical to the call under test):
        #   inp = torch.ones(1, 4, 4); shear = torch.tensor([[0.5, 0.0]])
        #   expected = kornia.geometry.transform.Shear(shear, align_corners=False)(inp)
        # Pre-#3945 literal from the same snippet, when warp_affine(align_corners=False) still
        # normalized against a corner-aligned grid (the 0.75 / 0.25 are that sub-pixel artifact):
        #   [[[0.75, 1.0, 1.0, 1.0], [0.25, 1.0, 1.0, 1.0], [0.0, 0.75, 1.0, 1.0], [0.0, 0.25, 1.0, 1.0]]]
        expected = torch.tensor(
            [[[1.0, 1.0, 1.0, 1.0], [0.5, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [0.0, 0.5, 1.0, 1.0]]],
            device=device,
            dtype=dtype,
        )

        # prepare transformation
        shear = torch.tensor([[0.5, 0.0]], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Shear(shear, align_corners=False)
        self.assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    def test_shear_y(self, device, dtype):
        # prepare input data
        inp = torch.tensor(
            [[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
            device=device,
            dtype=dtype,
        )
        # y-shear by 0.5 about the origin: the transpose of the test_shear_x case above.
        # Pre-#3945 literal (shear = [[0.0, 0.5]], otherwise the test_shear_x snippet):
        #   [[[0.75, 0.25, 0.0, 0.0], [1.0, 1.0, 0.75, 0.25], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]]
        expected = torch.tensor(
            [[[1.0, 0.5, 0.0, 0.0], [1.0, 1.0, 1.0, 0.5], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
            device=device,
            dtype=dtype,
        )

        # prepare transformation
        shear = torch.tensor([[0.0, 0.5]], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Shear(shear, align_corners=False)
        self.assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    def test_shear_batch2(self, device, dtype):
        # prepare input data
        inp = torch.tensor(
            [[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)

        # the test_shear_x and test_shear_y expectations, stacked; the pre-#3945 literals
        # recorded in those two tests stack the same way
        expected = torch.tensor(
            [
                [[[1.0, 1.0, 1.0, 1.0], [0.5, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [0.0, 0.5, 1.0, 1.0]]],
                [[[1.0, 0.5, 0.0, 0.0], [1.0, 1.0, 1.0, 0.5], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
            ],
            device=device,
            dtype=dtype,
        )

        # prepare transformation
        shear = torch.tensor([[0.5, 0.0], [0.0, 0.5]], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Shear(shear, align_corners=False)
        self.assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    def test_shear_batch2_broadcast(self, device, dtype):
        # prepare input data
        inp = torch.tensor(
            [[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)

        # the test_shear_x expectation broadcast over the batch (pre-#3945 literal recorded there)
        expected = torch.tensor(
            [[[[1.0, 1.0, 1.0, 1.0], [0.5, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [0.0, 0.5, 1.0, 1.0]]]],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1)

        # prepare transformation
        shear = torch.tensor([[0.5, 0.0]], device=device, dtype=dtype)
        transform = kornia.geometry.transform.Shear(shear, align_corners=False)
        self.assert_close(transform(inp), expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        # test parameters
        shear = torch.tensor([[0.5, 0.0]], device=device, dtype=torch.float64)

        # evaluate function gradient
        input = torch.rand(1, 2, 3, 4, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.transform.shear, (input, shear), requires_grad=(True, False))

    @pytest.mark.skip("Need deep look into it since crashes everywhere.")
    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):
        shear = torch.tensor([[0.5, 0.0]], device=device, dtype=dtype)
        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width, device=device, dtype=dtype)
        trans = kornia.geometry.transform.Shear(shear, align_corners=False)
        trans_traced = torch.jit.trace(kornia.geometry.transform.Shear(shear), img)
        self.assert_close(trans(img), trans_traced(img), atol=1e-4, rtol=1e-4)


class TestAffine2d(BaseTester):
    @pytest.mark.parametrize("image_batch,matrix_batch", [(3, 1), (1, 3), (0, 1)])
    def test_function_batch_broadcast(self, device, dtype, image_batch, matrix_batch):
        image = torch.rand(image_batch, 2, 5, 7, device=device, dtype=dtype)
        matrix = torch.tensor([[1.0, 0.1, 0.3], [0.2, 1.0, -0.4]], device=device, dtype=dtype)
        matrix = matrix.expand(matrix_batch, -1, -1)
        batch = 0 if image_batch == 0 else max(image_batch, matrix_batch)
        expected = kornia.geometry.transform.warp_affine(
            image.expand(batch, -1, -1, -1), matrix.expand(batch, -1, -1), (5, 7)
        )
        self.assert_close(kornia.geometry.transform.affine(image, matrix), expected)

    def test_function_invalid_batch(self, device, dtype):
        image = torch.rand(3, 2, 5, 7, device=device, dtype=dtype)
        matrix = torch.eye(2, 3, device=device, dtype=dtype).expand(2, -1, -1)
        with pytest.raises(RuntimeError):
            kornia.geometry.transform.affine(image, matrix)

    def test_affine_no_args(self):
        with pytest.raises(RuntimeError):
            kornia.geometry.transform.Affine()

    def test_affine_batch_size_mismatch(self, device, dtype):
        angle = torch.rand(1, device=device, dtype=dtype)
        translation = torch.rand(2, 2, device=device, dtype=dtype)
        with pytest.raises(RuntimeError):
            kornia.geometry.transform.Affine(angle, translation)

    def test_affine_rotate(self, device, dtype):
        torch.manual_seed(0)
        angle = torch.rand(1, device=device, dtype=dtype) * 90.0
        input = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        transform = kornia.geometry.transform.Affine(angle=angle).to(device=device, dtype=dtype)
        actual = transform(input)
        expected = kornia.geometry.transform.rotate(input, angle)
        self.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_affine_translate(self, device, dtype):
        torch.manual_seed(0)
        translation = torch.rand(1, 2, device=device, dtype=dtype) * 2.0
        input = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        transform = kornia.geometry.transform.Affine(translation=translation).to(device=device, dtype=dtype)
        actual = transform(input)
        expected = kornia.geometry.transform.translate(input, translation)
        self.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_affine_scale(self, device, dtype):
        torch.manual_seed(0)
        _scale_factor = torch.rand(1, device=device, dtype=dtype) * 2.0
        scale_factor = torch.stack([_scale_factor, _scale_factor], dim=1)
        input = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        transform = kornia.geometry.transform.Affine(scale_factor=scale_factor).to(device=device, dtype=dtype)
        actual = transform(input)
        expected = kornia.geometry.transform.scale(input, scale_factor)
        self.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_affine_shear_only_5662(self, device, dtype, batch_size):
        shear = torch.tensor([[0.1, -0.2]], device=device, dtype=dtype).repeat(batch_size, 1)
        image = torch.arange(batch_size * 30, device=device, dtype=dtype).reshape(batch_size, 1, 5, 6) / 30
        # Explicit identity components define the same shear without relying on constructor defaults.
        expected = kornia.geometry.transform.Affine(
            angle=torch.zeros(batch_size, device=device, dtype=dtype),
            translation=torch.zeros(batch_size, 2, device=device, dtype=dtype),
            scale_factor=torch.ones(batch_size, 2, device=device, dtype=dtype),
            shear=shear,
        )(image)
        actual = kornia.geometry.transform.Affine(shear=shear)(image)
        self.assert_close(actual, expected)

    def test_affine_shear_checked_with_other_parameters_5662(self, device, dtype):
        # shear takes part in the constructor's device and dtype check, as angle, translation and scale_factor do.
        shear = torch.tensor([[0.1, -0.2]], device=device, dtype=dtype)
        other = torch.float16 if dtype != torch.float16 else torch.float32  # float64 is unavailable on MPS
        with pytest.raises(TypeCheckError):
            kornia.geometry.transform.Affine(angle=torch.zeros(1, device=device, dtype=other), shear=shear)
        with pytest.raises(DeviceError):
            kornia.geometry.transform.Affine(angle=torch.zeros(1, device="meta", dtype=dtype), shear=shear)

    @staticmethod
    def _affine_parameters(kind, batch_size, device, dtype):
        torch.manual_seed(0)
        if kind == "angle":
            value = 60.0 * torch.rand(batch_size)
        elif kind == "scale_factor":
            value = 1.0 + 0.3 * torch.rand(batch_size, 2)
        else:  # translation in pixels, shear in radians
            value = 0.3 * torch.rand(batch_size, 2)
        return {kind: value.to(device, dtype)}

    @pytest.mark.parametrize("kind", ["angle", "translation", "scale_factor", "shear"])
    @pytest.mark.parametrize("with_center", [False, True])
    def test_affine_batch1_parameters_warp_every_image_6395(self, device, dtype, kind, with_center):
        # One parameter set warps every image of a batch, with or without center, as Rotate and Translate do.
        params = self._affine_parameters(kind, 1, device, dtype)
        center = torch.tensor([[3.0, 2.0]], device=device, dtype=dtype) if with_center else None
        torch.manual_seed(1)
        images = torch.rand(3, 2, 5, 7).to(device, dtype)

        transform = kornia.geometry.transform.Affine(**params, center=center)
        expected = torch.cat([transform(image[None]) for image in images])
        self.assert_close(transform(images), expected)

    @pytest.mark.parametrize("unbatched", [False, True])
    def test_affine_parameter_sets_warp_one_image_without_center_6395(self, device, dtype, unbatched):
        # Without center, B parameter sets warp one image into (B, C, H, W), as they do with the image centre given.
        params = self._affine_parameters("angle", 3, device, dtype)
        params.update(self._affine_parameters("translation", 3, device, dtype))
        torch.manual_seed(1)
        image = torch.rand(1, 2, 5, 7).to(device, dtype)
        image = image[0] if unbatched else image
        image_centre = torch.tensor([[3.0, 2.0]], device=device, dtype=dtype).expand(3, -1)

        actual = kornia.geometry.transform.Affine(**params)(image)
        expected = kornia.geometry.transform.Affine(**params, center=image_centre)(image)
        assert actual.shape == (3, 2, 5, 7)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("with_center", [False, True])
    def test_affine_parameter_and_input_batch_mismatch_raises_6395(self, device, dtype, with_center):
        # Two parameter sets for three images pair in no sensible way: the error names both batch sizes.
        params = self._affine_parameters("angle", 2, device, dtype)
        center = torch.tensor([[3.0, 2.0]], device=device, dtype=dtype).expand(2, -1) if with_center else None
        images = torch.zeros(3, 2, 5, 7, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="Affine has 2 parameter sets for an input batch of 3"):
            kornia.geometry.transform.Affine(**params, center=center)(images)

    @pytest.mark.parametrize("as_module", [False, True])
    def test_convention_shear_raw_factors_5661(self, device, dtype, as_module):
        # Both off-diagonal factors act on the original coordinates, about pixel (0, 0).
        shear = torch.tensor([[0.5, 0.25]], device=device, dtype=dtype)
        image = torch.arange(81, device=device, dtype=dtype).reshape(1, 1, 9, 9) / 80
        matrix = torch.tensor([[[1.0, 0.5, 0.0], [0.25, 1.0, 0.0]]], device=device, dtype=dtype)
        expected = kornia.geometry.transform.warp_affine(image, matrix, (9, 9), align_corners=True)
        if as_module:
            actual = kornia.geometry.transform.Shear(shear, align_corners=True)(image)
        else:
            actual = kornia.geometry.transform.shear(image, shear, align_corners=True)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("custom_center", [False, True])
    def test_convention_affine_shear_radians_5661(self, device, dtype, custom_center):
        # tan(sx) = 1 and tan(sy) = 2: x' = x - (y - cy), then y' = y - 2 (x' - cx).
        shear = torch.tensor([[math.pi / 4, math.atan(2.0)]], device=device, dtype=dtype)
        center = torch.tensor([[2.0, 3.0] if custom_center else [4.0, 4.0]], device=device, dtype=dtype)
        cx, cy = (2.0, 3.0) if custom_center else (4.0, 4.0)
        matrix = torch.tensor(
            [[[1.0, -1.0, cy], [-2.0, 3.0, 2.0 * (cx - cy)], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )
        angle = torch.zeros(1, device=device, dtype=dtype)
        translation = torch.zeros(1, 2, device=device, dtype=dtype)
        scale = torch.ones(1, 2, device=device, dtype=dtype)
        # Float16 rounds the angles before tan (versus errors >= 1 for a wrong order or swapped angles).
        tolerance = {"atol": 5e-3, "rtol": 1e-3} if dtype == torch.float16 else {}
        self.assert_close(
            kornia.geometry.transform.get_shear_matrix2d(center, shear[:, 0], shear[:, 1]), matrix, **tolerance
        )
        self.assert_close(
            kornia.geometry.transform.get_affine_matrix2d(translation, center, scale, angle, shear[:, 0], shear[:, 1]),
            matrix,
            **tolerance,
        )
        image = torch.zeros(1, 1, 9, 9, device=device, dtype=dtype)
        image[0, 0, 6, 5] = 1
        expected = torch.zeros_like(image)
        row, col = (6, 2) if custom_center else (8, 3)
        expected[0, 0, row, col] = 1
        transform = kornia.geometry.transform.Affine(
            shear=shear, center=center if custom_center else None, mode="nearest", align_corners=True
        )
        self.assert_close(transform(image), expected)

    def test_wart_shear_and_affine_shear_the_other_way_5661(self, device, dtype):
        # One pixel at row 8, column 4. shear() reads 0.5 as a factor about pixel (0, 0) and moves it to column
        # 4 + 0.5 * 8 = 8; Affine reads it as an angle about the centre and moves it to 4 - tan(0.5) * (8 - 4) = 1.8.
        shear = torch.tensor([[0.5, 0.0]], device=device, dtype=dtype)
        image = torch.zeros(1, 1, 9, 9, device=device, dtype=dtype)
        image[0, 0, 8, 4] = 1
        sheared = kornia.geometry.transform.shear(image, shear, mode="nearest", align_corners=True)
        affine = kornia.geometry.transform.Affine(shear=shear, mode="nearest", align_corners=True)
        assert sheared[0, 0, 8].argmax().item() == 8
        assert affine(image)[0, 0, 8].argmax().item() == 2

    def test_affine_rotate_translate(self, device, dtype):
        batch_size = 2

        input = torch.tensor(
            [[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
            device=device,
            dtype=dtype,
        ).repeat(batch_size, 1, 1, 1)

        angle = torch.tensor(180.0, device=device, dtype=dtype).repeat(batch_size)
        translation = torch.tensor([1.0, 0.0], device=device, dtype=dtype).repeat(batch_size, 1)

        expected = torch.tensor(
            [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
            device=device,
            dtype=dtype,
        ).repeat(batch_size, 1, 1, 1)

        transform = kornia.geometry.transform.Affine(angle=angle, translation=translation, align_corners=True).to(
            device=device, dtype=dtype
        )
        actual = transform(input)
        # The half dtypes round the 180-degree rotation and the bilinear weights: the output is off by up to 3.9e-3
        # in float16 and 2.3e-2 in bfloat16 on CPU and CUDA, against 1.0 for a dropped translation.
        tol = {torch.float16: 1e-2, torch.bfloat16: 5e-2}.get(dtype, 1e-4)
        self.assert_close(actual, expected, atol=tol, rtol=tol)

    def test_compose_affine_matrix_3x3(self, device, dtype):
        """To get parameters:
        import torchvision as tv
        from PIL import Image
        from torch import Tensor as T
        import math
        import random
        img_size = (96,96)
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.manual_seed(seed)
        tfm = tv.transforms.RandomAffine(degrees=(-25.0,25.0),
                                        scale=(0.6, 1.4) ,
                                        translate=(0, 0.1),
                                        shear=(-25., 25., -20., 20.))
        angle, translations, scale, shear = tfm.get_params(tfm.degrees, tfm.translate,
                                                        tfm.scale, tfm.shear, img_size)
        print (angle, translations, scale, shear)
        output_size = img_size
        center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)

        matrix = tv.transforms.functional._get_inverse_affine_matrix(center, angle, translations, scale, shear)
        matrix = np.array(matrix).reshape(2,3)
        print (matrix)
        """
        import math

        from torch import Tensor as T

        batch_size, _, height, width = 1, 1, 96, 96
        angle, translations = 6.971339922894188, (0.0, -4.0)
        scale, shear = [0.7785685905190581, 0.7785685905190581], [11.8235607082617, 7.06797949691645]
        matrix_expected = T([[1.27536969, 4.26828945e-01, -3.2349e01], [2.18297196e-03, 1.29424165e00, -9.1996e00]])
        center = T([float(width), float(height)]).view(1, 2) / 2.0 + 0.5
        center = center.expand(batch_size, -1)
        matrix_kornia = kornia.geometry.transform.get_affine_matrix2d(
            T(translations).view(-1, 2),
            center,
            T([scale]).view(-1, 2),
            T([angle]).view(-1),
            T([math.radians(shear[0])]).view(-1, 1),
            T([math.radians(shear[1])]).view(-1, 1),
        )
        matrix_kornia = matrix_kornia.inverse()[0, :2].detach().cpu()
        self.assert_close(matrix_kornia, matrix_expected, atol=1e-4, rtol=1e-4)

    def test_broadcasting_issue_3176(self, device, dtype):
        # Issue 3176: RandomRotation with multi-channel masks caused crash
        # Scenario: Tensor is (B=1, C, H, W) but Matrix is (B=8, 2, 3)
        # This implies applying N different transformations to a single image
        B_mat, B_ten = 8, 1
        C, H, W = 3, 64, 64

        input_tensor = torch.rand(B_ten, C, H, W, device=device, dtype=dtype)

        matrix = torch.rand(B_mat, 2, 3, device=device, dtype=dtype)

        output = kornia.geometry.transform.affine(input_tensor, matrix, mode="bilinear")

        assert output.shape == (B_mat, C, H, W)

    def test_warp_affine_fill_value(self, device, dtype):
        # Feature: Support padding_mode="fill" with custom fill_value
        # Scenario: 1-channel mask, fill with 1.0
        B, C, H, W = 1, 1, 10, 10
        src = torch.zeros(B, C, H, W, device=device, dtype=dtype)

        M = torch.eye(2, 3, device=device, dtype=dtype).unsqueeze(0)
        M[..., 0, 2] = 5.0

        # Fill with 1.0
        fill_val = torch.tensor([1.0], device=device, dtype=dtype)

        out = kornia.geometry.transform.warp_affine(src, M, (H, W), padding_mode="fill", fill_value=fill_val)

        assert out[0, 0, 0, 0] == 1.0

        assert out[0, 0, 0, 9] == 0.0


class TestGetAffineMatrix(BaseTester):
    def test_smoke(self, device, dtype):
        H, W = 5, 5
        translation = torch.tensor([[0.0, 0.0]], device=device, dtype=dtype)
        # NOTE: ideally the center should be [W * 0.5, H * 0.5]
        center = torch.tensor([[W // 2, H // 2]], device=device, dtype=dtype)
        zoom1 = torch.ones([1, 1], device=device, dtype=dtype) * 0.5
        zoom2 = torch.ones([1, 1], device=device, dtype=dtype) * 1.0
        zoom = torch.cat([zoom1, zoom2], -1)
        angle = torch.zeros([1], device=device, dtype=dtype)
        affine_mat = kornia.geometry.get_affine_matrix2d(translation, center, zoom, angle)

        img = torch.ones(1, 1, H, W, device=device, dtype=dtype)
        expected = torch.zeros_like(img)
        expected[..., 1:4] = 1.0

        out = kornia.geometry.transform.warp_affine(img, affine_mat[:, :2], (H, W))
        self.assert_close(out, expected)


class TestGetShearMatrix(BaseTester):
    def test_get_shear_matrix2d_with_controlled_values(self, device, dtype):
        # Define controlled values for shear angles and center
        sx = torch.tensor([0.5], device=device, dtype=dtype)
        sy = torch.tensor([0.25], device=device, dtype=dtype)
        center = torch.tensor([[0.0, 0.0]], device=device, dtype=dtype)

        # Calculate the shear matrix using your function
        out = kornia.geometry.transform.get_shear_matrix2d(center, sx=sx, sy=sy)

        # Define the expected shear matrix with controlled numbers
        expected = torch.tensor(
            [[[1.0, -0.5463, 0.0], [-0.2553, 1.1395, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )

        self.assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_get_shear_matrix2d_default_params_device_dtype(self, device, dtype):
        # When sx and sy are None, the defaults should inherit device and dtype from center.
        center = torch.tensor([[128.0, 128.0]], device=device, dtype=dtype)
        out = kornia.geometry.transform.get_shear_matrix2d(center, sx=None, sy=None)
        assert out.device.type == center.device.type, "Output device must match center device"
        assert out.dtype == center.dtype, "Output dtype must match center dtype"

    def test_get_shear_matrix3d_default_params_device_dtype(self, device, dtype):
        # When shear params are None, the defaults should inherit device and dtype from center.
        center = torch.tensor([[64.0, 64.0, 32.0]], device=device, dtype=dtype)
        out = kornia.geometry.transform.get_shear_matrix3d(
            center, sxy=None, sxz=None, syx=None, syz=None, szx=None, szy=None
        )
        assert out.device.type == center.device.type, "Output device must match center device"
        assert out.dtype == center.dtype, "Output dtype must match center dtype"

    def test_get_shear_matrix3d_keeps_center_for_every_shear_subset(self, device, dtype):
        # The translation used to come from the positive-tangent matrix while the returned linear part uses the
        # negated tangents, so shears that meet in a product term moved the center (#6302). The center is asymmetric
        # so that no coordinate can stand in for another.
        names = ("sxy", "sxz", "syx", "syz", "szx", "szy")
        values = (0.3, -0.25, 0.2, 0.15, -0.35, 0.1)
        center = torch.tensor([[7.0, 23.0, 41.0]], device=device, dtype=dtype)
        center_h = torch.tensor([7.0, 23.0, 41.0, 1.0], device=device, dtype=dtype)
        # bfloat16 rounds near 41 in steps of 0.25, while the smallest displacement the bug caused here is 1.24 px.
        tol = 0.5 if dtype in (torch.float16, torch.bfloat16) else None
        for size in range(1, 7):
            for subset in itertools.combinations(range(6), size):
                shears = {names[i]: torch.tensor([values[i]], device=device, dtype=dtype) for i in subset}
                out = kornia.geometry.transform.get_shear_matrix3d(center, **shears)
                self.assert_close(out[0] @ center_h, center_h, rtol=tol and 0.0, atol=tol)


class TestTransformModuleBuffers(BaseTester):
    @pytest.mark.parametrize(
        "module_name,with_center",
        [(name, center) for name in ("Rotate", "Scale", "Affine") for center in (False, True)]
        + [("Translate", False), ("Shear", False)],
    )
    def test_module_to(self, device, dtype, module_name, with_center):
        initial_dtype = torch.float64 if dtype == torch.float32 else torch.float32
        values = {
            "Rotate": {"angle": [23.0]},
            "Scale": {"scale_factor": [[0.8, 1.2]]},
            "Affine": {
                "angle": [23.0],
                "translation": [[0.4, -0.2]],
                "scale_factor": [[0.8, 1.2]],
                "shear": [[0.1, -0.05]],
            },
            "Translate": {"translation": [[0.4, -0.2]]},
            "Shear": {"shear": [[0.1, -0.05]]},
        }[module_name]
        if with_center and module_name in ("Rotate", "Scale", "Affine"):
            values["center"] = [[1.5, 2.0]]
        kwargs = {name: torch.tensor(value, dtype=initial_dtype) for name, value in values.items()}
        constructor = getattr(kornia.geometry.transform, module_name)
        transform = constructor(**kwargs).to(device=device, dtype=dtype)
        # Reference constructs directly on the destination, without calling Module.to.
        expected_transform = constructor(
            **{name: value.to(device=device, dtype=dtype) for name, value in kwargs.items()}
        )
        image = torch.arange(30, device=device, dtype=dtype).reshape(1, 1, 5, 6) / 30
        self.assert_close(transform(image), expected_transform(image))
        for name in kwargs:
            assert getattr(transform, name).device == image.device
            assert getattr(transform, name).dtype == image.dtype
        # These fixed constructor values have never been trainable parameters or checkpoint entries.
        assert not list(transform.parameters())
        assert not transform.state_dict()

    def test_constructor_gradient_after_to(self, device, dtype):
        angle = torch.tensor([17.0], requires_grad=True)
        transform = kornia.geometry.transform.Rotate(angle).to(device=device, dtype=dtype)
        image = torch.arange(30, device=device, dtype=dtype).reshape(1, 1, 5, 6) / 30
        weights = image.flip(-1)
        actual = torch.autograd.grad((transform(image) * weights).sum(), angle)[0]
        reference_angle = angle.detach().to(device=device, dtype=dtype).requires_grad_()
        expected = torch.autograd.grad(
            (kornia.geometry.transform.rotate(image, reference_angle) * weights).sum(), reference_angle
        )[0]
        self.assert_close(actual.to(expected), expected)

    @pytest.mark.parametrize(
        "module_name,name,value",
        [
            ("Rotate", "angle", [17.0]),
            ("Scale", "scale_factor", [[0.8, 1.2]]),
            ("Affine", "angle", [17.0]),
            ("Affine", "shear", [[0.1, -0.05]]),
            ("Translate", "translation", [[0.4, -0.2]]),
            ("Shear", "shear", [[0.1, -0.05]]),
        ],
    )
    def test_trainable_constructor_parameter(self, device, dtype, module_name, name, value):
        parameter = torch.nn.Parameter(torch.tensor(value))
        transform = getattr(kornia.geometry.transform, module_name)(**{name: parameter})
        transform = transform.to(device=device, dtype=dtype)
        stored = getattr(transform, name)
        assert isinstance(stored, torch.nn.Parameter)
        assert dict(transform.named_parameters())[name] is stored
        assert set(transform.state_dict()) == {name}
        image = torch.arange(30, device=device, dtype=dtype).reshape(1, 1, 5, 6) / 30
        gradient = torch.autograd.grad((transform(image) * image.flip(-1)).sum(), stored)[0]
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
