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
from __future__ import annotations

import re
from functools import partial
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from kornia.core._compat import torch_version
from kornia.core.exceptions import BaseError, TypeCheckError
from kornia.filters import GuidedBlur, box_blur, guided_blur

from testing.base import BaseTester, supports_reflect_padding, supports_replicate_padding

# The guided filter is ill-conditioned in half precision: the window variance is E[g^2] - E[g]^2, a cancellation,
# and the coefficient cov / (var + eps) amplifies its error by up to 1 / eps. Worst absolute error against float64
# over 150 seeds: 0.008 (float16) and 0.092 (bfloat16) at eps=0.1, 0.0013 and 0.0102 at eps=1.0, 42 (bfloat16) at
# eps=0.01; float32 2.4e-7 at eps=1.0. So the half dtypes use eps=1.0 and a tolerance a few times that error on
# non-divisible sizes, float32 and float64 eps=0.1 and the harness's. bfloat16's 0.05 still catches only gross errors
# (an align_corners=True mutant slips through most of them); float16's 0.01 catches it.
_HALF_TOLERANCE = {torch.float16: 0.01, torch.bfloat16: 0.05}


def _value_eps(dtype):
    return 1.0 if dtype in _HALF_TOLERANCE else 0.1


def _fast_guided_filter_reference(guidance, input, kernel_size, eps, subsample, border_type="reflect"):
    """He and Sun, "Fast Guided Filter" (arXiv:1505.00996), Algorithm 1, per pixel in float64 on the CPU.

    The coefficient maps ``a`` (``C x C_in`` per pixel) and ``b`` are solved on the subsampled grid, window-averaged and
    resized back to the full ``(H, W)`` of the input, which is the contract for any ``H`` and ``W``.
    """
    guidance, input = guidance.cpu().double(), input.cpu().double()
    (_, C, H, W), C_in = guidance.shape, input.shape[1]
    size = (max(H // subsample, 1), max(W // subsample, 1))
    g = F.interpolate(guidance, size=size, mode="nearest")
    p = F.interpolate(input, size=size, mode="nearest")
    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    window = tuple((k - 1) // subsample + 1 for k in kernel_size)

    def mean(x):
        return box_blur(x, window, border_type)

    mean_g, mean_p = mean(g), mean(p)
    corr_gg = mean(torch.einsum("bihw,bjhw->bijhw", g, g).flatten(1, 2)).unflatten(1, (C, C))
    corr_gp = mean(torch.einsum("bihw,bjhw->bijhw", g, p).flatten(1, 2)).unflatten(1, (C, C_in))
    var_gg = corr_gg - torch.einsum("bihw,bjhw->bijhw", mean_g, mean_g)
    cov_gp = corr_gp - torch.einsum("bihw,bjhw->bijhw", mean_g, mean_p)
    a = torch.linalg.solve(
        var_gg.permute(0, 3, 4, 1, 2) + eps * torch.eye(C, dtype=g.dtype), cov_gp.permute(0, 3, 4, 1, 2)
    )
    b = mean_p.permute(0, 2, 3, 1) - torch.einsum("bhwc,bhwcq->bhwq", mean_g.permute(0, 2, 3, 1), a)
    mean_a = mean(a.permute(0, 3, 4, 1, 2).flatten(1, 2))
    mean_b = mean(b.permute(0, 3, 1, 2))
    mean_a = F.interpolate(mean_a, size=(H, W), mode="bilinear").unflatten(1, (C, C_in))
    mean_b = F.interpolate(mean_b, size=(H, W), mode="bilinear")
    return mean_b + torch.einsum("bchw,bcqhw->bqhw", guidance, mean_a)


def _scale_factor_interpolate(x, size=None, scale_factor=None, mode=None, **kwargs):
    """``interpolate`` as the filter called it before ``size=``: with ``scale_factor`` (``1 / s`` down, ``s`` up).

    Stands in for ``kornia.filters.guided.interpolate``. A call that already passes ``scale_factor`` goes through
    untouched; a ``size=`` call is turned into the call that was made before, for the same integer ratio, so the two can
    be compared bit for bit: nearest with ``1 / s`` when it shrinks, bilinear with ``s`` when it grows. The mode of the
    call under test is deliberately not taken over.
    """
    if size is None:
        return F.interpolate(x, scale_factor=scale_factor, mode=mode, **kwargs)
    assert not kwargs
    (h, w), (H, W) = x.shape[-2:], size
    ratio = max(h, H) // min(h, H)
    assert (
        ratio > 1
        and ratio * min(h, H) == max(h, H)
        and ratio * min(w, W) == max(w, W)
        and max(w, W) // min(w, W) == ratio
    )
    if H < h:
        return F.interpolate(x, scale_factor=1 / ratio, mode="nearest")
    return F.interpolate(x, scale_factor=ratio, mode="bilinear")


class TestGuidedBlur(BaseTester):
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("guide_dim", [1, 3])
    @pytest.mark.parametrize("input_dim", [1, 3])
    @pytest.mark.parametrize("kernel_size", [5, (3, 5)])
    def test_smoke(self, batch_size, guide_dim, input_dim, kernel_size, device, dtype):
        H, W = 8, 16
        guide = torch.randn(batch_size, guide_dim, H, W, device=device, dtype=dtype)
        inp = torch.randn(batch_size, input_dim, H, W, device=device, dtype=dtype)

        # tensor eps -> with batch dim
        eps = torch.rand(batch_size, device=device, dtype=dtype)
        actual_A = guided_blur(guide, inp, kernel_size, eps)
        assert isinstance(actual_A, torch.Tensor)
        assert actual_A.shape == (batch_size, input_dim, H, W)

        # float and tuple sigmas -> same sigmas across batch
        eps_ = eps[0].item()
        actual_B = guided_blur(guide, inp, kernel_size, eps_)
        assert isinstance(actual_B, torch.Tensor)
        assert actual_B.shape == (batch_size, input_dim, H, W)

        self.assert_close(actual_A[0], actual_B[0])

        # fast guided filter
        actual_C = guided_blur(guide, inp, kernel_size, eps_, subsample=4)
        assert isinstance(actual_C, torch.Tensor)
        assert actual_C.shape == (batch_size, input_dim, H, W)

        # self-guidance
        actual_D = guided_blur(inp, inp, kernel_size, eps_)
        assert isinstance(actual_D, torch.Tensor)
        assert actual_D.shape == (batch_size, input_dim, H, W)

    @pytest.mark.parametrize("guide_dim", [1, 3])
    @pytest.mark.parametrize("kernel_size", [5, (3, 5)])
    @pytest.mark.parametrize("subsample", [1, 2])
    def test_separable_matches_nonseparable(
        self,
        guide_dim,
        kernel_size,
        subsample,
        device,
        dtype,
    ) -> None:
        height, width = 12, 16
        guide = torch.linspace(
            0.1,
            1.0,
            steps=guide_dim * height * width,
            device=device,
            dtype=dtype,
        ).reshape(1, guide_dim, height, width)
        inp = torch.linspace(
            1.0,
            0.1,
            steps=2 * height * width,
            device=device,
            dtype=dtype,
        ).reshape(1, 2, height, width)

        expected = guided_blur(
            guide,
            inp,
            kernel_size,
            eps=0.1,
            subsample=subsample,
            separable=False,
        )
        actual = guided_blur(
            guide,
            inp,
            kernel_size,
            eps=0.1,
            subsample=subsample,
            separable=True,
        )

        if dtype == torch.float16:
            # The two convolution decompositions accumulate rounding errors in different orders at
            # half precision. Only float16 needs this: measured worst case is 1.953e-03 against the
            # harness's 1e-03, while every bfloat16 case fits inside _DTYPE_PRECISIONS already.
            tolerance = 3 * torch.finfo(dtype).eps
            self.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        else:
            self.assert_close(actual, expected)

    @pytest.mark.parametrize("guide_dim", [1, 3])
    def test_module_forwards_separable_to_all_box_blurs(
        self,
        guide_dim,
        device,
        dtype,
    ) -> None:
        guide = torch.ones(1, guide_dim, 8, 8, device=device, dtype=dtype)
        inp = torch.ones(1, 2, 8, 8, device=device, dtype=dtype)
        received_separable_values = []

        def tracked_box_blur(
            input_tensor: torch.Tensor,
            kernel_size: tuple[int, int] | int,
            border_type: str = "reflect",
            separable: bool = False,
        ) -> torch.Tensor:
            received_separable_values.append(separable)
            return box_blur(input_tensor, kernel_size, border_type, separable=separable)

        with patch(
            "kornia.filters.guided.box_blur",
            side_effect=tracked_box_blur,
        ):
            GuidedBlur(3, 0.1, separable=True)(guide, inp)

        assert received_separable_values
        assert all(received_separable_values), "Expected all box_blur calls to have separable=True"

    @pytest.mark.parametrize("shape", [(1, 1, 8, 15), (2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [5, (3, 5)])
    def test_cardinality(self, shape, kernel_size, device, dtype):
        guide = torch.zeros(shape, device=device, dtype=dtype)
        inp = torch.zeros(shape, device=device, dtype=dtype)
        actual = guided_blur(guide, inp, kernel_size, 0.1)
        assert actual.shape == shape

    def test_exception(self):
        from kornia.core.exceptions import BaseError, TypeCheckError

        with pytest.raises(TypeCheckError) as errinfo:
            guided_blur(torch.rand(1, 1, 5, 5), 3, 3, 0.1)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        with pytest.raises(BaseError) as errinfo:
            guided_blur(torch.rand(1, 1, 5, 5), torch.rand(2, 1, 5, 5), 3, 0.1)
        assert "same batch size and spatial dimensions" in str(errinfo.value)

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        guide = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        actual = guided_blur(guide, inp, 3, 0.1)
        assert actual.is_contiguous()

    @pytest.mark.parametrize("separable", [False, True])
    def test_gradcheck(self, separable, device) -> None:
        guide = torch.rand(1, 2, 5, 4, device=device, dtype=torch.float64)
        img = torch.rand(1, 2, 5, 4, device=device, dtype=torch.float64)
        operation = partial(guided_blur, separable=separable)
        self.gradcheck(operation, (guide, img, 3, 0.1), nondet_tol=1e-4)

        eps = torch.rand(1, device=device, dtype=torch.float64)
        self.gradcheck(operation, (guide, img, 3, eps), nondet_tol=1e-4)

    @pytest.mark.parametrize("shape", [(1, 1, 8, 16), (2, 3, 12, 8)])
    @pytest.mark.parametrize("kernel_size", [5, (3, 5)])
    @pytest.mark.parametrize("eps", [0.1, 0.01])
    @pytest.mark.parametrize("subsample", [1, 2])
    @pytest.mark.parametrize("separable", [False, True])
    def test_module(self, shape, kernel_size, eps, subsample, device, dtype, separable) -> None:
        guide = torch.rand(shape, device=device, dtype=dtype)
        img = torch.rand(shape, device=device, dtype=dtype)

        op = guided_blur
        op_module = GuidedBlur(
            kernel_size,
            eps,
            subsample=subsample,
            separable=separable,
        )
        self.assert_close(
            op_module(guide, img),
            op(guide, img, kernel_size, eps, subsample=subsample, separable=separable),
        )

    @pytest.mark.parametrize("tensor_eps", [False, True], ids=["float_eps", "tensor_eps"])
    def test_multichannel_guidance_in_half_precision(self, tensor_eps, device, dtype) -> None:
        """Multi-channel guidance must run in float16/bfloat16 rather than dying in the solver.

        ``_guided_blur_multichannel_guidance`` solves a C x C system per pixel, and
        ``torch.linalg.solve`` has no half-precision LU kernel: before the fix this raised
        ``NotImplementedError: "lu_cpu" not implemented for 'Half'`` on CPU, and on MPS it aborted
        the process outright with ``Only MPSDataTypeFloat32 is supported``. Single-channel guidance
        never reached the solver, so only ``guide_dim > 1`` was affected.

        The ``tensor_eps`` case is the second half of the same bug: ``torch.tensor(0.1)`` is float32
        even next to a half guidance and takes part in dtype promotion, so the matrix handed to
        ``solve`` is wider than the right-hand side.
        """
        if dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("regression test for the half-precision solve path")

        # ``constant`` keeps this test on the solver: the default ``reflect`` border needs a
        # half-precision ``reflection_pad2d``, which CPU PyTorch 2.5.1 does not have for float16.
        border_type = "constant"
        eps = torch.tensor(0.1, device=device) if tensor_eps else 0.1

        guide = torch.rand(1, 3, 12, 16, device=device, dtype=dtype)
        inp = torch.rand(1, 2, 12, 16, device=device, dtype=dtype)

        actual = guided_blur(guide, inp, 5, eps, border_type=border_type)

        assert actual.dtype == dtype
        assert torch.isfinite(actual).all()

        # The solve runs in float32, so the result must track the exact answer to within the
        # accumulated error of the surrounding half-precision arithmetic, not the solver's.
        expected = guided_blur(guide.float(), inp.float(), 5, eps, border_type=border_type)
        tolerance = 8 * torch.finfo(dtype).eps
        self.assert_close(actual.float(), expected, rtol=tolerance, atol=tolerance)

    @pytest.mark.skipif(
        torch_version() in {"1.9.1", "2.1.0", "2.1.1", "2.1.2"},
        reason=(
            "https://github.com/pytorch/pytorch/issues/110696 "
            "- Failing with: Argument of Integer should be of numeric type, got s3 + 3."
        ),
    )
    @pytest.mark.parametrize("kernel_size", [5, (5, 7)])
    @pytest.mark.parametrize("subsample", [1, 2])
    @pytest.mark.parametrize("separable", [False, True])
    def test_dynamo(self, kernel_size, subsample, separable, device, dtype, torch_optimizer) -> None:
        guide = torch.ones(2, 3, 8, 8, device=device, dtype=dtype)
        data = torch.ones(2, 3, 8, 8, device=device, dtype=dtype)
        op = GuidedBlur(kernel_size, 0.1, subsample=subsample, separable=separable)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(guide, data), op_optimized(guide, data))

        op = GuidedBlur(
            kernel_size,
            torch.tensor(0.1, device=device, dtype=dtype),
            subsample=subsample,
            separable=separable,
        )
        op_optimized = torch_optimizer(op)

        self.assert_close(op(guide, data), op_optimized(guide, data))

    def test_opencv_grayscale(self, device, dtype):
        guide = [[100, 130, 58, 36], [215, 142, 173, 166], [114, 150, 190, 60], [23, 83, 84, 216]]
        guide = torch.tensor(guide, device=device, dtype=dtype).view(1, 1, 4, 4) / 255

        img = [[95, 130, 108, 228], [98, 142, 187, 166], [114, 166, 190, 141], [150, 83, 174, 216]]
        img = torch.tensor(img, device=device, dtype=dtype).view(1, 1, 4, 4) / 255

        kernel_size = 3
        eps = 0.01

        # Expected output generated with OpenCV:
        # import cv2
        # expected = cv2.ximgproc.guidedFilter(
        #   guide.squeeze().numpy(),
        #   img.squeeze().numpy(),
        #   (kernel_size - 1) // 2,
        #   eps,
        # )
        expected = [
            [0.4487294, 0.5163902, 0.5981981, 0.70094436],
            [0.4850059, 0.53724647, 0.62616897, 0.6686147],
            [0.5010369, 0.5631456, 0.6808387, 0.5960593],
            [0.5304646, 0.53203756, 0.57674146, 0.80308396],
        ]
        expected = torch.tensor(expected, device=device, dtype=dtype).view(1, 1, 4, 4)

        # OpenCV uses hard-coded BORDER_REFLECT mode, which also reflects the outermost pixels
        # https://github.com/opencv/opencv_contrib/blob/853144ef93c4ffa55661619b861539090943c5b6/modules/ximgproc/src/guided_filter.cpp#L162
        # PyTorch's `reflect` border type corresponds to OpenCV's BORDER_REFLECT_101
        # To match the border's behavior, we use kernel_size = 3 and border_type="replicate" for testing
        out = guided_blur(guide, img, kernel_size, eps, border_type="replicate")
        self.assert_close(out, expected)

    def test_opencv_rgb(self, device, dtype):
        guide = [
            [[170, 89, 182, 255], [199, 209, 216, 205], [196, 213, 218, 191], [207, 126, 224, 249]],
            [[61, 104, 274, 225], [65, 112, 14, 148], [78, 247, 176, 120], [124, 69, 155, 211]],
            [[73, 111, 94, 175], [77, 117, 123, 130], [83, 139, 163, 120], [132, 84, 137, 155]],
        ]
        guide = torch.tensor(guide, device=device, dtype=dtype).view(1, 3, 4, 4) / 255

        img = [
            [[170, 189, 182, 255], [169, 239, 206, 215], [196, 213, 28, 191], [207, 16, 234, 240]],
            [[61, 144, 74, 225], [20, 112, 176, 148], [34, 147, 116, 120], [124, 61, 155, 211]],
            [[73, 111, 90, 175], [177, 117, 163, 130], [89, 139, 163, 120], [132, 84, 137, 135]],
        ]
        img = torch.tensor(img, device=device, dtype=dtype).view(1, 3, 4, 4) / 255

        kernel_size = 3
        eps = 0.01

        # Expected output generated with OpenCV:
        # import cv2
        # expected = cv2.ximgproc.guidedFilter(
        #   guide.squeeze().permute(1, 2, 0).numpy(),
        #   img.squeeze().permute(1, 2, 0).numpy(),
        #   (kernel_size - 1) // 2,
        #   eps,
        # ).transpose(2, 0, 1)
        expected = [
            [
                [0.7039907, 0.7277061, 0.7474556, 0.904094],
                [0.7095674, 0.76176095, 0.77444744, 0.7774203],
                [0.67807436, 0.7721572, 0.70001286, 0.7042719],
                [0.73099065, 0.28477466, 0.7464762, 0.8454268],
            ],
            [
                [0.25627214, 0.4922768, 0.3593133, 0.76788116],
                [0.21797341, 0.42890117, 0.56577384, 0.58102953],
                [0.25184435, 0.5643642, 0.59704626, 0.5153022],
                [0.42154774, 0.24721909, 0.56817913, 0.7258603],
            ],
            [
                [0.431774, 0.40672457, 0.39094293, 0.63833976],
                [0.47457936, 0.51558167, 0.58189815, 0.5340911],
                [0.45442006, 0.5345709, 0.5615816, 0.5071402],
                [0.49547666, 0.37159446, 0.5301453, 0.55153173],
            ],
        ]
        expected = torch.tensor(expected, device=device, dtype=dtype).view(1, 3, 4, 4)

        out = guided_blur(guide, img, kernel_size, eps, border_type="replicate")
        if dtype == torch.bfloat16:
            # Multi-channel guidance chains three box blurs, a 3x3 solve and an einsum, so the error
            # is a few eps rather than one. Measured against this op's own float64 result the error
            # is 2.95 eps for bfloat16, 1.39 for float16 and 3.27 for float32 -- i.e. bfloat16 is not
            # an outlier, the harness's 1-eps budget is simply too tight here. bfloat16 also cannot
            # represent the inputs to better than 3.6e-3, over half that budget, before any
            # arithmetic runs. float16 and float32 still use the harness tolerances.
            self.assert_close(out, expected, rtol=4 * torch.finfo(dtype).eps, atol=4 * torch.finfo(dtype).eps)
        else:
            self.assert_close(out, expected)

    @staticmethod
    def _skip_without_padding(device, dtype, border_type):
        supported = {"reflect": supports_reflect_padding, "replicate": supports_replicate_padding}
        if border_type in supported and not supported[border_type](device, dtype):
            pytest.skip(f"this torch build has no {border_type} padding kernel for {dtype} on {device.type}")

    @pytest.mark.parametrize("guide_dim", [1, 3])
    @pytest.mark.parametrize("subsample", [2, 3])
    @pytest.mark.parametrize("shape", [(7, 10), (9, 11), (8, 13), (13, 8)])
    @pytest.mark.parametrize("kernel_size", [5, (3, 5)])
    def test_subsample_size_not_divisible(self, guide_dim, subsample, shape, kernel_size, device, dtype):
        """Any ``H`` and ``W`` work with ``subsample`` and the output has the input's size.

        Before the fix the upsampled coefficient maps came out ``floor(H / s) * s`` pixels tall and wide, and the final
        blend died with a raw broadcast error (1-channel guidance) or an ``einsum`` error (multi-channel guidance).
        The values are checked against the fast guided filter of the paper, resized to the full size.
        """
        self._skip_without_padding(device, dtype, "reflect")
        H, W = shape
        guide = torch.rand(2, guide_dim, H, W, device=device, dtype=dtype)
        inp = torch.rand(2, 2, H, W, device=device, dtype=dtype)

        eps = _value_eps(dtype)
        actual = guided_blur(guide, inp, kernel_size, eps, subsample=subsample)

        assert actual.shape == inp.shape
        assert actual.dtype == dtype
        expected = _fast_guided_filter_reference(guide, inp, kernel_size, eps, subsample).to(device=device, dtype=dtype)
        tolerance = _HALF_TOLERANCE.get(dtype)
        self.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        # the module forwards the same argument
        self.assert_close(GuidedBlur(kernel_size, eps, subsample=subsample)(guide, inp), actual)

    @pytest.mark.parametrize("guide_dim", [1, 3])
    @pytest.mark.parametrize("subsample", [2, 3])
    @pytest.mark.parametrize("shape", [(1, 10), (10, 1), (1, 1), (2, 9), (9, 2)])
    def test_subsample_larger_than_an_axis(self, guide_dim, subsample, shape, device, dtype):
        """An axis shorter than ``subsample`` is subsampled to one pixel, not to none.

        ``replicate`` padding, because ``reflect`` cannot pad a one-pixel axis with any window larger than one pixel,
        with or without ``subsample``.
        """
        self._skip_without_padding(device, dtype, "replicate")
        H, W = shape
        guide = torch.rand(1, guide_dim, H, W, device=device, dtype=dtype)
        inp = torch.rand(1, 2, H, W, device=device, dtype=dtype)

        eps = _value_eps(dtype)
        actual = guided_blur(guide, inp, 5, eps, border_type="replicate", subsample=subsample)

        assert actual.shape == inp.shape
        assert torch.isfinite(actual).all()
        expected = _fast_guided_filter_reference(guide, inp, 5, eps, subsample, "replicate").to(
            device=device, dtype=dtype
        )
        tolerance = _HALF_TOLERANCE.get(dtype)
        self.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    @pytest.mark.parametrize("guide_dim", [1, 3])
    @pytest.mark.parametrize("shape,subsample", [((8, 12), 2), ((12, 8), 2), ((9, 12), 3), ((16, 8), 4), ((6, 6), 6)])
    @pytest.mark.parametrize("kernel_size", [5, (3, 5)])
    def test_subsample_divisible_size_is_unchanged(self, guide_dim, shape, subsample, kernel_size, device, dtype):
        """Sizes that worked before the resize fix give the same bits: ``size=`` and ``scale_factor=`` agree."""
        self._skip_without_padding(device, dtype, "reflect")
        H, W = shape
        guide = torch.rand(2, guide_dim, H, W, device=device, dtype=dtype)
        inp = torch.rand(2, 3, H, W, device=device, dtype=dtype)

        actual = guided_blur(guide, inp, kernel_size, 0.01, subsample=subsample)
        with patch("kornia.filters.guided.interpolate", side_effect=_scale_factor_interpolate) as scale_factor_call:
            expected = guided_blur(guide, inp, kernel_size, 0.01, subsample=subsample)

        assert scale_factor_call.call_count == 4  # guidance and input down, the two coefficient maps up
        assert torch.equal(actual, expected)

    @pytest.mark.parametrize("guide_dim", [1, 3])
    def test_subsample_divisible_size_keeps_the_values_from_before_size_resize(self, guide_dim, device, dtype):
        self._skip_without_padding(device, dtype, "replicate")
        guide = torch.tensor(
            [[11, 0, 7, 1, 13, 14], [7, 14, 14, 12, 2, 0], [10, 4, 11, 8, 3, 1], [14, 6, 3, 2, 6, 4]],
            device=device,
            dtype=dtype,
        ).view(1, 1, 4, 6)
        inp = torch.tensor(
            [[15, 12, 8, 7, 14, 10], [0, 7, 6, 2, 5, 14], [10, 5, 12, 1, 6, 15], [13, 1, 8, 13, 4, 9]],
            device=device,
            dtype=dtype,
        ).view(1, 1, 4, 6)
        if guide_dim == 3:
            guide = torch.cat([guide, guide.flip(-1), guide.flip(-2)], dim=1)
        expected = {
            # Generated with ``guided_blur(guide / 16, inp / 16, 3, 0.01, border_type="replicate", subsample=2)`` in
            # float64 on kornia 4df7b3040, which resized by ``scale_factor`` (guide: 11, 0, 7, ... as above).
            1: [
                [0.7266702, 0.3351883, 0.5368115, 0.3295509, 0.6382735, 0.6360668],
                [0.5934205, 0.7918876, 0.7382599, 0.6238963, 0.3651183, 0.3326832],
                [0.6852125, 0.4912689, 0.6089832, 0.4800173, 0.3805514, 0.3665639],
                [0.7858949, 0.5469661, 0.4165676, 0.3695447, 0.3933863, 0.375],
            ],
            3: [
                [0.7919061, 0.4808834, 0.4991143, 0.3269996, 0.624895, 0.5421822],
                [0.6066366, 0.6224128, 0.6795592, 0.5507318, 0.3280907, 0.3211122],
                [0.5977503, 0.6155415, 0.6254567, 0.505276, 0.3778669, 0.354401],
                [0.6942412, 0.4737252, 0.4527047, 0.3636655, 0.4088435, 0.375],
            ],
        }[guide_dim]
        expected = torch.tensor(expected, device=device, dtype=dtype).view(1, 1, 4, 6)

        actual = guided_blur(guide / 16, inp / 16, 3, 0.01, border_type="replicate", subsample=2)

        # measured against float64 on this fixture: 5e-4 for float16, 1.1e-2 for bfloat16
        tolerance = {torch.float16: 0.01, torch.bfloat16: 0.05}.get(dtype)
        self.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    @pytest.mark.parametrize("guide_dim", [1, 3])
    @pytest.mark.parametrize(
        "subsample,error",
        [
            (0, BaseError),
            (-1, BaseError),
            (True, BaseError),
            (False, BaseError),
            (1.0, TypeCheckError),
            (1.5, TypeCheckError),
            (2.0, TypeCheckError),
            (np.float64(2.0), TypeCheckError),
            (np.True_, TypeCheckError),
            ("2", TypeCheckError),
            (None, TypeCheckError),
        ],
    )
    def test_exception_subsample_not_a_positive_int(self, guide_dim, subsample, error):
        """``subsample`` is a positive ``int``: a value outside the range raises ``BaseError``, a non-integer
        ``TypeCheckError``, and the message shows the value.

        Before the check, with 1-channel guidance ``0``, ``-1``, ``False``, ``True``, ``1.0`` and ``0.5`` ran as no
        subsampling, and on images of 8 x 8 and up a float above 1 ran as that factor where the round trip lands on
        ``H`` and ``W`` (``2.0`` on 8 x 12, ``1.5`` on 12 x 12) and failed at the final blend with a broadcast
        ``RuntimeError`` otherwise. With multi-channel guidance ``0``, ``-1`` and ``False`` failed in ``view`` with a
        ``RuntimeError``, ``True`` ran as 1, and on images of 8 x 8 and up every float failed in ``view`` with a
        ``TypeError``. On a tiny image a float failed earlier, in ``interpolate`` (``1.5`` on 1 x 10) or in the reflect
        ``pad`` (``2.0`` on 2 x 12). ``"2"`` and ``None`` failed in the comparison ``subsample > 1`` with a
        ``TypeError``, with either guidance.
        """
        guide = torch.rand(1, guide_dim, 8, 12)
        inp = torch.rand(1, 2, 8, 12)
        message = rf"`subsample` must be a positive integer\. Got: {re.escape(repr(subsample))}"

        # ``TypeCheckError`` is a ``BaseError``: the exact type tells the type check from the range check
        with pytest.raises(error, match=message) as excinfo:
            guided_blur(guide, inp, 5, 0.1, subsample=subsample)
        assert type(excinfo.value) is error
        # the module rejects it when it is built, not on the first call
        with pytest.raises(error, match=message) as excinfo:
            GuidedBlur(5, 0.1, subsample=subsample)
        assert type(excinfo.value) is error

    @pytest.mark.parametrize("guide_dim", [1, 3])
    @pytest.mark.parametrize("shape", [(8, 12), (7, 10)])
    def test_subsample_one_is_no_subsampling(self, guide_dim, shape, device, dtype):
        self._skip_without_padding(device, dtype, "reflect")
        guide = torch.rand(1, guide_dim, *shape, device=device, dtype=dtype)
        inp = torch.rand(1, 2, *shape, device=device, dtype=dtype)

        actual = guided_blur(guide, inp, 5, 0.01, subsample=1)

        assert torch.equal(actual, guided_blur(guide, inp, 5, 0.01))
        assert torch.equal(GuidedBlur(5, 0.01, subsample=1)(guide, inp), actual)

    @pytest.mark.parametrize("guide_dim", [1, 3])
    @pytest.mark.parametrize(
        "subsample,kernel_size",
        [(np.int64(2), 5), (np.int32(3), 5), (np.uint8(2), 5), (np.uint8(2), 301)],
        ids=["int64", "int32", "uint8", "uint8-wide-window"],
    )
    def test_subsample_accepts_an_integer_that_is_not_a_python_int(
        self, guide_dim, subsample, kernel_size, device, dtype
    ):
        """``numpy`` integers are accepted and give the result of the equal Python ``int``.

        ``guided_blur`` continues with ``int(subsample)``: ``(301 - 1) // numpy.uint8(2)`` is 300, which does not fit
        the ``uint8`` the window arithmetic would otherwise stay in (``OverflowError``). The size, 66, is a multiple of
        every ``subsample`` here, so these cases pin the integer types, not the handling of other sizes.
        """
        guide = torch.rand(1, guide_dim, 66, 66, device=device, dtype=dtype)
        inp = torch.rand(1, 2, 66, 66, device=device, dtype=dtype)

        actual = guided_blur(guide, inp, kernel_size, 0.01, border_type="constant", subsample=subsample)

        assert torch.equal(
            actual, guided_blur(guide, inp, kernel_size, 0.01, border_type="constant", subsample=int(subsample))
        )

    @pytest.mark.parametrize("guide_dim", [1, 3])
    def test_gradcheck_subsample_size_not_divisible(self, guide_dim, device) -> None:
        guide = torch.rand(1, guide_dim, 7, 9, device=device, dtype=torch.float64)
        img = torch.rand(1, 2, 7, 9, device=device, dtype=torch.float64)
        self.gradcheck(partial(guided_blur, subsample=2), (guide, img, 5, 0.1), nondet_tol=1e-4)

    @pytest.mark.parametrize("guide_dim", [1, 3])
    @pytest.mark.parametrize("shape", [(9, 11), (7, 10)])
    def test_dynamo_subsample_size_not_divisible(self, guide_dim, shape, device, dtype, torch_optimizer) -> None:
        self._skip_without_padding(device, dtype, "reflect")
        guide = torch.rand(2, guide_dim, *shape, device=device, dtype=dtype)
        data = torch.rand(2, 3, *shape, device=device, dtype=dtype)
        op = GuidedBlur(5, 0.1, subsample=2)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(guide, data), op_optimized(guide, data))
