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

import warnings

import numpy as np
import pytest
import torch

import kornia
from kornia.core._compat import torch_version_lt
from kornia.core.check import are_checks_enabled, disable_checks, enable_checks
from kornia.core.exceptions import BaseError, ValueCheckError

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, dynamo_is_available


class TestPyrUp(BaseTester):
    def test_shape(self, device, dtype):
        inp = torch.zeros(1, 2, 4, 4, device=device, dtype=dtype)
        pyr = kornia.geometry.PyrUp()
        assert pyr(inp).shape == (1, 2, 8, 8)

    def test_shape_batch(self, device, dtype):
        inp = torch.zeros(2, 2, 4, 4, device=device, dtype=dtype)
        pyr = kornia.geometry.PyrUp()
        assert pyr(inp).shape == (2, 2, 8, 8)

    def test_gradcheck(self, device):
        img = torch.rand(1, 2, 5, 4, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.pyrup, (img,), nondet_tol=1e-8)


class TestPyrDown(BaseTester):
    @pytest.mark.parametrize("dynamo", [False, True])
    @pytest.mark.parametrize("operation", ["pyrdown", "build_pyramid", "rescale"])
    def test_onnx_dynamic_spatial_shape(self, operation, dynamo, tmp_path):
        import numpy as np

        pytest.importorskip("onnx")
        ort = pytest.importorskip("onnxruntime")
        if dynamo:
            if not dynamo_is_available():
                pytest.skip("Dynamo ONNX export is unavailable on this torch/python pair")
            if torch_version_lt(2, 9, 0):
                pytest.skip("Dynamo ONNX export of a dynamic height or width needs torch >= 2.9")
            pytest.importorskip("onnxscript")

        class Wrapper(torch.nn.Module):
            def forward(self, x):
                if operation == "pyrdown":
                    return kornia.geometry.transform.pyrdown(x)
                if operation == "build_pyramid":
                    return kornia.geometry.transform.build_pyramid(x, 3)[-1]
                return kornia.geometry.transform.rescale(x, 0.5)

        model = Wrapper().eval()
        path = tmp_path / "model.onnx"
        export_kwargs = {
            "input_names": ["x"],
            "output_names": ["y"],
            "opset_version": 18,
            "dynamo": dynamo,
        }
        if dynamo:
            export_kwargs["dynamic_shapes"] = (
                {
                    0: torch.export.Dim("batch"),
                    2: torch.export.Dim("height", min=6),
                    3: torch.export.Dim("width", min=6),
                },
            )
        else:
            export_kwargs["dynamic_axes"] = {
                "x": {0: "batch", 2: "height", 3: "width"},
                "y": {0: "out_batch", 2: "out_height", 3: "out_width"},
            }
        torch.onnx.export(model, (torch.rand(2, 1, 22, 44),), path, **export_kwargs)

        x = torch.rand(3, 1, 33, 64)
        actual = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"x": x.numpy()})[0]
        np.testing.assert_allclose(actual, model(x).numpy(), rtol=1e-5, atol=1e-6)

    def test_area_rescale_static_torchscript_export(self, tmp_path):
        import numpy as np

        pytest.importorskip("onnx")
        ort = pytest.importorskip("onnxruntime")

        class Wrapper(torch.nn.Module):
            def forward(self, x):
                return kornia.geometry.transform.rescale(x, 0.5, interpolation="area")

        model = Wrapper().eval()
        path = tmp_path / "model.onnx"
        x = torch.rand(1, 1, 22, 44)
        torch.onnx.export(model, (x,), path, input_names=["x"], output_names=["y"], opset_version=18, dynamo=False)
        actual = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"x": x.numpy()})[0]
        np.testing.assert_allclose(actual, model(x).numpy(), rtol=1e-5, atol=1e-6)

    @pytest.mark.skipif(not dynamo_is_available(), reason="torch.compile is unavailable on this torch/python pair")
    def test_eager_backend_trace_keeps_rescales_empty_output(self):
        # kornia's is_exporting() is also true inside torch.compile on torch 2.5.1 and 2.9.1.
        x = torch.rand(1, 1, 22, 44)
        actual = torch.compile(lambda t: kornia.geometry.transform.rescale(t, 0.01), backend="eager", fullgraph=True)(x)
        assert actual.shape == kornia.geometry.transform.rescale(x, 0.01).shape == (1, 1, 0, 0)

    @pytest.mark.parametrize("dynamo", [False, True])
    def test_onnx_fractional_factor_preserves_truncation(self, dynamo, tmp_path):
        import numpy as np

        pytest.importorskip("onnx")
        ort = pytest.importorskip("onnxruntime")
        if dynamo:
            if not dynamo_is_available():
                pytest.skip("Dynamo ONNX export is unavailable on this torch/python pair")
            pytest.importorskip("onnxscript")

        class Wrapper(torch.nn.Module):
            def forward(self, x):
                return kornia.geometry.transform.pyrdown(x, factor=1.1)

        model = Wrapper().eval()
        path = tmp_path / "model.onnx"
        torch.onnx.export(
            model,
            (torch.rand(1, 1, 33, 55),),
            path,
            input_names=["x"],
            output_names=["y"],
            opset_version=18,
            dynamo=dynamo,
        )
        x = torch.rand(1, 1, 33, 55)
        expected = model(x)
        assert expected.shape[-2:] == (29, 49)
        actual = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"x": x.numpy()})[0]
        np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-6)

    @pytest.mark.skipif(not dynamo_is_available(), reason="torch.compile is unavailable on this torch/python pair")
    def test_fractional_factor_compile_shape(self):
        model = torch.compile(
            lambda x: kornia.geometry.transform.pyrdown(x, factor=1.1), backend="eager", fullgraph=True
        )
        assert model(torch.rand(1, 1, 33, 55)).shape[-2:] == (29, 49)

    def test_shape(self, device, dtype):
        inp = torch.zeros(1, 2, 4, 4, device=device, dtype=dtype)
        pyr = kornia.geometry.PyrDown()
        assert pyr(inp).shape == (1, 2, 2, 2)

    def test_shape_custom_factor(self, device, dtype):
        inp = torch.zeros(1, 2, 9, 9, device=device, dtype=dtype)
        pyr = kornia.geometry.PyrDown(factor=3.0)
        assert pyr(inp).shape == (1, 2, 3, 3)

    @pytest.mark.parametrize("as_module", [False, True])
    @pytest.mark.parametrize("align_corners", [False, True])
    @pytest.mark.parametrize(
        "height, width, factor, expected_size",
        [
            (11, 22, 1.1, (10, 20)),
            (9, 18, 1.8, (5, 10)),
            (9, 15, 1.2, (7, 12)),
            (11, 15, 2.0, (5, 7)),
            (33, 55, 1.1, (29, 49)),
            (39, 65, 1.3, (30, 50)),
        ],
    )
    def test_fractional_factor_axis_symmetry(
        self, device, dtype, as_module, align_corners, height, width, factor, expected_size
    ):
        inp = torch.arange(height * width, device=device, dtype=dtype).reshape(1, 1, height, width)
        inp = inp / (height * width)
        if as_module:
            pyr = kornia.geometry.PyrDown(factor=factor, align_corners=align_corners)
            out = pyr(inp)
            transposed = pyr(inp.transpose(-2, -1))
        else:
            out = kornia.geometry.pyrdown(inp, factor=factor, align_corners=align_corners)
            transposed = kornia.geometry.pyrdown(inp.transpose(-2, -1), factor=factor, align_corners=align_corners)
        assert out.shape == (1, 1, *expected_size)
        self.assert_close(out, transposed.transpose(-2, -1))

    def test_dynamo(self, device, dtype, torch_optimizer):
        inp = torch.rand(1, 1, 11, 22, device=device, dtype=dtype)
        pyr = kornia.geometry.PyrDown(factor=1.1)
        actual = torch_optimizer(pyr)(inp)
        assert actual.shape == (1, 1, 10, 20)
        self.assert_close(actual, pyr(inp))

    def test_shape_batch(self, device, dtype):
        inp = torch.zeros(2, 2, 4, 4, device=device, dtype=dtype)
        pyr = kornia.geometry.PyrDown()
        assert pyr(inp).shape == (2, 2, 2, 2)

    def test_symmetry_preserving(self, device, dtype):
        inp = torch.zeros(1, 1, 6, 6, device=device, dtype=dtype)
        inp[:, :, 2:4, 2:4] = 1.0
        pyr_out = kornia.geometry.PyrDown()(inp).squeeze()
        self.assert_close(pyr_out, pyr_out.flip(0))
        self.assert_close(pyr_out, pyr_out.flip(1))

    @pytest.mark.parametrize("factor", [2.0, 1.1, 1.8])
    def test_gradcheck(self, device, factor):
        img = torch.rand(1, 2, 5, 4, device=device, dtype=torch.float64)
        self.gradcheck(lambda x: kornia.geometry.pyrdown(x, factor=factor), (img,), nondet_tol=1e-8)

    @pytest.mark.parametrize("op, side", [("pyrdown", 5), ("pyrup", 4)])
    def test_convention_align_corners_and_border_type_change_output(self, op, side, device, dtype):
        # The align_corners (default False) and border_type (default 'reflect') defaults of pyrdown and
        # pyrup, which are independent implementations, each change the output.
        fn = getattr(kornia.geometry.transform, op)
        x = torch.arange(float(side * side), device=device, dtype=dtype).view(1, 1, side, side)

        out_default = fn(x)
        self.assert_close(out_default, fn(x, align_corners=False), rtol=1e-2, atol=1e-2)
        assert not torch.allclose(out_default, fn(x, align_corners=True), atol=1e-2, rtol=1e-2)
        self.assert_close(out_default, fn(x, border_type="reflect"), rtol=1e-2, atol=1e-2)
        assert not torch.allclose(out_default, fn(x, border_type="constant"), atol=1e-2, rtol=1e-2)

    def test_convention_floor_not_ceil_on_odd_size(self, device, dtype):
        # pyrdown uses floor(side / factor); OpenCV's pyrDown uses (side + 1) // 2, so an odd side
        # differs: 5x5 at the default factor=2.0 gives 2x2, not 3x3.
        x = torch.rand(1, 1, 5, 5, device=device, dtype=dtype)
        assert kornia.geometry.transform.pyrdown(x).shape == (1, 1, 2, 2)


class TestScalePyramid(BaseTester):
    @pytest.mark.parametrize("sigma, ksize", [(1.6, 13), (0.7, 7), (3.2, 27)])
    def test_reference_kernel_rounding(self, device, sigma, ksize):
        # Preserve the original float64 calculation followed by a single float32 rounding.
        x = torch.arange(ksize, device="cpu", dtype=torch.float64) - ksize // 2
        reference = torch.exp(-0.5 * x**2 / sigma**2)
        reference = (reference / reference.sum()).float()
        with device:
            kernel = kornia.geometry.ScalePyramid._make_gaussian_kernel1d(sigma, ksize)
        self.assert_close(kernel.cpu(), reference, rtol=0, atol=0)

    @pytest.mark.parametrize("default_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
    @pytest.mark.parametrize("init_sigma", [0.4, 1.6])
    @pytest.mark.parametrize("double_image", [False, True])
    def test_default_device_buffers(self, device, dtype, default_dtype, init_sigma, double_image):
        with torch.device("cpu"):
            reference = kornia.geometry.ScalePyramid(init_sigma=init_sigma, double_image=double_image)
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(default_dtype)
            with device:
                pyramid = kornia.geometry.ScalePyramid(init_sigma=init_sigma, double_image=double_image)
        finally:
            torch.set_default_dtype(original_dtype)

        expected_device = torch.empty(0, device=device).device
        assert pyramid.state_dict().keys() == reference.state_dict().keys()
        assert (pyramid._gk_init is None) == (init_sigma == 0.4)
        for name, kernel in pyramid.named_buffers():
            assert kernel.device == expected_device
            assert kernel.dtype == torch.float32
            self.assert_close(kernel.cpu(), reference.get_buffer(name), rtol=0, atol=0)

        pyramid.to(dtype=dtype)
        for name, kernel in pyramid.named_buffers():
            assert kernel.device == expected_device
            assert kernel.dtype == dtype
            self.assert_close(kernel.cpu(), reference.get_buffer(name).to(dtype), rtol=0, atol=0)

    def test_shape_tuple(self, device, dtype):
        inp = torch.zeros(3, 2, 41, 41, device=device, dtype=dtype)
        SP = kornia.geometry.ScalePyramid(n_levels=1, min_size=30)
        out = SP(inp)
        assert len(out) == 3
        assert len(out[0]) == 1
        assert len(out[1]) == 1
        assert len(out[2]) == 1

    def test_shape_batch(self, device, dtype):
        inp = torch.zeros(3, 2, 31, 31, device=device, dtype=dtype)
        SP = kornia.geometry.ScalePyramid(n_levels=1)
        sp, _, _ = SP(inp)
        assert sp[0].shape == (3, 2, 3 + 1, 31, 31)

    def test_shape_batch_double(self, device, dtype):
        inp = torch.zeros(3, 2, 31, 31, device=device, dtype=dtype)
        SP = kornia.geometry.ScalePyramid(n_levels=1, double_image=True)
        sp, _, _ = SP(inp)
        assert sp[0].shape == (3, 2, 1 + 3, 62, 62)

    def test_n_levels_shape(self, device, dtype):
        inp = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        SP = kornia.geometry.ScalePyramid(n_levels=3)
        sp, _, _ = SP(inp)
        assert sp[0].shape == (1, 1, 3 + 3, 32, 32)

    def test_blur_order(self, device, dtype):
        inp = torch.rand(1, 1, 31, 31, device=device, dtype=dtype)
        SP = kornia.geometry.ScalePyramid(n_levels=3)
        sp, _, _ = SP(inp)
        for _, pyr_level in enumerate(sp):
            for _, img in enumerate(pyr_level):
                img = img.squeeze().view(3, -1)
                max_per_blur_level_val, _ = img.max(dim=1)
                assert torch.argmax(max_per_blur_level_val).item() == 0

    def test_symmetry_preserving(self, device, dtype):
        PS = 16
        R = 2
        inp = torch.zeros(1, 1, PS, PS, device=device, dtype=dtype)
        inp[..., PS // 2 - R : PS // 2 + R, PS // 2 - R : PS // 2 + R] = 1.0
        SP = kornia.geometry.ScalePyramid(n_levels=3)
        sp, _, _ = SP(inp)
        for _, pyr_level in enumerate(sp):
            for _, img in enumerate(pyr_level):
                img = img.squeeze()
                self.assert_close(img, img.flip(1))
                self.assert_close(img, img.flip(2))

    def test_gradcheck(self, device):
        img = torch.rand(1, 2, 7, 9, device=device, dtype=torch.float64)
        from kornia.geometry import ScalePyramid as SP

        def sp_tuple(img):
            sp, _, _ = SP()(img)
            return tuple(sp)

        self.gradcheck(sp_tuple, (img,), nondet_tol=1e-4)

    @pytest.mark.parametrize(("double_image", "input_sigma"), [(False, 0.5), (True, 1.0)])
    def test_init_sigma_below_input_blur(self, device, dtype, double_image, input_sigma):
        # An init_sigma below the assumed input blur leaves the first level at the input blur,
        # so the pyramid and its sigmas must match the one built with init_sigma == input blur
        # (#4796).
        inp = torch.rand(1, 1, 32, 32, device=device, dtype=dtype)
        low = kornia.geometry.ScalePyramid(n_levels=1, init_sigma=input_sigma / 2, double_image=double_image)
        ref = kornia.geometry.ScalePyramid(n_levels=1, init_sigma=input_sigma, double_image=double_image)
        pyr, sigmas, _ = low.to(device)(inp)
        ref_pyr, ref_sigmas, _ = ref.to(device)(inp)
        expected = torch.tensor([1.0, 2.0, 4.0, 8.0], device=device, dtype=dtype) * input_sigma
        for octave in range(len(sigmas)):
            self.assert_close(sigmas[octave][0], expected)
            self.assert_close(sigmas[octave], ref_sigmas[octave])
            self.assert_close(pyr[octave], ref_pyr[octave])

    @pytest.mark.parametrize(
        "extra_levels",
        [
            *[0, -1, -3, 1.5, 3.0, None, "3", True, False],
            *[np.int64(0), np.int32(-1), np.float64(3.0), np.bool_(True), np.bool_(False)],
            *[np.array(0), np.array(3.0), np.array(True)],
            *[torch.tensor(0), torch.tensor(-1), torch.tensor(3.0), torch.tensor(True)],
        ],
    )
    def test_invalid_extra_levels_6400(self, extra_levels):
        # extra_levels=0 made `pyr[-1][-0]` pick level 0, so every octave after the first was seeded from the
        # least blurred level while `sigmas` still labelled it with the base sigma.
        with pytest.raises(BaseError, match="Invalid extra_levels, it must be a positive integer"):
            kornia.geometry.ScalePyramid(extra_levels=extra_levels)

    def test_non_scalar_extra_levels_6400(self):
        # Not parametrized: conftest compares every parameter with `torch.float16`, which is ambiguous for an array.
        for extra_levels in (np.array([1, 2]), torch.tensor([1, 2])):
            with pytest.raises(BaseError, match="Invalid extra_levels, it must be a positive integer"):
                kornia.geometry.ScalePyramid(extra_levels=extra_levels)

    @pytest.mark.parametrize(
        "extra_levels",
        [
            pytest.param(np.int8(2), id="np.int8"),
            pytest.param(np.uint8(2), id="np.uint8"),
            pytest.param(np.int32(2), id="np.int32"),
            pytest.param(np.int64(2), id="np.int64"),
            pytest.param(np.array(2), id="np.array-0d"),
            pytest.param(torch.tensor(2), id="tensor-0d-int64"),
            pytest.param(torch.tensor(2, dtype=torch.int32), id="tensor-0d-int32"),
        ],
    )
    def test_integer_like_extra_levels_6400(self, device, dtype, extra_levels):
        # None of these is an `int` subclass, but all were accepted before extra_levels was validated.
        inp = torch.rand(1, 1, 96, 96, device=device, dtype=dtype)
        pyramid = kornia.geometry.ScalePyramid(extra_levels=extra_levels)
        assert type(pyramid.extra_levels) is int
        assert pyramid.extra_levels == 2
        for got, expected in zip(pyramid(inp), kornia.geometry.ScalePyramid(extra_levels=2)(inp)):
            assert len(got) == len(expected)
            for got_octave, expected_octave in zip(got, expected):
                self.assert_close(got_octave, expected_octave, rtol=0, atol=0)

    @pytest.mark.parametrize("extra_levels", [0, -2, 1.5, None, True, np.bool_(True), torch.tensor(True)])
    def test_invalid_extra_levels_rejected_with_checks_disabled_6400(self, extra_levels):
        # Not a KORNIA_CHECK: with checks off, extra_levels=0 would construct and then fail inside forward.
        checks_were_enabled = are_checks_enabled()
        disable_checks()
        try:
            with pytest.raises(ValueCheckError, match="Invalid extra_levels"):
                kornia.geometry.ScalePyramid(extra_levels=extra_levels)
        finally:
            if checks_were_enabled:
                enable_checks()

    @pytest.mark.parametrize(
        "n_levels",
        [
            *[0, -1, -2, 1.5, 3.0, None, "3", True, False],
            *[np.int64(0), np.int32(-1), np.float64(3.0), np.bool_(True)],
            *[torch.tensor(0), torch.tensor(-1), torch.tensor(3.0), torch.tensor(True)],
        ],
    )
    def test_invalid_n_levels(self, n_levels):
        # n_levels=0 divided by zero in the level-to-level blur step 2 ** (1 / n_levels), n_levels=-1 with
        # extra_levels=1 built a one-level pyramid without an error, and other negatives failed inside forward.
        with pytest.raises(ValueCheckError, match="Invalid n_levels, it must be a positive integer"):
            kornia.geometry.ScalePyramid(n_levels=n_levels)

    def test_negative_n_levels_no_longer_builds_a_pyramid(self):
        # The one negative value that used to construct and run: it returned three octaves of a single level.
        with pytest.raises(ValueCheckError, match="Invalid n_levels"):
            kornia.geometry.ScalePyramid(n_levels=-1, extra_levels=1)

    @pytest.mark.parametrize(
        "n_levels",
        [
            pytest.param(np.int64(2), id="np.int64"),
            pytest.param(np.array(2), id="np.array-0d"),
            pytest.param(torch.tensor(2), id="tensor-0d-int64"),
            pytest.param(torch.tensor(2, dtype=torch.int32), id="tensor-0d-int32"),
        ],
    )
    def test_integer_like_n_levels(self, device, dtype, n_levels):
        # These were accepted before n_levels was validated; they now build the same pyramid as the Python int.
        torch.manual_seed(0)
        inp = torch.rand(1, 1, 96, 96).to(device=device, dtype=dtype)
        pyramid = kornia.geometry.ScalePyramid(n_levels=n_levels)
        assert type(pyramid.n_levels) is int
        assert pyramid.n_levels == 2
        for got, expected in zip(pyramid(inp), kornia.geometry.ScalePyramid(n_levels=2)(inp)):
            assert len(got) == len(expected)
            for got_octave, expected_octave in zip(got, expected):
                self.assert_close(got_octave, expected_octave, rtol=0, atol=0)

    @pytest.mark.parametrize("min_size", [-1, -0.5, float("nan"), torch.tensor(-1)])
    def test_invalid_min_size(self, min_size):
        # A negative or NaN min_size never stopped the octave loop, which halved the image to an empty one and failed
        # inside F.interpolate.
        with pytest.raises(ValueCheckError, match="Invalid min_size, it must be a non-negative number"):
            kornia.geometry.ScalePyramid(min_size=min_size)

    @pytest.mark.parametrize(("min_size", "num_octaves"), [(0, 7), (0.0, 7), (15.0, 3), (float("inf"), 1)])
    def test_min_size_bounds(self, device, dtype, min_size, num_octaves):
        # min_size=0 keeps halving down to one pixel: 64, 32, ..., 1.
        torch.manual_seed(0)
        inp = torch.rand(1, 1, 64, 64).to(device=device, dtype=dtype)
        pyr, _, _ = kornia.geometry.ScalePyramid(min_size=min_size)(inp)
        assert len(pyr) == num_octaves
        assert pyr[-1].shape[-2:] == (64 // 2 ** (num_octaves - 1),) * 2

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"n_levels": 0}, "Invalid n_levels"),
            ({"n_levels": True}, "Invalid n_levels"),
            ({"min_size": -1}, "Invalid min_size"),
        ],
    )
    def test_invalid_n_levels_and_min_size_rejected_with_checks_disabled(self, kwargs, message):
        # Not a KORNIA_CHECK, like extra_levels: with checks off, these would construct and then fail inside forward.
        checks_were_enabled = are_checks_enabled()
        disable_checks()
        try:
            with pytest.raises(ValueCheckError, match=message):
                kornia.geometry.ScalePyramid(**kwargs)
        finally:
            if checks_were_enabled:
                enable_checks()

    @pytest.mark.parametrize("n_levels", [1, 3])
    @pytest.mark.parametrize("extra_levels", [1, 2, 3])
    def test_octave_is_seeded_from_level_n_levels_6400(self, device, dtype, n_levels, extra_levels):
        # Level n_levels is blurred to twice the first level's sigma, so downscaling it by 2 gives the next octave
        # the same octave-relative blur. It does not depend on how many spare levels follow it.
        inp = torch.rand(1, 1, 96, 96, device=device, dtype=dtype)
        pyr, sigmas, _ = kornia.geometry.ScalePyramid(n_levels, extra_levels=extra_levels)(inp)
        assert len(pyr) >= 3
        for octave in range(len(pyr) - 1):
            assert pyr[octave].shape[2] == n_levels + extra_levels
            size = (pyr[octave].shape[-2] // 2, pyr[octave].shape[-1] // 2)

            def downscale(level, size=size, octave=octave):
                return torch.nn.functional.interpolate(
                    pyr[octave][:, :, level], size=size, mode="bilinear", align_corners=True
                )

            self.assert_close(pyr[octave + 1][:, :, 0], downscale(n_levels))
            assert not torch.allclose(pyr[octave + 1][:, :, 0], downscale(0))
            # The labels agree with the seed: level n_levels is at twice the first level's sigma, and the next
            # octave's first level, downscaled by 2, is back at the first level's sigma.
            self.assert_close(sigmas[octave][:, n_levels], 2.0 * sigmas[octave][:, 0])
            self.assert_close(sigmas[octave + 1][:, 0], sigmas[octave][:, 0])


class TestPyramidLevelValidation(BaseTester):
    @pytest.mark.parametrize("op", ["build_pyramid", "build_laplacian_pyramid"])
    @pytest.mark.parametrize("max_level", [0, -1, -2, 1.5, -1.5, None, "2", True, False])
    def test_invalid_max_level_3927(self, op, max_level, device, dtype):
        image = torch.rand(1, 1, 8, 8, device=device, dtype=dtype)
        with pytest.raises(BaseError, match="Invalid max_level, it must be a positive integer"):
            getattr(kornia.geometry.transform, op)(image, max_level)


class TestBuildPyramid(BaseTester):
    def test_smoke(self, device, dtype):
        sample = torch.ones(1, 2, 4, 5, device=device, dtype=dtype)
        pyramid = kornia.geometry.transform.build_pyramid(sample, max_level=1)
        assert len(pyramid) == 1
        assert pyramid[0].shape == (1, 2, 4, 5)

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    @pytest.mark.parametrize("channels", [1, 3])
    @pytest.mark.parametrize("max_level", [2, 3, 4])
    def test_num_levels(self, batch_size, channels, max_level, device, dtype):
        height, width = 16, 20
        sample = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        pyramid = kornia.geometry.transform.build_pyramid(sample, max_level)
        assert len(pyramid) == max_level
        for i in range(1, max_level):
            img = pyramid[i]
            denom = 2**i
            expected_shape = (batch_size, channels, height // denom, width // denom)
            assert img.shape == expected_shape

    def test_gradcheck(self, device):
        max_level = 1
        batch_size, channels, height, width = 1, 2, 7, 9
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.transform.build_pyramid, (img, max_level))


class TestBuildLaplacianPyramid(BaseTester):
    def test_smoke(self, device, dtype):
        sample = torch.ones(1, 2, 4, 5, device=device, dtype=dtype)
        pyramid = kornia.geometry.transform.build_laplacian_pyramid(sample, max_level=1)
        assert len(pyramid) == 1
        assert pyramid[0].shape == (1, 2, 4, 5)

    @pytest.mark.parametrize("height,width", [(5, 8), (8, 5)])
    def test_mixed_power_of_two_size_padding(self, height, width, device, dtype):
        sample = torch.rand(1, 2, height, width, device=device, dtype=dtype)
        pyramid = kornia.geometry.transform.build_laplacian_pyramid(sample, max_level=2)
        assert [img.shape for img in pyramid] == [torch.Size((1, 2, 8, 8)), torch.Size((1, 2, 4, 4))]

    @pytest.mark.parametrize(
        "height,width,max_level,level_sizes",
        [
            (8, 6, 2, ((8, 6), (4, 3))),
            (12, 8, 2, ((12, 8), (6, 4))),
            (8, 12, 3, ((8, 12), (4, 6), (2, 3))),
            (20, 64, 3, ((20, 64), (10, 32), (5, 16))),
            (20, 64, 4, ((32, 64), (16, 32), (8, 16), (4, 8))),
        ],
    )
    def test_padding_only_when_required_by_level_alignment(self, height, width, max_level, level_sizes, device, dtype):
        sample = torch.rand(1, 2, height, width, device=device, dtype=dtype)
        pyramid = kornia.geometry.transform.build_laplacian_pyramid(sample, max_level)
        assert [img.shape[-2:] for img in pyramid] == list(level_sizes)

    # The default rule until the #6319 deprecation ends. The sizes are worked out by hand from the rule, and the first
    # rows are the table of the issue.
    @pytest.mark.parametrize(
        "height,width,max_level,level_sizes",
        [
            (96, 96, 3, ((128, 128), (64, 64), (32, 32))),
            (480, 640, 1, ((512, 1024),)),
            (480, 640, 3, ((512, 1024), (256, 512), (128, 256))),
            (5, 8, 2, ((8, 8), (4, 4))),
            (20, 64, 4, ((32, 64), (16, 32), (8, 16), (4, 8))),
        ],
    )
    def test_default_warns_when_the_shape_will_change(self, height, width, max_level, level_sizes, device, dtype):
        sample = torch.rand(1, 2, height, width, device=device, dtype=dtype)
        with pytest.warns(FutureWarning, match="minimal_padding=True") as record:
            pyramid = kornia.geometry.transform.build_laplacian_pyramid(sample, max_level)
        assert len(record) == 1
        assert record[0].filename == __file__
        assert [tuple(img.shape[-2:]) for img in pyramid] == list(level_sizes)

    @pytest.mark.parametrize(
        "height,width,max_level,level_sizes",
        [
            (64, 96, 3, ((64, 96), (32, 48), (16, 24))),
            (96, 64, 3, ((96, 64), (48, 32), (24, 16))),
            (64, 100, 1, ((64, 100),)),
            (64, 64, 3, ((64, 64), (32, 32), (16, 16))),
            # Both rules pad to the same size: (7, 8) to (8, 8) with a stride of 4, and (3, 8) to (4, 8) with one of 2.
            (7, 8, 3, ((8, 8), (4, 4), (2, 2))),
            (3, 8, 2, ((4, 8), (2, 4))),
        ],
    )
    def test_default_stays_silent_when_the_shape_will_not_change(
        self, height, width, max_level, level_sizes, device, dtype
    ):
        sample = torch.rand(1, 2, height, width, device=device, dtype=dtype)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", "`build_laplacian_pyramid`", FutureWarning)
            pyramid = kornia.geometry.transform.build_laplacian_pyramid(sample, max_level)
            minimal = kornia.geometry.transform.build_laplacian_pyramid(sample, max_level, minimal_padding=True)
        assert [tuple(img.shape[-2:]) for img in pyramid] == list(level_sizes)
        for default_level, minimal_level in zip(pyramid, minimal, strict=True):
            assert torch.equal(default_level, minimal_level)

    @pytest.mark.device_agnostic
    def test_default_warns_exactly_when_a_returned_shape_differs(self):
        # A call that raises under the default rule has no shape to change, so it must not warn either.
        build = kornia.geometry.transform.build_laplacian_pyramid
        for border_type in ("reflect", "replicate"):
            for max_level in (1, 2, 3, 4):
                for height in range(1, 18):
                    for width in range(1, 18):
                        sample = torch.zeros(1, 1, height, width)
                        with warnings.catch_warnings(record=True) as record:
                            warnings.simplefilter("always")
                            try:
                                default = [img.shape for img in build(sample, max_level, border_type)]
                            except RuntimeError:
                                assert not any("`build_laplacian_pyramid`" in str(item.message) for item in record)
                                continue
                        minimal = [img.shape for img in build(sample, max_level, border_type, minimal_padding=True)]
                        warned = any("`build_laplacian_pyramid`" in str(item.message) for item in record)
                        assert warned == (default != minimal), (height, width, max_level, border_type)

    @pytest.mark.parametrize(
        "height,width,max_level,level_sizes",
        [
            (96, 96, 3, ((96, 96), (48, 48), (24, 24))),
            (64, 96, 3, ((64, 96), (32, 48), (16, 24))),
            (480, 640, 1, ((480, 640),)),
            (480, 640, 3, ((480, 640), (240, 320), (120, 160))),
            (64, 100, 1, ((64, 100),)),
            (7, 9, 1, ((7, 9),)),
            (5, 8, 2, ((6, 8), (3, 4))),
            (3, 8, 2, ((4, 8), (2, 4))),
            (7, 9, 3, ((8, 12), (4, 6), (2, 3))),
            (20, 64, 4, ((24, 64), (12, 32), (6, 16), (3, 8))),
            (17, 100, 4, ((24, 104), (12, 52), (6, 26), (3, 13))),
        ],
    )
    def test_minimal_padding_sizes(self, height, width, max_level, level_sizes, device, dtype):
        sample = torch.rand(2, 3, height, width, device=device, dtype=dtype)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", "`build_laplacian_pyramid`", FutureWarning)
            pyramid = kornia.geometry.transform.build_laplacian_pyramid(sample, max_level, minimal_padding=True)
        assert [img.shape for img in pyramid] == [torch.Size((2, 3, *size)) for size in level_sizes]

    def test_minimal_padding_one_level_is_the_input(self, device, dtype):
        sample = torch.rand(1, 2, 7, 9, device=device, dtype=dtype)
        pyramid = kornia.geometry.transform.build_laplacian_pyramid(sample, 1, minimal_padding=True)
        assert len(pyramid) == 1
        assert torch.equal(pyramid[0], sample)

    # The expected padded image is written by hand: reflection repeats the rows and columns before the last one.
    @pytest.mark.parametrize(
        "height,width,max_level,rows,columns",
        [
            (5, 8, 2, [0, 1, 2, 3, 4, 3], list(range(8))),
            (8, 5, 2, list(range(8)), [0, 1, 2, 3, 4, 3]),
            (7, 9, 3, [0, 1, 2, 3, 4, 5, 6, 5], [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5]),
        ],
    )
    @pytest.mark.parametrize("align_corners", [False, True])
    def test_minimal_padding_rebuilds_the_reflected_image(
        self, height, width, max_level, rows, columns, align_corners, device, dtype
    ):
        sample = torch.rand(2, 3, height, width, device=device, dtype=dtype)
        pyramid = kornia.geometry.transform.build_laplacian_pyramid(
            sample, max_level, "reflect", align_corners, minimal_padding=True
        )
        rebuilt = pyramid[-1]
        for level in reversed(pyramid[:-1]):
            rebuilt = kornia.geometry.transform.pyrup(rebuilt, "reflect", align_corners) + level
        self.assert_close(rebuilt, sample[:, :, rows][:, :, :, columns])

    # F.pad cannot reflect a padding at least as large as the side, so such a side repeats its last row or column.
    # With `border_type="reflect"` these sizes raise instead (see the next test): only these border types get here.
    @pytest.mark.parametrize(
        "height,width,max_level,rows,columns",
        [
            (1, 8, 2, [0, 0], list(range(8))),
            (8, 1, 2, list(range(8)), [0, 0]),
            (1, 8, 3, [0, 0, 0, 0], list(range(8))),
            (8, 1, 3, list(range(8)), [0, 0, 0, 0]),
            (1, 1, 3, [0, 0, 0, 0], [0, 0, 0, 0]),
            (2, 8, 3, [0, 1, 1, 1], list(range(8))),
            (8, 2, 3, list(range(8)), [0, 1, 1, 1]),
            (3, 8, 4, [0, 1, 2, 2, 2, 2, 2, 2], list(range(8))),
            # The padding of the height stays below the side and reflects, that of the width does not.
            (3, 1, 3, [0, 1, 2, 1], [0, 0, 0, 0]),
        ],
    )
    @pytest.mark.parametrize("border_type", ["constant", "replicate", "circular"])
    def test_minimal_padding_repeats_a_side_too_small_to_reflect(
        self, height, width, max_level, rows, columns, border_type, device, dtype
    ):
        sample = torch.rand(2, 3, height, width, device=device, dtype=dtype)
        build = kornia.geometry.transform.build_laplacian_pyramid
        pyramid = build(sample, max_level, border_type, minimal_padding=True)
        assert pyramid[0].shape[-2:] == (len(rows), len(columns))
        rebuilt = pyramid[-1]
        for level in reversed(pyramid[:-1]):
            rebuilt = kornia.geometry.transform.pyrup(rebuilt, border_type) + level
        self.assert_close(rebuilt, sample[:, :, rows][:, :, :, columns])

    # The blur of `pyrdown` and `pyrup` reflects two pixels, so it needs at least three: `(4, 4, 3)` raises from torch
    # by way of its side-2 level under the default rule too, and only the message is new here.
    @pytest.mark.parametrize(
        "height,width,max_level",
        [(1, 8, 2), (8, 1, 2), (2, 8, 2), (1, 8, 3), (4, 4, 3), (4, 16, 3), (16, 4, 3), (8, 8, 4), (1, 1, 2)],
    )
    def test_minimal_padding_reflect_border_rejects_a_side_too_small(self, height, width, max_level, device, dtype):
        sample = torch.rand(1, 1, height, width, device=device, dtype=dtype)
        with pytest.raises(BaseError, match="border_type"):
            kornia.geometry.transform.build_laplacian_pyramid(sample, max_level, minimal_padding=True)

    @pytest.mark.parametrize(
        "height,width,max_level,level_sizes",
        [
            # The smallest sides that the blur of each level takes.
            (3, 3, 2, ((4, 4), (2, 2))),
            (5, 5, 3, ((8, 8), (4, 4), (2, 2))),
            (1, 5, 1, ((1, 5),)),
            (1, 1, 1, ((1, 1),)),
        ],
    )
    def test_minimal_padding_reflect_border_smallest_sides(self, height, width, max_level, level_sizes, device, dtype):
        sample = torch.rand(1, 1, height, width, device=device, dtype=dtype)
        pyramid = kornia.geometry.transform.build_laplacian_pyramid(sample, max_level, minimal_padding=True)
        assert [tuple(img.shape[-2:]) for img in pyramid] == list(level_sizes)

    def test_minimal_padding_empty_batch(self, device, dtype):
        sample = torch.rand(0, 3, 5, 8, device=device, dtype=dtype)
        pyramid = kornia.geometry.transform.build_laplacian_pyramid(sample, 2, minimal_padding=True)
        assert [img.shape for img in pyramid] == [torch.Size((0, 3, 6, 8)), torch.Size((0, 3, 3, 4))]

    def test_default_jit_trace(self, device, dtype):
        class Pyramid(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gain = torch.nn.Parameter(torch.tensor(2.0, device=device, dtype=dtype))

            def forward(self, x):
                return kornia.geometry.transform.build_laplacian_pyramid(self.gain * x, 3)

        model = Pyramid()
        sample = torch.rand(1, 1, 64, 96, device=device, dtype=dtype)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", "`build_laplacian_pyramid`", FutureWarning)
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            traced = torch.jit.trace(model, sample, check_trace=False)
            # The default rule cannot pad under a trace: its sizes are tensors there, without `bit_length`. This goes
            # away with the rule (#6319), and `minimal_padding=True` traces these sizes today.
            with pytest.raises(AttributeError, match="bit_length"):
                torch.jit.trace(model, torch.rand(1, 1, 96, 96, device=device, dtype=dtype), check_trace=False)
        for captured, eager in zip(traced(sample), model(sample), strict=True):
            self.assert_close(captured, eager)

    @pytest.mark.parametrize("height,width,max_level", [(96, 96, 3), (5, 8, 2), (64, 96, 3), (480, 640, 1)])
    def test_minimal_padding_jit_trace_with_parameters(self, height, width, max_level, device, dtype):
        # A module whose parameter feeds the op: the traced graph must run with the parameter's dtype and device.
        class Pyramid(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gain = torch.nn.Parameter(torch.tensor(2.0, device=device, dtype=dtype))

            def forward(self, x):
                return kornia.geometry.transform.build_laplacian_pyramid(self.gain * x, max_level, minimal_padding=True)

        sample = torch.rand(1, 1, height, width, device=device, dtype=dtype)
        model = Pyramid()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            traced = torch.jit.trace(model, sample, check_trace=False)
        for captured, eager in zip(traced(sample), model(sample), strict=True):
            self.assert_close(captured, eager)

    def test_convention_rebuilt_crop_keeps_the_input_under_both_padding_rules_6319(self, device, dtype):
        # The warning's promise: a rebuilt image cropped to the input size does not depend on the padding rule, up to
        # float rounding. The levels themselves do, even cropped, as each is a difference with a blur that sees the
        # padding: test_convention_padding_rule_reference_vectors_6319 pins them for this image.
        sample = torch.arange(5, device=device, dtype=dtype).view(1, 1, 5, 1).expand(1, 1, 5, 8).contiguous()
        with pytest.warns(FutureWarning, match="kornia#6319"):
            current = kornia.geometry.transform.build_laplacian_pyramid(sample, 2)
        minimal = kornia.geometry.transform.build_laplacian_pyramid(sample, 2, minimal_padding=True)
        for levels in (current, minimal):
            rebuilt = kornia.geometry.transform.pyrup(levels[1]) + levels[0]
            self.assert_close(rebuilt[..., :5, :8], sample)

    def test_convention_padding_rule_reference_vectors_6319(self, device, dtype):
        # Old-vs-new reference vectors: column 3 of level 0 for a 5x8 image whose rows hold their index, at two levels,
        # padded to 8 rows by the current rule and to 6 by `minimal_padding=True`. Generated by numpy alone, without
        # torch or kornia; the rows are constant along the width, so one column suffices:
        #
        #   import numpy as np
        #   def blur(x):  # [1, 4, 6, 4, 1] / 16 with reflected borders
        #       p = np.pad(x, 2, mode="reflect")
        #       return np.array([p[i : i + 5] @ (np.array([1, 4, 6, 4, 1]) / 16) for i in range(len(x))])
        #   def resample(x, n):  # bilinear, half-pixel centres, a negative source clamped to 0
        #       out = []
        #       for i in range(n):
        #           c = max((i + 0.5) * len(x) / n - 0.5, 0.0)
        #           i0 = min(int(np.floor(c)), len(x) - 1)
        #           out.append(x[i0] * (1 - (c - i0)) + x[min(i0 + 1, len(x) - 1)] * (c - i0))
        #       return np.array(out)
        #   def level0(padded):
        #       down = resample(blur(padded), len(padded) // 2)
        #       return padded - blur(resample(down, 2 * len(down)))
        #   old = level0(np.pad(np.arange(5.0), (0, 3), mode="reflect")).tolist()
        #   new = level0(np.pad(np.arange(5.0), (0, 1), mode="reflect")).tolist()
        sample = torch.arange(5, device=device, dtype=dtype).view(1, 1, 5, 1).expand(1, 1, 5, 8).contiguous()
        with pytest.warns(FutureWarning, match="up to float rounding, but the values of each level change"):
            current = kornia.geometry.transform.build_laplacian_pyramid(sample, 2)
        minimal = kornia.geometry.transform.build_laplacian_pyramid(sample, 2, minimal_padding=True)
        old = [-1.265625, -0.486328125, 0.009765625, 0.529296875, 1.298828125, 0.392578125, -0.33203125, -1.18359375]
        new = [-1.265625, -0.4921875, -0.03125, 0.3828125, 0.9453125, -0.21875]
        for levels, column in ((current, old), (minimal, new)):
            self.assert_close(levels[0][0, 0, :, 3], torch.tensor(column, device=device, dtype=dtype))

    @pytest.mark.parametrize("minimal_padding", [False, True])
    @pytest.mark.parametrize("height,width,max_level", [(5, 8, 2), (96, 96, 3), (64, 96, 3), (480, 640, 1)])
    def test_dynamo(self, height, width, max_level, minimal_padding, device, dtype, torch_optimizer, optimizer_backend):
        if optimizer_backend == "jit":
            pytest.skip("TorchScript cannot compile this closure")
        sample = torch.rand(1, 1, height, width, device=device, dtype=dtype)

        def pyramid(x):
            return kornia.geometry.transform.build_laplacian_pyramid(x, max_level, minimal_padding=minimal_padding)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            expected = pyramid(sample)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", "`build_laplacian_pyramid`", FutureWarning)
            actual = torch_optimizer(pyramid)(sample)
        for actual_level, expected_level in zip(actual, expected, strict=True):
            self.assert_close(actual_level, expected_level)

    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    @pytest.mark.parametrize("dynamic_sizes", [False, True])
    @pytest.mark.parametrize("minimal_padding", [False, True])
    @pytest.mark.parametrize("height,width,max_level", [(5, 8, 2), (96, 96, 3), (64, 96, 3), (480, 640, 1)])
    def test_export_does_not_warn(self, height, width, max_level, minimal_padding, dynamic_sizes, device, dtype):
        # torch 2.5.1 cannot capture `bool` of a symbolic size, which `is_powerof_two` takes, on the base as well.
        if dynamic_sizes and torch_version_lt(2, 14, 0):
            pytest.skip("Dynamic sizes are checked on torch>=2.14: torch 2.5.1 cannot capture this function with them")

        class Pyramid(torch.nn.Module):
            def forward(self, x):
                return kornia.geometry.transform.build_laplacian_pyramid(x, max_level, minimal_padding=minimal_padding)

        sample = torch.rand(1, 1, height, width, device=device, dtype=dtype)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", "`build_laplacian_pyramid`", FutureWarning)
            dynamic_shapes = {"x": {2: torch.export.Dim.AUTO, 3: torch.export.Dim.AUTO}} if dynamic_sizes else None
            program = torch.export.export(Pyramid(), (sample,), dynamic_shapes=dynamic_shapes)
            actual = program.module()(sample)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            expected = Pyramid()(sample)
        for actual_level, expected_level in zip(actual, expected, strict=True):
            self.assert_close(actual_level, expected_level)

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    @pytest.mark.parametrize("channels", [1, 3])
    @pytest.mark.parametrize("max_level", [2, 3, 4])
    def test_num_levels(self, batch_size, channels, max_level, device, dtype):
        height, width = 16, 32
        sample = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        pyramid = kornia.geometry.transform.build_laplacian_pyramid(sample, max_level)
        assert len(pyramid) == max_level
        for i in range(1, max_level):
            img = pyramid[i]
            denom = 2**i
            expected_shape = (batch_size, channels, height // denom, width // denom)
            assert img.shape == expected_shape

    def test_gradcheck(self, device):
        max_level = 1
        batch_size, channels, height, width = 1, 2, 7, 9
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.transform.build_laplacian_pyramid, (img, max_level), nondet_tol=1e-8)

    @pytest.mark.parametrize(
        "height,width,max_level,border_type", [(5, 8, 2, "reflect"), (7, 9, 1, "reflect"), (1, 3, 2, "replicate")]
    )
    def test_gradcheck_minimal_padding(self, height, width, max_level, border_type, device):
        img = torch.rand(1, 2, height, width, device=device, dtype=torch.float64)
        build = kornia.geometry.transform.build_laplacian_pyramid
        self.gradcheck(lambda x: build(x, max_level, border_type, minimal_padding=True), (img,), nondet_tol=1e-8)
