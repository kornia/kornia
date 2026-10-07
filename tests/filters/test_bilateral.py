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

import math

import pytest
import torch

from kornia.core.exceptions import BaseError
from kornia.filters import (
    BilateralBlur,
    JointBilateralBlur,
    bilateral_blur,
    gaussian_blur2d,
    joint_bilateral_blur,
)

from testing.base import BaseTester, supports_reflect_padding, supports_replicate_padding


class TestBilateralBlur(BaseTester):
    @pytest.mark.parametrize("shape", [(1, 1, 8, 15), (2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [5, (3, 5)])
    @pytest.mark.parametrize("color_distance_type", ["l1", "l2"])
    def test_smoke(self, shape, kernel_size, color_distance_type, device, dtype):
        inp = torch.zeros(shape, device=device, dtype=dtype)

        # tensor sigmas -> with batch dim
        sigma_color = torch.linspace(0.5, 1.0, shape[0], device=device, dtype=dtype)
        sigma_space = torch.stack((sigma_color + 0.5, sigma_color + 1.0), dim=-1)
        actual_A = bilateral_blur(inp, kernel_size, sigma_color, sigma_space, "reflect", color_distance_type)
        assert isinstance(actual_A, torch.Tensor)
        assert actual_A.shape == shape

        # float and tuple sigmas -> same sigmas across batch
        sigma_color_ = sigma_color[0].item()
        sigma_space_ = tuple(sigma_space[0].tolist())
        actual_B = bilateral_blur(inp, kernel_size, sigma_color_, sigma_space_, "reflect", color_distance_type)
        assert isinstance(actual_B, torch.Tensor)
        assert actual_B.shape == shape

        self.assert_close(actual_A[0], actual_B[0])

    @pytest.mark.parametrize("shape", [(1, 1, 8, 15), (2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [5, (3, 5)])
    def test_cardinality(self, shape, kernel_size, device, dtype):
        inp = torch.zeros(shape, device=device, dtype=dtype)
        actual = bilateral_blur(inp, kernel_size, 0.1, (1, 1))
        assert actual.shape == shape

    def test_exception(self):
        from kornia.core.exceptions import TypeCheckError

        with pytest.raises(TypeCheckError) as errinfo:
            bilateral_blur(torch.rand(1, 1, 5, 5), 3, 1, 1)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        with pytest.raises(ValueError) as errinfo:
            bilateral_blur(torch.rand(1, 1, 5, 5), 3, 0.1, (1, 1), color_distance_type="l3")
        assert "color_distance_type only accepts l1 or l2" in str(errinfo)

    @pytest.mark.parametrize("kernel_size", [4, (3, 4), (4, 3), 0, -1])
    def test_exception_kernel_size(self, kernel_size):
        # The window is centred on the pixel, so every entry must be a positive odd integer. The check runs before any
        # padding, and the module constructors run it too.
        from kornia.core.exceptions import BaseError

        image = torch.rand(1, 1, 8, 9)
        calls = (
            lambda: bilateral_blur(image, kernel_size, 0.1, (1.0, 1.0)),
            lambda: joint_bilateral_blur(image, image, kernel_size, 0.1, (1.0, 1.0)),
            lambda: BilateralBlur(kernel_size, 0.1, (1.0, 1.0)),
            lambda: JointBilateralBlur(kernel_size, 0.1, (1.0, 1.0)),
        )
        for call in calls:
            with pytest.raises(BaseError, match="Kernel size must be an odd integer bigger than 0"):
                call()

    @pytest.mark.parametrize(
        "sigma_color",
        [0.0, 0, -0.1, [0.0, 0.0], [0.1, 0.0], [-0.1, 0.1]],
        ids=["float_zero", "int_zero", "negative_float", "zero_tensor", "one_zero_row", "negative_row"],
    )
    def test_convention_sigma_color_must_be_positive_5169(self, sigma_color, device, dtype):
        # The colour kernel divides by sigma_color squared, so a zero entry divides by zero and the sign of a negative
        # one is lost: every entry must be positive. The check names the argument, runs before any padding and covers
        # the joint filter and the modules.
        from kornia.core.exceptions import BaseError

        image = torch.rand(2, 3, 8, 9, device=device, dtype=dtype)
        if isinstance(sigma_color, list):
            sigma_color = torch.tensor(sigma_color, device=device, dtype=dtype)
        calls = (
            lambda: bilateral_blur(image, 3, sigma_color, (1.0, 1.0)),
            lambda: joint_bilateral_blur(image, image, 3, sigma_color, (1.0, 1.0)),
            lambda: BilateralBlur(3, sigma_color, (1.0, 1.0))(image),
            lambda: JointBilateralBlur(3, sigma_color, (1.0, 1.0))(image, image),
        )
        for call in calls:
            with pytest.raises(BaseError, match="sigma_color must be positive"):
                call()

    @pytest.mark.parametrize("name", ["sigma_color", "sigma_space"])
    @pytest.mark.parametrize(
        "batch, rows",
        [(2, 3), (4, 2), (4, 3), (1, 3)],
        ids=["2_vs_3", "4_vs_2", "4_vs_3", "1_vs_3"],
    )
    def test_sigma_batch_mismatch_5430(self, name, batch, rows, device, dtype):
        # A tensor sigma batch that is neither 1 nor the input batch raises at the entry with a message naming the
        # argument and both sizes, not a torch broadcast error. Input batch 1 with 3 sigma rows used to succeed and
        # silently return a batch of 3, so it is covered too. The check covers the joint filter and the modules.
        from kornia.core.exceptions import BaseError

        image = torch.rand(batch, 3, 8, 9, device=device, dtype=dtype)
        sigma_color = 0.5
        sigma_space = (1.0, 1.0)
        if name == "sigma_color":
            sigma_color = torch.linspace(0.5, 1.0, rows, device=device, dtype=dtype)
        else:
            sigma_space = torch.full((rows, 2), 1.5, device=device, dtype=dtype)

        calls = (
            lambda: bilateral_blur(image, 3, sigma_color, sigma_space),
            lambda: joint_bilateral_blur(image, image, 3, sigma_color, sigma_space),
            lambda: BilateralBlur(3, sigma_color, sigma_space)(image),
            lambda: JointBilateralBlur(3, sigma_color, sigma_space)(image, image),
        )
        for call in calls:
            with pytest.raises(BaseError, match=f"{name} batch of {rows} for an input batch of {batch}"):
                call()

    @pytest.mark.parametrize("name", ["sigma_color", "sigma_space"])
    @pytest.mark.parametrize("per_sample", [False, True], ids=["shared", "per_sample"])
    def test_sigma_batch_accepted_5430(self, name, per_sample, device, dtype):
        # A sigma batch of 1 is shared by the input and a batch equal to the input gives each sample its own sigma:
        # both keep working and match filtering every sample alone.
        batch = 4
        rows = batch if per_sample else 1
        image = torch.rand(batch, 3, 8, 9, device=device, dtype=dtype)
        if name == "sigma_color":
            sigma_color = torch.linspace(0.3, 0.9, rows, device=device, dtype=dtype)
            kwargs = {"sigma_color": sigma_color, "sigma_space": (1.2, 1.4)}
        else:
            sigma_space = torch.stack((torch.linspace(0.8, 1.6, rows), torch.linspace(1.0, 2.0, rows)), dim=-1)
            kwargs = {"sigma_color": 0.5, "sigma_space": sigma_space.to(device=device, dtype=dtype)}

        actual = bilateral_blur(image, 3, **kwargs)
        assert actual.shape == image.shape

        for i in range(batch):
            row = i if per_sample else 0
            single = {k: (v[row : row + 1] if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
            self.assert_close(actual[i : i + 1], bilateral_blur(image[i : i + 1], 3, **single))

    @pytest.mark.parametrize("sigma_dtype", [torch.float16, torch.float64])
    def test_tensor_sigma_space_keeps_the_input_dtype_5521(self, sigma_dtype, device, dtype):
        # A tensor sigma_space of another dtype is cast to the input's, as a tensor sigma_color is, so it neither
        # promotes the output nor changes the result of the equivalent float sigma_space. It is built on the CPU, which
        # also has float64, so on another device it is moved to the input's device as well.
        image = torch.rand(2, 3, 8, 9, device=device, dtype=dtype)
        sigma_space = torch.tensor([[1.25, 1.5]], dtype=sigma_dtype)

        actual = bilateral_blur(image, 3, 0.5, sigma_space)
        assert actual.dtype == dtype
        self.assert_close(actual, bilateral_blur(image, 3, 0.5, (1.25, 1.5)))
        assert BilateralBlur(3, 0.5, sigma_space)(image).dtype == dtype

    def test_integer_input_keeps_a_tensor_sigma_space_5521(self, device):
        # Only a floating input casts sigma_space: a uint8 cast would truncate 1.5 to 1. With a huge sigma_color every
        # colour weight is 1 whatever the uint8 differences do (#5155), so the filter is gaussian_blur2d's.
        if device.type not in ("cpu", "mps"):
            pytest.skip("uint8 reflect padding is pinned on the CPU and MPS only")
        image = (torch.arange(63, device=device).view(1, 1, 7, 9) * 4).to(torch.uint8)
        sigma_space = torch.tensor([[1.5, 1.5]], device=device)

        actual = bilateral_blur(image, 3, 1e6, sigma_space)
        self.assert_close(actual, gaussian_blur2d(image.float(), 3, sigma_space))

    @pytest.mark.parametrize("shape", [(), (2,), (1, 1)], ids=["0d", "1d", "one_column"])
    def test_sigma_space_shape_is_checked_before_its_batch_5430(self, shape, device, dtype):
        # The (B, 2) shape check runs before the batch check: a 0-d sigma_space has no batch to read, and a 1-D pair
        # on a batch-1 input would otherwise be reported as a sigma_space batch of 2.
        from kornia.core.exceptions import ShapeError

        image = torch.ones(1, 1, 8, 9, device=device, dtype=dtype)
        sigma_space = torch.full(shape, 1.5, device=device, dtype=dtype)
        with pytest.raises(ShapeError):
            bilateral_blur(image, 3, 0.5, sigma_space)

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        actual = bilateral_blur(inp, 3, 1, (1, 1))
        assert actual.is_contiguous()

    def test_gradcheck(self, device):
        img = torch.rand(1, 2, 5, 4, device=device, dtype=torch.float64)
        sigma_color = torch.rand(1, device=device, dtype=torch.float64)
        sigma_space = torch.rand(1, 2, device=device, dtype=torch.float64)

        self.gradcheck(bilateral_blur, (img, 3, 1, (1, 1)), nondet_tol=1e-4)
        self.gradcheck(bilateral_blur, (img, 3, sigma_color, (1, 1)), nondet_tol=1e-4)
        self.gradcheck(bilateral_blur, (img, 3, 1, sigma_space), nondet_tol=1e-4)
        self.gradcheck(bilateral_blur, (img, 3, sigma_color, sigma_space), nondet_tol=1e-4)

    @pytest.mark.parametrize("shape", [(1, 1, 8, 15), (2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [5, (3, 5)])
    @pytest.mark.parametrize("sigma_color", [1, 0.1])
    @pytest.mark.parametrize("sigma_space", [(1, 1), (1.5, 1)])
    @pytest.mark.parametrize("color_distance_type", ["l1", "l2"])
    def test_module(self, shape, kernel_size, sigma_color, sigma_space, color_distance_type, device, dtype):
        img = torch.rand(shape, device=device, dtype=dtype)
        params = (kernel_size, sigma_color, sigma_space, "reflect", color_distance_type)

        op = bilateral_blur
        op_module = BilateralBlur(*params)
        self.assert_close(op_module(img), op(img, *params))

    @pytest.mark.parametrize("kernel_size", [5, (5, 7)])
    @pytest.mark.parametrize("color_distance_type", ["l1", "l2"])
    def test_dynamo(self, kernel_size, color_distance_type, device, dtype, torch_optimizer):
        data = torch.ones(2, 3, 8, 8, device=device, dtype=dtype)
        op = BilateralBlur(kernel_size, 1, (1, 1), color_distance_type=color_distance_type)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data), op_optimized(data))

        sigma_color = torch.rand(1, device=device, dtype=dtype)
        sigma_space = torch.rand(1, 2, device=device, dtype=dtype)
        op = BilateralBlur(kernel_size, sigma_color, sigma_space, color_distance_type=color_distance_type)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data), op_optimized(data))

    def test_dynamo_tensor_sigma_color_fullgraph_5169(self, device, dtype, torch_optimizer):
        """The data-dependent sigma_color check is skipped under compile, so a tensor sigma_color stays one graph."""
        data = torch.rand(2, 3, 8, 8, device=device, dtype=dtype)
        op = BilateralBlur(3, torch.tensor([0.3, 0.7], device=device, dtype=dtype), (1.0, 1.0))
        op_optimized = torch_optimizer(op, fullgraph=True)
        self.assert_close(op_optimized(data), op(data))

    @pytest.mark.parametrize("per_sample", [False, True], ids=["shared_sigma", "per_sample_sigma"])
    def test_dynamo_sigma_batch_check_is_dynamic_5430(self, per_sample, device, dtype, torch_optimizer):
        """The sigma batch check does not specialize the batch: one dynamic graph serves every batch (#5430)."""
        from torch._dynamo.testing import CompileCounter

        def op(x, sigma_color, sigma_space):
            return bilateral_blur(x, 3, sigma_color, sigma_space, "constant")

        counter = CompileCounter()
        compiled = torch_optimizer(op, backend=counter, fullgraph=True, dynamic=True)
        # no batch equals another axis, including sigma_space's 2 columns, so duck sizing cannot tie the batch to it
        for batch in (4, 5, 6):
            rows = batch if per_sample else 1
            image = torch.rand(batch, 3, 7, 9, device=device, dtype=dtype)
            sigma_color = torch.rand(rows, device=device, dtype=dtype) + 0.5
            sigma_space = torch.rand(rows, 2, device=device, dtype=dtype) + 0.5
            self.assert_close(compiled(image, sigma_color, sigma_space), op(image, sigma_color, sigma_space))
        assert counter.frame_count == 1

    def test_opencv_grayscale(self, device, dtype):
        img = [[95, 130, 108, 228], [98, 142, 187, 166], [114, 166, 190, 141], [150, 83, 174, 216]]
        img = torch.tensor(img, device=device, dtype=dtype).view(1, 1, 4, 4) / 255

        kernel_size = 5
        sigma_color = 0.1
        sigma_distance = (0.5, 0.5)

        # Expected output generated with OpenCV:
        # import cv2
        # expected = cv2.bilateralFilter(img[0, 0].numpy(), 5, 0.1, 0.5)
        expected = [
            [0.38708255, 0.5060622, 0.43372786, 0.8876763],
            [0.39813757, 0.55695623, 0.72320986, 0.6593296],
            [0.4527661, 0.6484203, 0.7295754, 0.5705908],
            [0.5774919, 0.32919288, 0.6949335, 0.83184093],
        ]
        expected = torch.tensor(expected, device=device, dtype=dtype).view(1, 1, 4, 4)

        out = bilateral_blur(img, kernel_size, sigma_color, sigma_distance)
        self.assert_close(out, expected, rtol=1e-2, atol=1e-2)

    def test_opencv_rgb(self, device, dtype):
        img = [
            [[170, 189, 182, 255], [169, 209, 216, 215], [196, 213, 228, 191], [207, 126, 224, 249]],
            [[61, 104, 74, 225], [65, 112, 176, 148], [78, 147, 176, 120], [124, 61, 155, 211]],
            [[73, 111, 90, 175], [77, 117, 163, 130], [83, 139, 163, 120], [132, 84, 137, 155]],
        ]
        img = torch.tensor(img, device=device, dtype=dtype).view(1, 3, 4, 4) / 255

        kernel_size = 5
        sigma_color = 0.1
        sigma_distance = (0.5, 0.5)

        # Expected output generated with OpenCV:
        # import cv2
        # expected = cv2.bilateralFilter(img[0].permute(1, 2, 0).numpy(), 5, 0.1, 0.5)
        expected = [
            [
                [0.6658919, 0.7486991, 0.7140039, 0.9999949],
                [0.6656203, 0.815614, 0.852062, 0.84256846],
                [0.7658699, 0.83580506, 0.88873357, 0.7496973],
                [0.8123873, 0.49414372, 0.87789816, 0.97619873],
            ],
            [
                [0.24242306, 0.40987095, 0.29138556, 0.8823465],
                [0.2543548, 0.43856043, 0.68934506, 0.58119816],
                [0.3045888, 0.5758538, 0.6885629, 0.47137713],
                [0.48865014, 0.23922202, 0.6099074, 0.82698977],
            ],
            [
                [0.28948042, 0.43686634, 0.35377124, 0.686273],
                [0.30078027, 0.4582056, 0.63826954, 0.5113827],
                [0.32491508, 0.5446742, 0.63721484, 0.47087318],
                [0.51836246, 0.32941142, 0.5409612, 0.60793126],
            ],
        ]
        expected = torch.tensor(expected, device=device, dtype=dtype).view(1, 3, 4, 4)

        out = bilateral_blur(img, kernel_size, sigma_color, sigma_distance)
        self.assert_close(out, expected, rtol=1e-2, atol=1e-2)


class TestJointBilateralBlur(BaseTester):
    @pytest.mark.parametrize("input_depth", [1, 3])
    @pytest.mark.parametrize("guidance_depth", [1, 3])
    def test_smoke(self, input_depth, guidance_depth, device, dtype):
        b, h, w = 2, 8, 15
        kernel_size = 5
        sigma_color = 0.1
        sigma_space = (2, 2)
        inp = torch.rand(b, input_depth, h, w, device=device, dtype=dtype)
        guide = torch.rand(b, guidance_depth, h, w, device=device, dtype=dtype)

        out = joint_bilateral_blur(inp, guide, kernel_size, sigma_color, sigma_space)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (b, input_depth, h, w)

    def test_same_input(self, device, dtype):
        shape = (2, 3, 8, 15)
        kernel_size = 5
        sigma_color = 0.1
        sigma_space = (2, 2)
        inp = torch.rand(shape, device=device, dtype=dtype)

        out1 = joint_bilateral_blur(inp, inp, kernel_size, sigma_color, sigma_space)
        out2 = bilateral_blur(inp, kernel_size, sigma_color, sigma_space)
        self.assert_close(out1, out2)

    @pytest.mark.parametrize("shape", [(1, 1, 8, 15), (2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [5, (3, 5)])
    def test_cardinality(self, shape, kernel_size, device, dtype):
        inp = torch.zeros(shape, device=device, dtype=dtype)
        guide = torch.zeros(shape, device=device, dtype=dtype)
        actual = joint_bilateral_blur(inp, guide, kernel_size, 0.1, (1, 1))
        assert actual.shape == shape

    def test_exception(self):
        inp = torch.rand(1, 1, 5, 5)
        guide = torch.rand(1, 1, 5, 5)

        from kornia.core.exceptions import TypeCheckError

        with pytest.raises(TypeCheckError) as errinfo:
            joint_bilateral_blur(inp, guide, 3, 1, 1)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        with pytest.raises(Exception) as errinfo:
            joint_bilateral_blur(inp, torch.randn(1, 1, 2, 4), 3, 1, (1, 1))
        assert "guidance and input should have the same" in str(errinfo)

        with pytest.raises(Exception) as errinfo:
            joint_bilateral_blur(inp, torch.randn(2, 1, 5, 5), 3, 1, (1, 1))
        assert "guidance and input should have the same" in str(errinfo)

        with pytest.raises(ValueError) as errinfo:
            joint_bilateral_blur(inp, guide, 3, 0.1, (1, 1), color_distance_type="l3")
        assert "color_distance_type only accepts l1 or l2" in str(errinfo)

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)
        guide = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        actual = joint_bilateral_blur(inp, guide, 3, 1, (1, 1))
        assert actual.is_contiguous()

    def test_gradcheck(self, device):
        img = torch.rand(1, 2, 5, 4, device=device, dtype=torch.float64)
        guide = torch.rand(1, 2, 5, 4, device=device, dtype=torch.float64)
        self.gradcheck(joint_bilateral_blur, (img, guide, 3, 1, (1, 1)), nondet_tol=1e-4)

    def test_module(self, device, dtype):
        shape = (2, 3, 11, 7)
        kernel_size = 5
        sigma_color = 0.1
        sigma_space = (2, 2)
        img = torch.rand(shape, device=device, dtype=dtype)
        guide = torch.rand(shape, device=device, dtype=dtype)
        params = (kernel_size, sigma_color, sigma_space)

        op = joint_bilateral_blur
        op_module = JointBilateralBlur(*params)
        self.assert_close(op_module(img, guide), op(img, guide, *params))

    @pytest.mark.parametrize("kernel_size", [5, (5, 7)])
    @pytest.mark.parametrize("color_distance_type", ["l1", "l2"])
    def test_dynamo(self, kernel_size, color_distance_type, device, dtype, torch_optimizer):
        data = torch.rand(2, 3, 8, 8, device=device, dtype=dtype)
        guide = torch.rand(2, 3, 8, 8, device=device, dtype=dtype)
        op = JointBilateralBlur(kernel_size, 1, (1, 1), color_distance_type=color_distance_type)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data, guide), op_optimized(data, guide))

        sigma_color = torch.rand(1, device=device, dtype=dtype)
        sigma_space = torch.rand(1, 2, device=device, dtype=dtype)
        op = JointBilateralBlur(kernel_size, sigma_color, sigma_space, color_distance_type=color_distance_type)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data, guide), op_optimized(data, guide))

    def test_opencv_grayscale(self, device, dtype):
        img = [[95, 130, 108, 228], [98, 142, 187, 166], [114, 166, 190, 141], [150, 83, 174, 216]]
        img = torch.tensor(img, device=device, dtype=dtype).view(1, 1, 4, 4) / 255

        guide = [[161, 87, 93, 6], [91, 182, 97, 154], [70, 123, 109, 70], [119, 28, 60, 109]]
        guide = torch.tensor(guide, device=device, dtype=dtype).view(1, 1, 4, 4) / 255

        kernel_size = 5
        sigma_color = 0.1
        sigma_distance = (0.5, 0.5)

        # Expected output generated with OpenCV:
        # import cv2
        # expected = cv2.ximgproc.jointBilateralFilter(
        #   guide.squeeze().numpy(),
        #   img.squeeze().numpy(),
        #   kernel_size,
        #   sigma_color,
        #   sigma_distance[0],
        # )
        expected = [
            [0.38221005, 0.5027215, 0.49131155, 0.8937083],
            [0.3976327, 0.55548316, 0.69680846, 0.65291953],
            [0.44903287, 0.65470666, 0.7295845, 0.5840189],
            [0.5867507, 0.3472942, 0.66494286, 0.81431836],
        ]
        expected = torch.tensor(expected, device=device, dtype=dtype).view(1, 1, 4, 4)

        out = joint_bilateral_blur(img, guide, kernel_size, sigma_color, sigma_distance)
        self.assert_close(out, expected)

    def test_opencv_rgb(self, device, dtype):
        img = [
            [[170, 189, 182, 255], [169, 209, 216, 215], [196, 213, 228, 191], [207, 126, 224, 249]],
            [[61, 104, 74, 225], [65, 112, 176, 148], [78, 147, 176, 120], [124, 61, 155, 211]],
            [[73, 111, 90, 175], [77, 117, 163, 130], [83, 139, 163, 120], [132, 84, 137, 155]],
        ]
        img = torch.tensor(img, device=device, dtype=dtype).view(1, 3, 4, 4) / 255

        guide = [
            [[136, 196, 198, 21], [149, 185, 196, 141], [115, 110, 87, 155], [126, 82, 109, 207]],
            [[188, 42, 48, 0], [73, 200, 58, 173], [53, 140, 130, 34], [129, 3, 41, 73]],
            [[85, 36, 51, 0], [33, 82, 37, 87], [35, 70, 57, 36], [50, 10, 30, 43]],
        ]
        guide = torch.tensor(guide, device=device, dtype=dtype).view(1, 3, 4, 4) / 255

        kernel_size = 5
        sigma_color = 0.1
        sigma_distance = (0.5, 0.5)

        # Expected output generated with OpenCV:
        # import cv2
        # expected = cv2.ximgproc.jointBilateralFilter(
        #   guide.squeeze().permute(1, 2, 0).numpy(),
        #   img.squeeze().permute(1, 2, 0).numpy(),
        #   kernel_size,
        #   sigma_color,
        #   sigma_distance[0],
        # ).transpose(2, 0, 1)
        expected = [
            [
                [0.6671455, 0.74172455, 0.7328562, 1.0],
                [0.66403687, 0.81948805, 0.8357967, 0.8431371],
                [0.7673652, 0.836736, 0.8925936, 0.7494889],
                [0.8120746, 0.49431974, 0.8778994, 0.9764218],
            ],
            [
                [0.23984201, 0.40574068, 0.35012922, 0.88235295],
                [0.2555589, 0.43905944, 0.65692204, 0.58039117],
                [0.30529743, 0.5791127, 0.68724936, 0.47124338],
                [0.48745763, 0.23940866, 0.6072882, 0.82737976],
            ],
            [
                [0.2868149, 0.43398118, 0.39570162, 0.6862745],
                [0.30228183, 0.45868847, 0.6153678, 0.50980365],
                [0.3252297, 0.5474381, 0.6367775, 0.47098273],
                [0.51799667, 0.32952034, 0.53696644, 0.6078228],
            ],
        ]
        expected = torch.tensor(expected, device=device, dtype=dtype).view(1, 3, 4, 4)

        out = joint_bilateral_blur(img, guide, kernel_size, sigma_color, sigma_distance)
        self.assert_close(out, expected)

    def test_wider_guidance_keeps_the_input_dtype_5521(self, device, dtype):
        # A floating guidance is cast to the input's dtype, so a wider one does not promote the output.
        if device.type == "mps":
            pytest.skip("MPS has no float64")
        image = torch.rand(2, 3, 8, 9, device=device, dtype=dtype)
        guidance = torch.rand(2, 1, 8, 9, device=device, dtype=torch.float64)

        actual = joint_bilateral_blur(image, guidance, 3, 0.5, (1.5, 1.5))
        assert actual.dtype == dtype
        self.assert_close(actual, joint_bilateral_blur(image, guidance.to(dtype), 3, 0.5, (1.5, 1.5)))
        assert JointBilateralBlur(3, 0.5, (1.5, 1.5))(image, guidance).dtype == dtype

    def test_integer_input_keeps_a_floating_guidance_5521(self, device):
        # Only a floating input casts guidance: a uint8 cast would truncate a [0, 1] guidance to zeros. The guidance is
        # what gets differenced, so a uint8 input filters as its float copy does. A tuple sigma_space would build the
        # spatial kernel in uint8 (#5155); a float tensor one keeps it in floating point.
        if device.type not in ("cpu", "mps"):
            pytest.skip("uint8 reflect padding is pinned on the CPU and MPS only")
        image = (torch.arange(63, device=device).view(1, 1, 7, 9) * 4).to(torch.uint8)
        guidance = torch.linspace(0, 1, 63, device=device).view(1, 1, 7, 9)
        sigma_space = torch.tensor([[1.5, 1.5]], device=device)

        actual = joint_bilateral_blur(image, guidance, 3, 0.1, sigma_space)
        self.assert_close(actual, joint_bilateral_blur(image.float(), guidance, 3, 0.1, sigma_space))


class TestConventionsBilateralBlur(BaseTester):
    """Pins for the colour and space parameters of :func:`bilateral_blur` and :func:`joint_bilateral_blur`."""

    @staticmethod
    def _skip_without_reflect_padding(device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip(f"this torch build has no reflect padding kernel for {dtype} on {device.type}")

    def test_convention_bilateral_blur_sigma_color_is_in_input_units(self, device, dtype):
        # sigma_color is compared with differences of input values, not with a normalised range: scaling the image by
        # s needs sigma_color * s for the same (scaled) result. A power of two keeps the rescaling exact in every dtype.
        self._skip_without_reflect_padding(device, dtype)
        torch.manual_seed(0)
        image = torch.rand(2, 3, 9, 13).to(device=device, dtype=dtype)
        for distance in ("l1", "l2"):
            reference = bilateral_blur(image, 3, 0.1, (1.0, 1.5), color_distance_type=distance)
            rescaled = bilateral_blur(16 * image, 3, 16 * 0.1, (1.0, 1.5), color_distance_type=distance)
            self.assert_close(rescaled, 16 * reference)
            # control: the unscaled sigma_color on the rescaled image is a different filter
            unscaled = bilateral_blur(16 * image, 3, 0.1, (1.0, 1.5), color_distance_type=distance)
            assert (unscaled - 16 * reference).abs().max() > 0.1

    @pytest.mark.parametrize("border_type", ["reflect", "replicate", "constant", "circular"])
    def test_convention_bilateral_blur_sigma_space_is_gaussian_blur2d_sigma(self, border_type, device, dtype):
        # sigma_space is gaussian_blur2d's (sigma_y, sigma_x) with the same kernel_size, and border_type pads as it
        # does: once sigma_color dwarfs every intensity difference the colour weights are 1 and the bilateral filter
        # is gaussian_blur2d.
        if border_type == "reflect":
            self._skip_without_reflect_padding(device, dtype)
        if border_type == "replicate" and not supports_replicate_padding(device, dtype):
            pytest.skip(f"this torch build has no replicate padding kernel for {dtype} on {device.type}")
        image = torch.zeros(1, 1, 23, 31, device=device, dtype=dtype)
        image[0, 0, 9, 17] = 1.0
        out = bilateral_blur(image, (15, 21), 1e6, (1.0, 3.0), border_type)
        self.assert_close(out, gaussian_blur2d(image, (15, 21), (1.0, 3.0), border_type))
        # control: the swapped sigma is a different blur of this delta
        assert (out - gaussian_blur2d(image, (15, 21), (3.0, 1.0), border_type)).abs().max() > 0.02

        # A delta one pixel from the top and right edges: the window reaches the padding.
        corner = torch.zeros(1, 1, 9, 12, device=device, dtype=dtype)
        corner[0, 0, 1, 10] = 1.0
        out = bilateral_blur(corner, (5, 7), 1e6, (1.0, 2.0), border_type)
        self.assert_close(out, gaussian_blur2d(corner, (5, 7), (1.0, 2.0), border_type))

    @pytest.mark.parametrize("distance", ["l1", "l2"])
    def test_convention_bilateral_blur_color_weight(self, distance, device, dtype):
        # A neighbour's weight is its spatial Gaussian weight times exp(-d^2 / (2 sigma_color^2)), normalised over the
        # window, with d = sum_c |dI_c| for 'l1' (OpenCV's colour distance) and d^2 = sum_c dI_c^2 for 'l2'.
        # One 1 x 3 window on a two-channel 1 x 3 image: the centre pixel is recomputed here from that formula.
        self._skip_without_reflect_padding(device, dtype)
        rows = [[0.2, 0.2, 0.9], [0.8, 0.8, 0.1]]  # the right neighbour differs by (0.7, -0.7), the left by nothing
        image = torch.tensor(rows, device=device, dtype=dtype).view(1, 2, 1, 3)
        sigma_color, sigma_x = 0.8, 1.2

        def centre(dist):
            weights = []
            for col in range(3):
                diff = [row[col] - row[1] for row in rows]
                d2 = sum(abs(v) for v in diff) ** 2 if dist == "l1" else sum(v * v for v in diff)
                weights.append(math.exp(-((col - 1) ** 2) / (2 * sigma_x**2)) * math.exp(-d2 / (2 * sigma_color**2)))
            return [sum(w * v for w, v in zip(weights, row)) / sum(weights) for row in rows]

        # the fixture tells the two distances apart well beyond the half-precision tolerances
        assert max(abs(a - b) for a, b in zip(centre("l1"), centre("l2"))) > 0.05
        out = bilateral_blur(image, (1, 3), sigma_color, (1.0, sigma_x), color_distance_type=distance)
        self.assert_close(out[0, :, 0, 1], torch.tensor(centre(distance), device=device, dtype=dtype))

    def test_convention_joint_bilateral_blur_filters_its_first_argument(self, device, dtype):
        # joint_bilateral_blur(input, guidance, ...) filters input with colour weights taken from guidance -- the
        # guidance second, unlike guided_blur(guidance, input, ...) -- and the guidance may have its own channel
        # count. A flat guidance makes every colour weight 1, so the output is gaussian_blur2d of the input, with a
        # one-channel guidance and with one that has the input's three channels.
        self._skip_without_reflect_padding(device, dtype)
        torch.manual_seed(0)
        image = torch.rand(2, 3, 9, 13).to(device=device, dtype=dtype)
        expected = gaussian_blur2d(image, (3, 5), (1.0, 2.0))
        for guidance_channels in (1, 3):
            flat = torch.full((2, guidance_channels, 9, 13), 0.5, device=device, dtype=dtype)
            for out in (
                joint_bilateral_blur(image, flat, (3, 5), 0.1, (1.0, 2.0)),
                JointBilateralBlur((3, 5), 0.1, (1.0, 2.0))(image, flat),
            ):
                assert out.shape == image.shape
                self.assert_close(out, expected)

    def test_wart_bilateral_blur_integer_image_wraps_its_differences_5155(self, device):
        """bilateral_blur subtracts uint8 values in uint8, so 10 - 250 wraps and blends edges it should keep (#5155)."""
        if device.type not in ("cpu", "mps"):
            pytest.skip("#5155 is pinned on the CPU and MPS only")
        # Columns alternate 10 and 250. A colour sigma of 50 keeps them apart, but the wrapped difference
        # 10 - 250 = 16 (mod 256) makes the 10s look close to the 250s, which come back near 162.
        # Snippet used to generate expected:
        #   s = torch.tensor([10, 250], dtype=torch.uint8).repeat(8, 5)[None, None]
        #   print(bilateral_blur(s, 3, 50.0, (1.0, 1.0))[0, 0, 3, :4])  # [10.0, 162.3, 10.0, 162.3]
        stripes = torch.tensor([10, 250], device=device, dtype=torch.uint8).repeat(8, 5)[None, None]
        out = bilateral_blur(stripes, 3, 50.0, (1.0, 1.0))
        assert out[0, 0, :, 1::2].max() < 200
        # the same filter on the same values in floating point keeps the stripes
        self.assert_close(bilateral_blur(stripes.float(), 3, 50.0, (1.0, 1.0)), stripes.float(), rtol=0.0, atol=0.01)

    def test_convention_bilateral_blur_wider_sigma_space_keeps_the_input_dtype_5521(self, device, dtype):
        """A tensor sigma_space or guidance of a wider dtype is cast to the input's, as sigma_color is (#5521)."""
        if device.type == "mps":
            pytest.skip("MPS has no float64")
        if dtype == torch.float64:
            pytest.skip("nothing is wider than a float64 input")
        self._skip_without_reflect_padding(device, dtype)
        image = torch.rand(1, 1, 7, 9, device=device).to(dtype)
        wide_space = torch.full((1, 2), 1.5, device=device, dtype=torch.float64)
        assert bilateral_blur(image, 3, 0.1, wide_space).dtype == dtype
        assert joint_bilateral_blur(image, image, 3, 0.1, wide_space).dtype == dtype
        assert joint_bilateral_blur(image, image.double(), 3, 0.1, (1.5, 1.5)).dtype == dtype
        wide_color = torch.tensor([0.1], device=device, dtype=torch.float64)
        assert bilateral_blur(image, 3, wide_color, (1.5, 1.5)).dtype == dtype

    @pytest.mark.parametrize("kernel_size", [4, (3, 4)])
    def test_convention_bilateral_blur_even_kernel_size_is_rejected_up_front_5163(self, kernel_size, device, dtype):
        """The bilateral filters reject an even kernel_size with a kornia error, the modules at construction (#5163)."""
        image = torch.rand(1, 1, 8, 9, device=device, dtype=dtype)
        with pytest.raises(BaseError):
            bilateral_blur(image, kernel_size, 0.1, (1.0, 1.0))
        with pytest.raises(BaseError):
            joint_bilateral_blur(image, image, kernel_size, 0.1, (1.0, 1.0))
        for module in (BilateralBlur, JointBilateralBlur):
            with pytest.raises(BaseError):
                module(kernel_size, 0.1, (1.0, 1.0))

    def test_convention_bilateral_blur_sigma_color_must_be_positive_5169(self, device, dtype):
        """The bilateral filters reject a sigma_color that is not positive, a float or any entry of a tensor (#5169)."""
        image = torch.rand(2, 3, 9, 13, device=device, dtype=dtype)
        for sigma_color in (
            torch.zeros(2, device=device, dtype=dtype),
            torch.tensor([0.1, -0.1], device=device, dtype=dtype),
            0.0,
            -0.1,
        ):
            with pytest.raises(BaseError):
                bilateral_blur(image, 3, sigma_color, (1.0, 1.0))
            with pytest.raises(BaseError):
                joint_bilateral_blur(image, image, 3, sigma_color, (1.0, 1.0))
