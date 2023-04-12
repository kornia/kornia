import pytest
import torch

from kornia.filters import BilateralBlur, JointBilateralBlur, bilateral_blur, joint_bilateral_blur
from kornia.testing import BaseTester, tensor_to_gradcheck_var


class TestBilateralBlur(BaseTester):
    @pytest.mark.parametrize("shape", [(1, 1, 8, 15), (2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [5, (3, 5)])
    @pytest.mark.parametrize("color_distance_type", ["l1", "l2"])
    def test_smoke(self, shape, kernel_size, color_distance_type, device, dtype):
        inp = torch.zeros(shape, device=device, dtype=dtype)

        # tensor sigmas -> with batch dim
        sigma_color = torch.rand(shape[0], device=device, dtype=dtype)
        sigma_space = torch.rand(shape[0], 2, device=device, dtype=dtype)
        actual_A = bilateral_blur(inp, kernel_size, sigma_color, sigma_space, "reflect", color_distance_type)
        assert isinstance(actual_A, torch.Tensor)
        assert actual_A.shape == shape

        # float and tuple sigmas -> same sigmas across batch
        sigma_color_ = sigma_color[0].item()
        sigma_space_ = tuple(sigma_space[0].cpu().numpy())
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
        with pytest.raises(Exception) as errinfo:
            bilateral_blur(torch.rand(1, 1, 5, 5), 3, 1, 1)
        assert 'Not a Tensor type. Go' in str(errinfo)

        with pytest.raises(ValueError) as errinfo:
            bilateral_blur(torch.rand(1, 1, 5, 5), 3, 0.1, (1, 1), color_distance_type="l3")
        assert 'color_distance_type only acceps l1 or l2' in str(errinfo)

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        actual = bilateral_blur(inp, 3, 1, (1, 1))
        assert actual.is_contiguous()

    def test_gradcheck(self, device):
        img = torch.rand(1, 2, 5, 4, device=device)
        img = tensor_to_gradcheck_var(img)  # to var
        self.gradcheck(bilateral_blur, (img, 3, 1, (1, 1)))

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

    @pytest.mark.parametrize('kernel_size', [5, (5, 7)])
    @pytest.mark.parametrize('color_distance_type', ["l1", "l2"])
    def test_dynamo(self, kernel_size, color_distance_type, device, dtype, torch_optimizer):
        inpt = torch.ones(2, 3, 8, 8, device=device, dtype=dtype)
        op = BilateralBlur(kernel_size, 1, (1, 1), color_distance_type=color_distance_type)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(inpt), op_optimized(inpt))

        sigma_color = torch.rand(1, device=device, dtype=dtype)
        sigma_space = torch.rand(1, 2, device=device, dtype=dtype)
        op = BilateralBlur(kernel_size, sigma_color, sigma_space, color_distance_type=color_distance_type)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(inpt), op_optimized(inpt))

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

        with pytest.raises(Exception) as errinfo:
            joint_bilateral_blur(inp, guide, 3, 1, 1)
        assert 'Not a Tensor type. Go' in str(errinfo)

        with pytest.raises(Exception) as errinfo:
            joint_bilateral_blur(inp, torch.randn(1, 1, 2, 4), 3, 1, (1, 1))
        assert 'guidance and input should have the same' in str(errinfo)

        with pytest.raises(Exception) as errinfo:
            joint_bilateral_blur(inp, torch.randn(2, 1, 5, 5), 3, 1, (1, 1))
        assert 'guidance and input should have the same' in str(errinfo)

        with pytest.raises(ValueError) as errinfo:
            joint_bilateral_blur(inp, guide, 3, 0.1, (1, 1), color_distance_type="l3")
        assert 'color_distance_type only acceps l1 or l2' in str(errinfo)

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)
        guide = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        actual = joint_bilateral_blur(inp, guide, 3, 1, (1, 1))
        assert actual.is_contiguous()

    def test_gradcheck(self, device):
        img = torch.rand(1, 2, 5, 4, device=device)
        guide = torch.rand(1, 2, 5, 4, device=device)
        img = tensor_to_gradcheck_var(img)  # to var
        guide = tensor_to_gradcheck_var(guide)
        self.gradcheck(joint_bilateral_blur, (img, guide, 3, 1, (1, 1)))

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

    @pytest.mark.parametrize('kernel_size', [5, (5, 7)])
    @pytest.mark.parametrize('color_distance_type', ["l1", "l2"])
    def test_dynamo(self, kernel_size, color_distance_type, device, dtype, torch_optimizer):
        inpt = torch.rand(2, 3, 8, 8, device=device, dtype=dtype)
        guide = torch.rand(2, 3, 8, 8, device=device, dtype=dtype)
        op = JointBilateralBlur(kernel_size, 1, (1, 1), color_distance_type=color_distance_type)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(inpt, guide), op_optimized(inpt, guide))

        sigma_color = torch.rand(1, device=device, dtype=dtype)
        sigma_space = torch.rand(1, 2, device=device, dtype=dtype)
        op = JointBilateralBlur(kernel_size, sigma_color, sigma_space, color_distance_type=color_distance_type)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(inpt, guide), op_optimized(inpt, guide))

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
        # expected = cv2.ximgproc.jointBilateralFilter(img[0, 0].numpy(), guide[0, 0].numpy(), 5, 0.1, 0.5)
        expected = [
            [0.38221005, 0.5027215, 0.49131155, 0.8937083],
            [0.3976327, 0.55548316, 0.69680846, 0.65291953],
            [0.44903287, 0.65470666, 0.7295845, 0.5840189],
            [0.5867507, 0.3472942, 0.66494286, 0.81431836],
        ]
        expected = torch.tensor(expected, device=device, dtype=dtype).view(1, 1, 4, 4)

        out = joint_bilateral_blur(img, guide, kernel_size, sigma_color, sigma_distance)
        self.assert_close(out, expected, rtol=1e-2, atol=1e-2)

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
        #     img[0].permute(1, 2, 0).numpy(),
        #     guide[0].permute(1, 2, 0).numpy(),
        #     5, 0.1, 0.5
        # )
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
        self.assert_close(out, expected, rtol=1e-2, atol=1e-2)
