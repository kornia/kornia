import pytest
import torch

from kornia.filters import BilateralBlur, bilateral_blur
from kornia.testing import BaseTester, tensor_to_gradcheck_var


class TestBilateralBlur(BaseTester):
    @pytest.mark.parametrize("shape", [(1, 1, 8, 15), (2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [5, (3, 5)])
    @pytest.mark.parametrize("sigma_color", [1, 0.1])
    @pytest.mark.parametrize("sigma_space", [(1, 1), (1.5, 1)])
    @pytest.mark.parametrize("color_distance_type", ["l1", "l2"])
    def test_smoke(self, shape, kernel_size, sigma_color, sigma_space, color_distance_type, device, dtype):
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

        sigma_color = torch.rand(inpt.shape[0], device=device, dtype=dtype)
        sigma_space = torch.rand(inpt.shape[0], 2, device=device, dtype=dtype)
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
        # expected = cv2.bilateralFilter(img[0].permute(1, 2, 0).numpy(), 5, 0.1, 0.5)
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
