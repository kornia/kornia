import pytest
import torch

from kornia.filters import GuidedBlur, guided_blur
from kornia.testing import BaseTester, tensor_to_gradcheck_var


class TestGuidedBlur(BaseTester):
    @pytest.mark.parametrize("shape", [(1, 1, 8, 15), (2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [5, (3, 5)])
    @pytest.mark.parametrize("eps", [0.1, 0.01])
    def test_smoke(self, shape, kernel_size, eps, device, dtype):
        guide = torch.zeros(shape, device=device, dtype=dtype)
        inp = torch.zeros(shape, device=device, dtype=dtype)

        # tensor eps -> with batch dim
        eps = torch.rand(shape[0], device=device, dtype=dtype)
        actual_A = guided_blur(guide, inp, kernel_size, eps)
        assert isinstance(actual_A, torch.Tensor)
        assert actual_A.shape == shape

        # float and tuple sigmas -> same sigmas across batch
        eps_ = eps[0].item()
        actual_B = guided_blur(guide, inp, kernel_size, eps_)
        assert isinstance(actual_B, torch.Tensor)
        assert actual_B.shape == shape

        self.assert_close(actual_A[0], actual_B[0])

    @pytest.mark.parametrize("shape", [(1, 1, 8, 15), (2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [5, (3, 5)])
    def test_cardinality(self, shape, kernel_size, device, dtype):
        guide = torch.zeros(shape, device=device, dtype=dtype)
        inp = torch.zeros(shape, device=device, dtype=dtype)
        actual = guided_blur(guide, inp, kernel_size, 0.1)
        assert actual.shape == shape

    def test_exception(self):
        pass

    #     with pytest.raises(Exception) as errinfo:
    #         bilateral_blur(torch.rand(1, 1, 5, 5), 3, 1, 1)
    #     assert 'Not a Tensor type. Go' in str(errinfo)

    #     with pytest.raises(ValueError) as errinfo:
    #         bilateral_blur(torch.rand(1, 1, 5, 5), 3, 0.1, (1, 1), color_distance_type="l3")
    #     assert 'color_distance_type only acceps l1 or l2' in str(errinfo)

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        guide = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)
        inp = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        actual = guided_blur(guide, inp, 3, 0.1)
        assert actual.is_contiguous()

    def test_gradcheck(self, device):
        guide = torch.rand(1, 2, 5, 4, device=device)
        img = torch.rand(1, 2, 5, 4, device=device)
        guide = tensor_to_gradcheck_var(guide)  # to var
        img = tensor_to_gradcheck_var(img)
        self.gradcheck(guided_blur, (guide, img, 3, 0.1))

    @pytest.mark.parametrize("shape", [(1, 1, 8, 15), (2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [5, (3, 5)])
    @pytest.mark.parametrize("eps", [0.1, 0.01])
    def test_module(self, shape, kernel_size, eps, device, dtype):
        guide = torch.rand(shape, device=device, dtype=dtype)
        img = torch.rand(shape, device=device, dtype=dtype)

        op = guided_blur
        op_module = GuidedBlur(kernel_size, eps)
        self.assert_close(op_module(guide, img), op(guide, img, kernel_size, eps))

    @pytest.mark.parametrize('kernel_size', [5, (5, 7)])
    def test_dynamo(self, kernel_size, device, dtype, torch_optimizer):
        guide = torch.ones(2, 3, 8, 8, device=device, dtype=dtype)
        inpt = torch.ones(2, 3, 8, 8, device=device, dtype=dtype)
        op = GuidedBlur(kernel_size, 0.1)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(guide, inpt), op_optimized(guide, inpt))

        op = GuidedBlur(kernel_size, torch.tensor(0.1, device=device, dtype=dtype))
        op_optimized = torch_optimizer(op)

        self.assert_close(op(guide, inpt), op_optimized(guide, inpt))

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
        #   eps
        # )
        expected = [
            [0.4487294, 0.5163902, 0.5981981, 0.70094436],
            [0.4850059, 0.53724647, 0.62616897, 0.6686147],
            [0.5010369, 0.5631456, 0.6808387, 0.5960593],
            [0.5304646, 0.53203756, 0.57674146, 0.80308396],
        ]
        expected = torch.tensor(expected, device=device, dtype=dtype).view(1, 1, 4, 4)

        out = guided_blur(guide, img, kernel_size, eps)
        self.assert_close(out, expected)

    def test_opencv_rgb(self, device, dtype):
        img = [
            [[170, 189, 182, 255], [169, 209, 216, 215], [196, 213, 228, 191], [207, 126, 224, 249]],
            [[61, 104, 74, 225], [65, 112, 176, 148], [78, 147, 176, 120], [124, 61, 155, 211]],
            [[73, 111, 90, 175], [77, 117, 163, 130], [83, 139, 163, 120], [132, 84, 137, 155]],
        ]
        img = torch.tensor(img, device=device, dtype=dtype).view(1, 3, 4, 4) / 255

        guide = [
            [[170, 189, 182, 255], [169, 209, 216, 215], [196, 213, 228, 191], [207, 126, 224, 249]],
            [[61, 104, 74, 225], [65, 112, 176, 148], [78, 147, 176, 120], [124, 61, 155, 211]],
            [[73, 111, 90, 175], [77, 117, 163, 130], [83, 139, 163, 120], [132, 84, 137, 155]],
        ]
        guide = torch.tensor(guide, device=device, dtype=dtype).view(1, 3, 4, 4) / 255

        kernel_size = 5
        eps = 0.1

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

        out = guided_blur(img, guide, kernel_size, eps)
        self.assert_close(out, expected, rtol=1e-2, atol=1e-2)
