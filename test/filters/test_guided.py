import pytest
import torch

from kornia.filters import GuidedBlur, guided_blur
from kornia.testing import BaseTester, tensor_to_gradcheck_var


class TestGuidedBlur(BaseTester):
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("guide_dim", [1, 3])
    @pytest.mark.parametrize("input_dim", [1, 3])
    @pytest.mark.parametrize("kernel_size", [5, (3, 5)])
    @pytest.mark.parametrize("eps", [0.1, 0.01])
    def test_smoke(self, batch_size, guide_dim, input_dim, kernel_size, eps, device, dtype):
        H, W = 8, 15
        guide = torch.zeros(batch_size, guide_dim, H, W, device=device, dtype=dtype)
        inp = torch.zeros(batch_size, input_dim, H, W, device=device, dtype=dtype)

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
        self.assert_close(out, expected)
