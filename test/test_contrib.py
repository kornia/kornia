from unittest.mock import patch, PropertyMock

import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.contrib.face_detection import FaceKeypoint
from kornia.testing import assert_close
from packaging import version


class TestVisionTransformer:
    @pytest.mark.parametrize("B", [1, 2])
    @pytest.mark.parametrize("H", [1, 3, 8])
    @pytest.mark.parametrize("D", [128, 768])
    @pytest.mark.parametrize("image_size", [32, 224])
    def test_smoke(self, device, dtype, B, H, D, image_size):
        patch_size = 16
        T = image_size ** 2 // patch_size ** 2 + 1  # tokens size

        img = torch.rand(B, 3, image_size, image_size, device=device, dtype=dtype)
        vit = kornia.contrib.VisionTransformer(image_size=image_size, num_heads=H, embed_dim=D).to(device, dtype)

        out = vit(img)
        assert isinstance(out, torch.Tensor) and out.shape == (B, T, D)

        feats = vit.encoder_results
        assert isinstance(feats, list) and len(feats) == 12
        for f in feats:
            assert f.shape == (B, T, D)

    def test_backbone(self, device, dtype):
        def backbone_mock(x):
            return torch.ones(1, 128, 14, 14, device=device, dtype=dtype)

        img = torch.rand(1, 3, 32, 32, device=device, dtype=dtype)
        vit = kornia.contrib.VisionTransformer(backbone=backbone_mock).to(device, dtype)
        out = vit(img)
        assert out.shape == (1, 197, 128)


class TestMobileViT:
    @pytest.mark.parametrize("B", [1, 2])
    @pytest.mark.parametrize("image_size", [(256, 256)])
    @pytest.mark.parametrize("mode", ['xxs', 'xs', 's'])
    @pytest.mark.parametrize("patch_size", [(2, 2)])
    def test_smoke(self, device, dtype, B, image_size, mode, patch_size):
        ih, iw = image_size
        channel = {'xxs': 320, 'xs': 384, 's': 640}

        img = torch.rand(B, 3, ih, iw, device=device, dtype=dtype)
        mvit = kornia.contrib.MobileViT(mode=mode, patch_size=patch_size).to(device, dtype)

        out = mvit(img)
        assert isinstance(out, torch.Tensor) and out.shape == (B, channel[mode], 8, 8)


class TestClassificationHead:
    @pytest.mark.parametrize("B, D, N", [(1, 8, 10), (2, 2, 5)])
    def test_smoke(self, device, dtype, B, D, N):
        feat = torch.rand(B, D, D, device=device, dtype=dtype)
        head = kornia.contrib.ClassificationHead(embed_size=D, num_classes=N).to(device, dtype)
        logits = head(feat)
        assert logits.shape == (B, N)


class TestConnectedComponents:
    def test_smoke(self, device, dtype):
        img = torch.rand(1, 1, 3, 4, device=device, dtype=dtype)
        out = kornia.contrib.connected_components(img, num_iterations=10)
        assert out.shape == (1, 1, 3, 4)

    @pytest.mark.parametrize("shape", [(1, 3, 4), (2, 1, 3, 4)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.rand(shape, device=device, dtype=dtype)
        out = kornia.contrib.connected_components(img, num_iterations=10)
        assert out.shape == shape

    def test_exception(self, device, dtype):
        img = torch.rand(1, 1, 3, 4, device=device, dtype=dtype)

        with pytest.raises(TypeError):
            assert kornia.contrib.connected_components(img, 1.0)

        with pytest.raises(TypeError):
            assert kornia.contrib.connected_components(img, 0)

        with pytest.raises(ValueError):
            img = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
            assert kornia.contrib.connected_components(img, 2)

    def test_value(self, device, dtype):
        img = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 14.0, 14.0, 0.0, 0.0, 11.0],
                        [0.0, 14.0, 14.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 34.0, 34.0, 0.0],
                        [0.0, 0.0, 0.0, 34.0, 34.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        out = kornia.contrib.connected_components(img, num_iterations=10)
        assert_close(out, expected)

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("1.9"), reason="Tuple cannot be used with PyTorch < v1.9"
    )
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 1, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.contrib.connected_components, (img,), raise_exception=True)

    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.contrib.connected_components
        op_jit = torch.jit.script(op)
        assert_close(op(img), op_jit(img))


class TestExtractTensorPatches:
    def test_smoke(self, device):
        img = torch.arange(16.0, device=device).view(1, 1, 4, 4)
        m = kornia.contrib.ExtractTensorPatches(3)
        assert m(img).shape == (1, 4, 1, 3, 3)

    def test_b1_ch1_h4w4_ws3(self, device):
        img = torch.arange(16.0, device=device).view(1, 1, 4, 4)
        m = kornia.contrib.ExtractTensorPatches(3)
        patches = m(img)
        assert patches.shape == (1, 4, 1, 3, 3)
        assert_close(img[0, :, :3, :3], patches[0, 0])
        assert_close(img[0, :, :3, 1:], patches[0, 1])
        assert_close(img[0, :, 1:, :3], patches[0, 2])
        assert_close(img[0, :, 1:, 1:], patches[0, 3])

    def test_b1_ch2_h4w4_ws3(self, device):
        img = torch.arange(16.0, device=device).view(1, 1, 4, 4)
        img = img.expand(-1, 2, -1, -1)  # copy all channels
        m = kornia.contrib.ExtractTensorPatches(3)
        patches = m(img)
        assert patches.shape == (1, 4, 2, 3, 3)
        assert_close(img[0, :, :3, :3], patches[0, 0])
        assert_close(img[0, :, :3, 1:], patches[0, 1])
        assert_close(img[0, :, 1:, :3], patches[0, 2])
        assert_close(img[0, :, 1:, 1:], patches[0, 3])

    def test_b1_ch1_h4w4_ws2(self, device):
        img = torch.arange(16.0, device=device).view(1, 1, 4, 4)
        m = kornia.contrib.ExtractTensorPatches(2)
        patches = m(img)
        assert patches.shape == (1, 9, 1, 2, 2)
        assert_close(img[0, :, 0:2, 1:3], patches[0, 1])
        assert_close(img[0, :, 0:2, 2:4], patches[0, 2])
        assert_close(img[0, :, 1:3, 1:3], patches[0, 4])
        assert_close(img[0, :, 2:4, 1:3], patches[0, 7])

    def test_b1_ch1_h4w4_ws2_stride2(self, device):
        img = torch.arange(16.0, device=device).view(1, 1, 4, 4)
        m = kornia.contrib.ExtractTensorPatches(2, stride=2)
        patches = m(img)
        assert patches.shape == (1, 4, 1, 2, 2)
        assert_close(img[0, :, 0:2, 0:2], patches[0, 0])
        assert_close(img[0, :, 0:2, 2:4], patches[0, 1])
        assert_close(img[0, :, 2:4, 0:2], patches[0, 2])
        assert_close(img[0, :, 2:4, 2:4], patches[0, 3])

    def test_b1_ch1_h4w4_ws2_stride21(self, device):
        img = torch.arange(16.0, device=device).view(1, 1, 4, 4)
        m = kornia.contrib.ExtractTensorPatches(2, stride=(2, 1))
        patches = m(img)
        assert patches.shape == (1, 6, 1, 2, 2)
        assert_close(img[0, :, 0:2, 1:3], patches[0, 1])
        assert_close(img[0, :, 0:2, 2:4], patches[0, 2])
        assert_close(img[0, :, 2:4, 0:2], patches[0, 3])
        assert_close(img[0, :, 2:4, 2:4], patches[0, 5])

    def test_b1_ch1_h3w3_ws2_stride1_padding1(self, device):
        img = torch.arange(9.0).view(1, 1, 3, 3).to(device)
        m = kornia.contrib.ExtractTensorPatches(2, stride=1, padding=1)
        patches = m(img)
        assert patches.shape == (1, 16, 1, 2, 2)
        assert_close(img[0, :, 0:2, 0:2], patches[0, 5])
        assert_close(img[0, :, 0:2, 1:3], patches[0, 6])
        assert_close(img[0, :, 1:3, 0:2], patches[0, 9])
        assert_close(img[0, :, 1:3, 1:3], patches[0, 10])

    def test_b2_ch1_h3w3_ws2_stride1_padding1(self, device):
        batch_size = 2
        img = torch.arange(9.0).view(1, 1, 3, 3).to(device)
        img = img.expand(batch_size, -1, -1, -1)
        m = kornia.contrib.ExtractTensorPatches(2, stride=1, padding=1)
        patches = m(img)
        assert patches.shape == (batch_size, 16, 1, 2, 2)
        for i in range(batch_size):
            assert_close(img[i, :, 0:2, 0:2], patches[i, 5])
            assert_close(img[i, :, 0:2, 1:3], patches[i, 6])
            assert_close(img[i, :, 1:3, 0:2], patches[i, 9])
            assert_close(img[i, :, 1:3, 1:3], patches[i, 10])

    def test_b1_ch1_h3w3_ws23(self, device):
        img = torch.arange(9.0).view(1, 1, 3, 3).to(device)
        m = kornia.contrib.ExtractTensorPatches((2, 3))
        patches = m(img)
        assert patches.shape == (1, 2, 1, 2, 3)
        assert_close(img[0, :, 0:2, 0:3], patches[0, 0])
        assert_close(img[0, :, 1:3, 0:3], patches[0, 1])

    def test_b1_ch1_h3w4_ws23(self, device):
        img = torch.arange(12.0).view(1, 1, 3, 4).to(device)
        m = kornia.contrib.ExtractTensorPatches((2, 3))
        patches = m(img)
        assert patches.shape == (1, 4, 1, 2, 3)
        assert_close(img[0, :, 0:2, 0:3], patches[0, 0])
        assert_close(img[0, :, 0:2, 1:4], patches[0, 1])
        assert_close(img[0, :, 1:3, 0:3], patches[0, 2])
        assert_close(img[0, :, 1:3, 1:4], patches[0, 3])

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        @torch.jit.script
        def op_script(img: torch.Tensor, height: int, width: int) -> torch.Tensor:
            return kornia.geometry.denormalize_pixel_coordinates(img, height, width)

        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(height, width, normalized_coordinates=True).to(device)

        actual = op_script(grid, height, width)
        expected = kornia.denormalize_pixel_coordinates(grid, height, width)

        assert_close(actual, expected)

    def test_gradcheck(self, device):
        img = torch.rand(2, 3, 4, 4).to(device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.contrib.extract_tensor_patches, (img, 3), raise_exception=True)


class TestCombineTensorPatches:
    def test_smoke(self, device, dtype):
        img = torch.arange(16, device=device, dtype=dtype).view(1, 1, 4, 4)
        m = kornia.contrib.CombineTensorPatches((2, 2))
        patches = kornia.contrib.extract_tensor_patches(img, window_size=(2, 2), stride=(2, 2))
        assert m(patches).shape == (1, 1, 4, 4)
        assert (img == m(patches)).all()

    def test_error(self, device, dtype):
        patches = kornia.contrib.extract_tensor_patches(
            torch.arange(16, device=device, dtype=dtype).view(1, 1, 4, 4), window_size=(2, 2), stride=(2, 2), padding=1
        )
        with pytest.raises(NotImplementedError):
            kornia.contrib.combine_tensor_patches(patches, window_size=(2, 2), stride=(3, 2))

    def test_padding1(self, device, dtype):
        img = torch.arange(16, device=device, dtype=dtype).view(1, 1, 4, 4)
        patches = kornia.contrib.extract_tensor_patches(img, window_size=(2, 2), stride=(2, 2), padding=1)
        m = kornia.contrib.CombineTensorPatches((2, 2), unpadding=1)
        assert m(patches).shape == (1, 1, 4, 4)
        assert (img == m(patches)).all()

    def test_gradcheck(self, device, dtype):
        patches = kornia.contrib.extract_tensor_patches(
            torch.arange(16.0, device=device, dtype=dtype).view(1, 1, 4, 4), window_size=(2, 2), stride=(2, 2)
        )
        img = utils.tensor_to_gradcheck_var(patches)  # to var
        assert gradcheck(kornia.contrib.combine_tensor_patches, (img, (2, 2), (2, 2)), raise_exception=True)


class TestLambdaModule:
    def add_2_layer(self, tensor):
        return tensor + 2

    def add_x_mul_y(self, tensor, x, y=2):
        return torch.mul(tensor + x, y)

    def test_smoke(self, device, dtype):
        B, C, H, W = 1, 3, 4, 5
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        func = self.add_2_layer
        if not callable(func):
            raise TypeError(f"Argument lambd should be callable, got {repr(type(func).__name__)}")
        assert isinstance(kornia.contrib.Lambda(func)(img), torch.Tensor)

    @pytest.mark.parametrize("x", [3, 2, 5])
    def test_lambda_with_arguments(self, x, device, dtype):
        B, C, H, W = 2, 3, 5, 7
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        func = self.add_x_mul_y
        lambda_module = kornia.contrib.Lambda(func)
        out = lambda_module(img, x)
        assert isinstance(out, torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 2, 3), (2, 3, 5, 7)])
    def test_lambda(self, shape, device, dtype):
        B, C, H, W = shape
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        func = kornia.color.bgr_to_grayscale
        lambda_module = kornia.contrib.Lambda(func)
        out = lambda_module(img)
        assert isinstance(out, torch.Tensor)

    def test_gradcheck(self, device, dtype):
        B, C, H, W = 1, 3, 4, 5
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        func = kornia.color.bgr_to_grayscale
        assert gradcheck(kornia.contrib.Lambda(func), (img,), raise_exception=True)


class TestImageStitcher:
    @pytest.mark.parametrize("estimator", ['ransac', 'vanilla'])
    def test_smoke(self, estimator, device, dtype):
        B, C, H, W = 1, 3, 224, 224
        input1 = torch.rand(B, C, H, W, device=device, dtype=dtype)
        input2 = torch.rand(B, C, H, W, device=device, dtype=dtype)
        return_value = {
            "keypoints0": torch.rand((15, 2), device=device, dtype=dtype),
            "keypoints1": torch.rand((15, 2), device=device, dtype=dtype),
            "confidence": torch.rand((15,), device=device, dtype=dtype),
            "batch_indexes": torch.zeros((15,), device=device, dtype=dtype),
        }
        with patch(
            'kornia.contrib.ImageStitcher.on_matcher', new_callable=PropertyMock, return_value=lambda x: return_value
        ):
            # NOTE: This will need to download the pretrained weights.
            # To avoid that, we mock as below
            matcher = kornia.feature.LoFTR(None)
            stitcher = kornia.contrib.ImageStitcher(matcher, estimator=estimator).to(device=device, dtype=dtype)
            out = stitcher(input1, input2)
            assert out.shape[:-1] == torch.Size([1, 3, 224])
            assert out.shape[-1] <= 448

    def test_exception(self, device, dtype):
        B, C, H, W = 1, 3, 224, 224
        input1 = torch.rand(B, C, H, W, device=device, dtype=dtype)
        input2 = torch.rand(B, C, H, W, device=device, dtype=dtype)
        # NOTE: This will need to download the pretrained weights.
        matcher = kornia.feature.LoFTR(None)

        with pytest.raises(NotImplementedError):
            stitcher = kornia.contrib.ImageStitcher(matcher, estimator='random').to(device=device, dtype=dtype)

        stitcher = kornia.contrib.ImageStitcher(matcher).to(device=device, dtype=dtype)
        with pytest.raises(RuntimeError):
            stitcher(input1, input2)


class TestHistMatch:

    def test_interp(self, device, dtype):
        xp = torch.tensor([1, 2, 3], device=device, dtype=dtype)
        fp = torch.tensor([4, 2, 0], device=device, dtype=dtype)
        x = torch.tensor([4, 5, 6], device=device, dtype=dtype)
        x_hat_expected = torch.tensor([-2., -4., -6.], device=device, dtype=dtype)
        x_hat = kornia.contrib.interp(x, xp, fp)
        assert (x_hat_expected == x_hat).all()

    def test_histmatch(self, device, dtype):
        torch.manual_seed(44)
        # generate random value by CPU.
        src = torch.randn(1, 4, 4).to(device=device, dtype=dtype)
        dst = torch.randn(1, 16, 16).to(device=device, dtype=dtype)
        out = kornia.contrib.histogram_matching(src, dst)
        exp = torch.tensor([[
            [0.9356, 0.8270, 1.3687, 0.5640],
            [0.6273, 0.9119, 0.4965, 0.4020],
            [0.4353, 0.1475, 0.3384, 0.2580],
            [0.0606, 0.7531, 0.2139, 0.6932]
        ]], device=device, dtype=dtype)
        assert exp.shape == out.shape
        assert_close(out, exp, rtol=1e-4, atol=1e-4)

    @pytest.mark.skip(reason="not differentiable now.")
    def test_grad(self, device):
        B, C, H, W = 1, 3, 32, 32
        src = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        dst = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.contrib.histogram_matching, (src, dst,), raise_exception=True)


class TestFaceDetection:
    def test_smoke(self, device, dtype):
        assert kornia.contrib.FaceDetector().to(device, dtype) is not None

    def test_valid(self, device, dtype):
        torch.manual_seed(44)
        img = torch.rand(1, 3, 320, 320, device=device, dtype=dtype)
        face_detection = kornia.contrib.FaceDetector().to(device, dtype)
        dets = face_detection(img)
        assert len(dets) == 1

    def test_jit(self, device, dtype):
        op = kornia.contrib.FaceDetector().to(device, dtype)
        op_jit = torch.jit.script(op)
        assert op_jit is not None

    def test_results(self, device, dtype):
        data = torch.tensor([
            0., 0., 100., 200., 10., 10., 20., 10., 10., 50., 100., 50., 150., 10., 0.99,
        ], device=device, dtype=dtype)
        res = kornia.contrib.FaceDetectorResult(data)
        assert res.xmin == 0.
        assert res.ymin == 0.
        assert res.xmax == 100.
        assert res.ymax == 200.
        assert res.score == 0.99
        assert res.width == 100.
        assert res.height == 200.
        assert res.top_left.tolist() == [0., 0.]
        assert res.top_right.tolist() == [100., 0.]
        assert res.bottom_right.tolist() == [100., 200.]
        assert res.bottom_left.tolist() == [0., 200.]
        assert res.get_keypoint(FaceKeypoint.EYE_LEFT).tolist() == [10., 10.]
        assert res.get_keypoint(FaceKeypoint.EYE_RIGHT).tolist() == [20., 10.]
        assert res.get_keypoint(FaceKeypoint.NOSE).tolist() == [10., 50.]
        assert res.get_keypoint(FaceKeypoint.MOUTH_LEFT).tolist() == [100., 50.]
        assert res.get_keypoint(FaceKeypoint.MOUTH_RIGHT).tolist() == [150., 10.]

    def test_results_raise(self, device, dtype):
        data = torch.zeros(14, device=device, dtype=dtype)
        with pytest.raises(ValueError):
            _ = kornia.contrib.FaceDetectorResult(data)
