from unittest.mock import PropertyMock, patch

import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.contrib.face_detection import FaceKeypoint
from kornia.testing import assert_close


class TestDiamondSquare:
    def test_smoke(self, device, dtype):
        torch.manual_seed(0)
        output_size = (1, 1, 3, 4)
        roughness = 0.5
        random_scale = 1.0
        out = kornia.contrib.diamond_square(output_size, roughness, random_scale, device=device, dtype=dtype)
        assert out.shape == output_size
        assert out.device == device
        assert out.dtype == dtype

    def test_normalize(self, device, dtype):
        torch.manual_seed(0)
        output_size = (1, 1, 3, 4)
        roughness = 0.5
        random_scale = 1.0
        normalize_range = (0.0, 1.0)
        expected_min = torch.tensor(normalize_range[0], device=device, dtype=dtype)
        expected_max = torch.tensor(normalize_range[1], device=device, dtype=dtype)
        out = kornia.contrib.diamond_square(
            output_size, roughness, random_scale, normalize_range=normalize_range, device=device, dtype=dtype
        )
        assert_close(out.min(), expected_min)
        assert_close(out.max(), expected_max)


class TestKMeans:
    @pytest.mark.parametrize("num_clusters", [3, 10, 1])
    @pytest.mark.parametrize("tolerance", [10e-4, 10e-5, 10e-1])
    @pytest.mark.parametrize("max_iterations", [10, 1000])
    def test_smoke(self, device, dtype, num_clusters, tolerance, max_iterations):
        N = 1000
        D = 2

        kmeans = kornia.contrib.KMeans(num_clusters, None, tolerance, max_iterations, device, 0)
        kmeans.fit(torch.rand((N, D), dtype=dtype))

        out1 = kmeans.get_cluster_assignments()
        out2 = kmeans.get_cluster_centers()

        # output is of type tensor
        assert isinstance(out1, torch.Tensor)
        assert isinstance(out2, torch.Tensor)

        # output is of correct shape
        assert out1.shape == (N,)
        assert out2.shape == (num_clusters, D)

        # output is of same dtype
        assert out1.dtype == torch.int64
        assert out2.dtype == dtype

    def test_exception(self, device, dtype):
        # case: cluster_center = 0:
        with pytest.raises(Exception) as errinfo:
            kornia.contrib.KMeans(0, None, 10e-4, 10, device, 0)
        assert "num_clusters can't be 0" in str(errinfo)

        # case: cluster centers is not a 2D tensor
        with pytest.raises(TypeError) as errinfo:
            starting_centers = torch.rand((2, 3, 5), device=device, dtype=dtype)
            kmeans = kornia.contrib.KMeans(None, starting_centers, 10e-4, 100, device, 0)
        assert "shape must be [[\'C\', \'D\']]. Got torch.Size([2, 3, 5])" in str(errinfo)

        # case: input data is not a 2D tensor
        with pytest.raises(TypeError) as errinfo:
            kmeans = kornia.contrib.KMeans(3, None, 10e-4, 100, device, 0)
            kmeans.fit(torch.rand((1000, 5, 60), dtype=dtype))
        assert "shape must be [[\'N\', \'D\']]. Got torch.Size([1000, 5, 60])" in str(errinfo)

        # case: column dimensions of cluster centers and data to be predicted do not match
        with pytest.raises(Exception) as errinfo:
            kmeans = kornia.contrib.KMeans(3, None, 10e-4, 100, device, 0)
            kmeans.fit(torch.rand((1000, 5), dtype=dtype))
            kmeans.predict(torch.rand((10, 7), dtype=dtype))
        assert "7 != 5" in str(errinfo)

    def test_value(self, device, dtype):
        # create example dataset
        torch.manual_seed(2023)
        x = 5 * torch.randn((500, 2), dtype=dtype) + torch.tensor((-13, 17))
        x = torch.vstack([x, torch.randn((500, 2), dtype=dtype) + torch.tensor((15, -12))])
        x = torch.vstack([x, 13 * torch.randn((500, 2), dtype=dtype) + torch.tensor((35, 15))])

        kmeans = kornia.contrib.KMeans(3, None, 10e-4, 10000, device, 2023)
        kmeans.fit(x)

        centers = kmeans.get_cluster_centers()
        prediciton = kmeans.predict(torch.tensor([[2, 3], [5, 6]]))

        expected_centers = torch.tensor([[37.0849, 16.0393], [-12.3770, 17.2630], [16.0148, -10.9370]], dtype=dtype)
        expected_prediciton = torch.tensor([2, 2], dtype=torch.int64)

        assert_close(centers, expected_centers)
        assert_close(prediciton, expected_prediciton)


class TestVisionTransformer:
    @pytest.mark.parametrize("B", [1, 2])
    @pytest.mark.parametrize("H", [1, 3, 8])
    @pytest.mark.parametrize("D", [128, 768])
    @pytest.mark.parametrize("image_size", [32, 224])
    def test_smoke(self, device, dtype, B, H, D, image_size):
        patch_size = 16
        T = image_size**2 // patch_size**2 + 1  # tokens size

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

    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 1, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.contrib.connected_components, (img,), raise_exception=True, fast_mode=True)

    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.contrib.connected_components
        op_jit = torch.jit.script(op)
        assert_close(op(img), op_jit(img))


def test_compute_padding():
    assert kornia.contrib.compute_padding((6, 6), (2, 2)) == (0, 0, 0, 0)
    assert kornia.contrib.compute_padding((7, 7), (2, 2)) == (1, 0, 1, 0)
    assert kornia.contrib.compute_padding((8, 7), (4, 4)) == (0, 0, 1, 0)


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
        assert gradcheck(kornia.contrib.extract_tensor_patches, (img, 3), raise_exception=True, fast_mode=True)


class TestCombineTensorPatches:
    def test_smoke(self, device, dtype):
        img = torch.arange(16, device=device, dtype=dtype).view(1, 1, 4, 4)
        m = kornia.contrib.CombineTensorPatches((4, 4), (2, 2))
        patches = kornia.contrib.extract_tensor_patches(img, window_size=(2, 2), stride=(2, 2))
        assert m(patches).shape == (1, 1, 4, 4)
        assert_close(img, m(patches))

    def test_error(self, device, dtype):
        patches = kornia.contrib.extract_tensor_patches(
            torch.arange(16, device=device, dtype=dtype).view(1, 1, 4, 4), window_size=(2, 2), stride=(2, 2), padding=1
        )
        with pytest.raises(NotImplementedError):
            kornia.contrib.combine_tensor_patches(patches, original_size=(4, 4), window_size=(2, 2), stride=(3, 2))

    def test_rect_odd_dim(self, device, dtype):
        img = torch.arange(12, device=device, dtype=dtype).view(1, 1, 4, 3)
        patches = kornia.contrib.extract_tensor_patches(img, window_size=(2, 2), stride=(2, 2), padding=(0, 0, 0, 1))
        m = kornia.contrib.combine_tensor_patches(
            patches, original_size=(4, 3), window_size=(2, 2), stride=(2, 2), unpadding=(0, 0, 0, 1)
        )
        assert m.shape == (1, 1, 4, 3)
        assert_close(img, m)

    def test_pad_error(self, device, dtype):
        patches = kornia.contrib.extract_tensor_patches(
            torch.arange(64, device=device, dtype=dtype).view(1, 1, 8, 8), window_size=(4, 4), stride=(4, 4), padding=1
        )
        with pytest.raises(NotImplementedError):
            kornia.contrib.combine_tensor_patches(
                patches, original_size=(8, 8), window_size=(4, 4), stride=(4, 4), unpadding=(1, 1, 1, 1)
            )

    def test_pad_triple_error(self, device, dtype):
        patches = kornia.contrib.extract_tensor_patches(
            torch.arange(36, device=device, dtype=dtype).view(1, 1, 6, 6), window_size=(4, 4), stride=(4, 4), padding=1
        )
        with pytest.raises(AssertionError):
            kornia.contrib.combine_tensor_patches(
                patches, original_size=(6, 6), window_size=(4, 4), stride=(4, 4), unpadding=(1, 1, 1)
            )

    def test_pad_quadruple(self, device, dtype):
        img = torch.arange(36, device=device, dtype=dtype).view(1, 1, 6, 6)
        patches = kornia.contrib.extract_tensor_patches(img, window_size=(4, 4), stride=(4, 4), padding=1)

        merged = kornia.contrib.combine_tensor_patches(
            patches, original_size=(6, 6), window_size=(4, 4), stride=(4, 4), unpadding=(1, 1, 1, 1)
        )
        assert merged.shape == (1, 1, 6, 6)

    def test_rectangle_array(self, device, dtype):
        img = torch.arange(24, device=device, dtype=dtype).view(1, 1, 4, 6)
        patches = kornia.contrib.extract_tensor_patches(img, window_size=(2, 2), stride=(2, 2), padding=1)
        m = kornia.contrib.CombineTensorPatches((4, 6), (2, 2), unpadding=1)
        assert m(patches).shape == (1, 1, 4, 6)
        assert_close(img, m(patches))

    def test_padding1(self, device, dtype):
        img = torch.arange(16, device=device, dtype=dtype).view(1, 1, 4, 4)
        patches = kornia.contrib.extract_tensor_patches(img, window_size=(2, 2), stride=(2, 2), padding=1)
        m = kornia.contrib.CombineTensorPatches((4, 4), (2, 2), unpadding=1)
        assert m(patches).shape == (1, 1, 4, 4)
        assert_close(img, m(patches))

    def test_padding2(self, device, dtype):
        img = torch.arange(64, device=device, dtype=dtype).view(1, 1, 8, 8)
        patches = kornia.contrib.extract_tensor_patches(img, window_size=(2, 2), stride=(2, 2), padding=1)
        m = kornia.contrib.CombineTensorPatches((8, 8), (2, 2), unpadding=1)
        assert m(patches).shape == (1, 1, 8, 8)
        assert_close(img, m(patches))

    def test_autopadding(self, device, dtype):
        img = torch.arange(104, device=device, dtype=dtype).view(1, 1, 8, 13)
        window_size = (3, 3)
        padding = kornia.contrib.compute_padding((8, 13), (3, 3))
        patches = kornia.contrib.extract_tensor_patches(
            img, window_size=window_size, stride=window_size, padding=padding
        )
        m = kornia.contrib.CombineTensorPatches((8, 13), (3, 3), unpadding=padding)
        assert m(patches).shape == (1, 1, 8, 13)
        assert_close(img, m(patches))

    def test_gradcheck(self, device, dtype):
        patches = kornia.contrib.extract_tensor_patches(
            torch.arange(16.0, device=device, dtype=dtype).view(1, 1, 4, 4), window_size=(2, 2), stride=(2, 2)
        )
        img = utils.tensor_to_gradcheck_var(patches)  # to var
        assert gradcheck(
            kornia.contrib.combine_tensor_patches, (img, (4, 4), (2, 2), (2, 2)), raise_exception=True, fast_mode=True
        )


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
        assert gradcheck(kornia.contrib.Lambda(func), (img,), raise_exception=True, fast_mode=True)


class TestImageStitcher:
    @pytest.mark.parametrize("estimator", ['ransac', 'vanilla'])
    def test_smoke(self, estimator, device, dtype):
        B, C, H, W = 1, 3, 6, 6
        sample1 = torch.tensor(
            [
                [0.5349, 0.1988, 0.6592, 0.6569, 0.2328, 0.4251],
                [0.2071, 0.6297, 0.3653, 0.8513, 0.8549, 0.5509],
                [0.2868, 0.2063, 0.4451, 0.3593, 0.7204, 0.0731],
                [0.9699, 0.1078, 0.8829, 0.4132, 0.7572, 0.6948],
                [0.5209, 0.5932, 0.8797, 0.6286, 0.7653, 0.1132],
                [0.8559, 0.6721, 0.6267, 0.5691, 0.7437, 0.9592],
            ],
            dtype=dtype,
            device=device,
        )
        sample2 = torch.tensor(
            [
                [0.3887, 0.2214, 0.3742, 0.1953, 0.7405, 0.2529],
                [0.2332, 0.9314, 0.9575, 0.5575, 0.4134, 0.4355],
                [0.7369, 0.0331, 0.0914, 0.8994, 0.9936, 0.4703],
                [0.1049, 0.5137, 0.2674, 0.4990, 0.7447, 0.7213],
                [0.4414, 0.5550, 0.6361, 0.1081, 0.3305, 0.5196],
                [0.2147, 0.2816, 0.6679, 0.7878, 0.5070, 0.3055],
            ],
            dtype=dtype,
            device=device,
        )
        sample1 = sample1.expand((B, C, H, W))
        sample2 = sample2.expand((B, C, H, W))
        return_value = {
            "keypoints0": torch.tensor(
                [
                    [0.1546, 0.9391],
                    [0.8077, 0.1051],
                    [0.6768, 0.5596],
                    [0.5092, 0.7195],
                    [0.2856, 0.8889],
                    [0.4342, 0.0203],
                    [0.6701, 0.0585],
                    [0.3828, 0.9038],
                    [0.7301, 0.0762],
                    [0.7864, 0.4490],
                    [0.3509, 0.0756],
                    [0.6782, 0.9297],
                    [0.4132, 0.3664],
                    [0.3134, 0.5039],
                    [0.2073, 0.2552],
                ],
                device=device,
                dtype=dtype,
            ),
            "keypoints1": torch.tensor(
                [
                    [0.2076, 0.2669],
                    [0.9679, 0.8137],
                    [0.9536, 0.8317],
                    [0.3718, 0.2456],
                    [0.3875, 0.8450],
                    [0.7592, 0.1687],
                    [0.5173, 0.6760],
                    [0.9446, 0.4570],
                    [0.6164, 0.1867],
                    [0.4732, 0.1786],
                    [0.4090, 0.8089],
                    [0.9742, 0.8943],
                    [0.5996, 0.7427],
                    [0.7038, 0.9210],
                    [0.6272, 0.0796],
                ],
                device=device,
                dtype=dtype,
            ),
            "confidence": torch.tensor(
                [
                    0.9314,
                    0.5951,
                    0.4187,
                    0.0318,
                    0.1434,
                    0.7952,
                    0.8306,
                    0.7511,
                    0.6407,
                    0.7379,
                    0.4363,
                    0.9220,
                    0.8453,
                    0.5075,
                    0.8141,
                ],
                device=device,
                dtype=dtype,
            ),
            "batch_indexes": torch.zeros((15,), device=device, dtype=dtype),
        }
        with patch(
            'kornia.contrib.ImageStitcher.on_matcher', new_callable=PropertyMock, return_value=lambda x: return_value
        ):
            # NOTE: This will need to download the pretrained weights.
            # To avoid that, we mock as below
            matcher = kornia.feature.LoFTR(None)
            stitcher = kornia.contrib.ImageStitcher(matcher, estimator=estimator).to(device=device, dtype=dtype)
            torch.manual_seed(1)  # issue kornia#2027
            out = stitcher(sample1, sample2)
            assert out.shape[:-1] == torch.Size([1, 3, 6])
            assert out.shape[-1] <= 12

    def test_exception(self, device, dtype):
        B, C, H, W = 1, 3, 224, 224
        sample1 = torch.rand(B, C, H, W, device=device, dtype=dtype)
        sample2 = torch.rand(B, C, H, W, device=device, dtype=dtype)
        # NOTE: This will need to download the pretrained weights.
        matcher = kornia.feature.LoFTR(None)

        with pytest.raises(NotImplementedError):
            stitcher = kornia.contrib.ImageStitcher(matcher, estimator='random').to(device=device, dtype=dtype)

        stitcher = kornia.contrib.ImageStitcher(matcher).to(device=device, dtype=dtype)
        with pytest.raises(RuntimeError):
            stitcher(sample1, sample2)


class TestConvDistanceTransform:
    @pytest.mark.parametrize("kernel_size", [3, 5, 7])
    def test_smoke(self, kernel_size, device, dtype):
        sample1 = torch.rand(1, 3, 100, 100, device=device, dtype=dtype)
        sample2 = torch.rand(1, 1, 100, 100, device=device, dtype=dtype)
        distance_transformer = kornia.contrib.DistanceTransform(kernel_size)

        output1 = distance_transformer(sample1)
        output2 = kornia.contrib.distance_transform(sample2, kernel_size)

        assert isinstance(output1, torch.Tensor)
        assert isinstance(output2, torch.Tensor)
        assert output1.shape == sample1.shape

    def test_module(self, device, dtype):
        B, C, H, W = 1, 1, 99, 100
        sample1 = torch.rand(B, C, H, W, device=device, dtype=dtype)
        distance_transformer = kornia.contrib.DistanceTransform().to(device, dtype)

        output1 = distance_transformer(sample1)
        output2 = kornia.contrib.distance_transform(sample1)
        tol_val: float = utils._get_precision(device, dtype)
        assert_close(output1, output2, rtol=tol_val, atol=tol_val)

    def test_exception(self, device, dtype):
        B, C, H, W = 1, 1, 224, 224
        sample1 = torch.rand(B, C, H, W, device=device, dtype=dtype)
        sample2 = torch.rand(C, H, W, device=device, dtype=dtype)

        # Non-odd kernel size
        with pytest.raises(ValueError):
            ConvDT = kornia.contrib.DistanceTransform(6)
            ConvDT.forward(sample1)

        with pytest.raises(ValueError):
            kornia.contrib.distance_transform(sample1, 4)

        # Invalid input dimensions
        with pytest.raises(ValueError):
            kornia.contrib.distance_transform(sample2)

        # Invalid input type
        with pytest.raises(TypeError):
            kornia.contrib.distance_transform(None)

    def test_value(self, device, dtype):
        B, C, H, W = 1, 1, 4, 4
        kernel_size = 7
        h = 0.35
        sample1 = torch.zeros(B, C, H, W, device=device, dtype=dtype)
        sample1[:, :, 1, 1] = 1.0
        expected_output1 = torch.tensor(
            [
                [
                    [
                        [1.4142135382, 1.0000000000, 1.4142135382, 2.2360680103],
                        [1.0000000000, 0.0000000000, 1.0000000000, 2.0000000000],
                        [1.4142135382, 1.0000000000, 1.4142135382, 2.2360680103],
                        [2.2360680103, 2.0000000000, 2.2360680103, 2.8284270763],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        output1 = kornia.contrib.distance_transform(sample1, kernel_size, h)
        assert_close(expected_output1, output1)

    def test_gradcheck(self, device):
        B, C, H, W = 1, 1, 32, 32
        sample1 = torch.ones(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.contrib.distance_transform, (sample1), raise_exception=True, fast_mode=True)

    def test_loss_grad(self, device, dtype):
        B, C, H, W = 1, 1, 32, 32
        sample1 = torch.rand(B, C, H, W, device=device, dtype=dtype, requires_grad=True)
        sample2 = torch.rand(B, C, H, W, device=device, dtype=dtype, requires_grad=True)
        tiny_module = torch.nn.Conv2d(1, 1, (3, 3), (1, 1), (1, 1)).to(device=device, dtype=dtype)
        sample1 = kornia.contrib.distance_transform(tiny_module(sample1))
        sample2 = kornia.contrib.distance_transform(sample2)
        loss = torch.nn.functional.mse_loss(sample1, sample2)
        loss.backward()


class TestHistMatch:
    def test_interp(self, device, dtype):
        xp = torch.tensor([1, 2, 3], device=device, dtype=dtype)
        fp = torch.tensor([4, 2, 0], device=device, dtype=dtype)
        x = torch.tensor([4, 5, 6], device=device, dtype=dtype)
        x_hat_expected = torch.tensor([-2.0, -4.0, -6.0], device=device, dtype=dtype)
        x_hat = kornia.contrib.interp(x, xp, fp)
        assert_close(x_hat_expected, x_hat)

    def test_histmatch(self, device, dtype):
        torch.manual_seed(44)
        # generate random value by CPU.
        src = torch.randn(1, 4, 4).to(device=device, dtype=dtype)
        dst = torch.randn(1, 16, 16).to(device=device, dtype=dtype)
        out = kornia.contrib.histogram_matching(src, dst)
        exp = torch.tensor(
            [
                [
                    [1.5902, 0.9295, 2.9409, 0.1211],
                    [0.2472, 1.2137, -0.1098, -0.4272],
                    [-0.2644, -1.1983, -0.6065, -0.8091],
                    [-1.4999, 0.6370, -0.9800, 0.4474],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        assert exp.shape == out.shape
        assert_close(out, exp, rtol=1e-4, atol=1e-4)

    @pytest.mark.skip(reason="not differentiable now.")
    def test_grad(self, device):
        B, C, H, W = 1, 3, 32, 32
        src = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        dst = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.contrib.histogram_matching, (src, dst), raise_exception=True, fast_mode=True)


class TestFaceDetection:
    def test_smoke(self, device, dtype):
        assert kornia.contrib.FaceDetector().to(device, dtype) is not None

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_valid(self, batch_size, device, dtype):
        torch.manual_seed(44)
        img = torch.rand(batch_size, 3, 320, 320, device=device, dtype=dtype)
        face_detection = kornia.contrib.FaceDetector().to(device, dtype)
        dets = face_detection(img)
        assert isinstance(dets, list)
        assert len(dets) == batch_size  # same as the number of images
        assert isinstance(dets[0], torch.Tensor)
        assert dets[0].shape[0] >= 0  # number of detections
        assert dets[0].shape[1] == 15  # dims of each detection

    def test_jit(self, device, dtype):
        op = kornia.contrib.FaceDetector().to(device, dtype)
        op_jit = torch.jit.script(op)
        assert op_jit is not None

    def test_results(self, device, dtype):
        data = torch.tensor(
            [0.0, 0.0, 100.0, 200.0, 10.0, 10.0, 20.0, 10.0, 10.0, 50.0, 100.0, 50.0, 150.0, 10.0, 0.99],
            device=device,
            dtype=dtype,
        )
        res = kornia.contrib.FaceDetectorResult(data)
        assert res.xmin == 0.0
        assert res.ymin == 0.0
        assert res.xmax == 100.0
        assert res.ymax == 200.0
        assert res.score == 0.99
        assert res.width == 100.0
        assert res.height == 200.0
        assert res.top_left.tolist() == [0.0, 0.0]
        assert res.top_right.tolist() == [100.0, 0.0]
        assert res.bottom_right.tolist() == [100.0, 200.0]
        assert res.bottom_left.tolist() == [0.0, 200.0]
        assert res.get_keypoint(FaceKeypoint.EYE_LEFT).tolist() == [10.0, 10.0]
        assert res.get_keypoint(FaceKeypoint.EYE_RIGHT).tolist() == [20.0, 10.0]
        assert res.get_keypoint(FaceKeypoint.NOSE).tolist() == [10.0, 50.0]
        assert res.get_keypoint(FaceKeypoint.MOUTH_LEFT).tolist() == [100.0, 50.0]
        assert res.get_keypoint(FaceKeypoint.MOUTH_RIGHT).tolist() == [150.0, 10.0]

    def test_results_raise(self, device, dtype):
        data = torch.zeros(14, device=device, dtype=dtype)
        with pytest.raises(ValueError):
            _ = kornia.contrib.FaceDetectorResult(data)


class TestEdgeDetector:
    def test_smoke(self, device, dtype):
        img = torch.rand(2, 3, 64, 64, device=device, dtype=dtype)
        net = kornia.contrib.EdgeDetector().to(device, dtype)
        out = net(img)
        assert out.shape == (2, 1, 64, 64)

    def test_jit(self, device, dtype):
        op = kornia.contrib.EdgeDetector().to(device, dtype)
        op_jit = torch.jit.script(op)
        assert op_jit is not None
