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

import os
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image as PILImage

from kornia.augmentation import ImageSequential as AugmentationImageSequential
from kornia.core.module import ImageModule, ImageModuleMixIn, ImageSequential

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, dynamo_is_available


class TestImageModuleMixIn:
    @pytest.fixture
    def img_module(self):
        class DummyModule(ImageModuleMixIn):
            pass

        return DummyModule()

    @pytest.fixture
    def sample_image(self):
        # Create a sample PIL image for testing
        return PILImage.fromarray(torch.randint(0, 255, (100, 100, 3)).numpy().astype(np.uint8))

    @pytest.fixture
    def sample_tensor(self):
        # Create a sample tensor for testing
        return torch.rand((3, 100, 100))

    @pytest.fixture
    def sample_numpy(self):
        # Create a sample numpy array for testing
        return torch.rand(100, 100, 3).numpy()

    def test_to_tensor_pil(self, img_module, sample_image):
        tensor = img_module.to_tensor(sample_image)
        assert isinstance(tensor, (torch.Tensor,))
        assert tensor.shape == (3, 100, 100)

    def test_to_tensor_numpy(self, img_module, sample_numpy):
        tensor = img_module.to_tensor(sample_numpy)
        assert isinstance(tensor, (torch.Tensor,))
        assert tensor.shape == (3, 100, 100)

    def test_to_tensor_tensor(self, img_module, sample_tensor):
        tensor = img_module.to_tensor(sample_tensor)
        assert tensor is sample_tensor

    def test_to_numpy_tensor(self, img_module, sample_tensor):
        array = img_module.to_numpy(sample_tensor)
        assert isinstance(array, (np.ndarray,))
        assert array.shape == (3, 100, 100)

    def test_to_numpy_numpy(self, img_module, sample_numpy):
        array = img_module.to_numpy(sample_numpy)
        assert array is sample_numpy

    def test_to_pil_tensor(self, img_module, sample_tensor):
        pil_image = img_module.to_pil(sample_tensor)
        assert isinstance(pil_image, (PILImage.Image,))

    def test_to_pil_pil(self, img_module, sample_image):
        pil_image = img_module.to_pil(sample_image)
        assert pil_image is sample_image

    def test_convert_input_output(self, img_module, sample_image, sample_numpy, sample_tensor):
        @img_module.convert_input_output(output_type="numpy")
        def dummy_func(tensor):
            return tensor

        output = dummy_func(sample_image)
        assert isinstance(output, (np.ndarray,))

    def test_show(self, img_module, sample_tensor):
        img_module._output_image = sample_tensor
        pil_image = img_module.show(display=False)
        assert isinstance(pil_image, (PILImage.Image,))

    def test_save(self, img_module, sample_tensor, tmpdir):
        img_module._output_image = sample_tensor
        save_path = tmpdir.join("test_image.jpg")
        img_module.save(name=save_path)
        assert os.path.exists(save_path)

    def test_to_pil_4d_tensor_returns_list(self, img_module):
        # 4D (B, C, H, W) tensor -> list of PIL Images
        t = torch.rand(3, 3, 16, 16)
        result = img_module.to_pil(t)
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(im, PILImage.Image) for im in result)

    def test_to_pil_numpy_raises(self, img_module, sample_numpy):
        with pytest.raises(NotImplementedError):
            img_module.to_pil(sample_numpy)

    def test_to_pil_1d_tensor_raises(self, img_module):
        with pytest.raises(NotImplementedError):
            img_module.to_pil(torch.rand(8))

    def test_to_numpy_pil(self, img_module, sample_image):
        arr = img_module.to_numpy(sample_image)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (100, 100, 3)

    def test_convert_input_output_invalid_type_raises(self, img_module, sample_tensor):
        with pytest.raises(ValueError, match="Invalid output_type"):

            @img_module.convert_input_output(output_type="invalid")
            def dummy_func(tensor):
                return tensor

    def test_convert_input_output_pil_output(self, img_module, sample_tensor):
        @img_module.convert_input_output(output_type="pil")
        def dummy_func(tensor):
            return tensor

        result = dummy_func(sample_tensor)
        assert isinstance(result, PILImage.Image)

    def test_convert_input_output_selective_input_names(self, img_module, sample_image):
        # Only convert arguments named "image", leave others unchanged
        @img_module.convert_input_output(input_names_to_handle=["image"], output_type="pt")
        def dummy_func(image, other):
            return image

        result = dummy_func(sample_image, "not_an_image")
        assert isinstance(result, torch.Tensor)

    def test_convert_input_output_default_only_converts_first_input(self, img_module, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        image_path = tmp_path / "mode.png"
        PILImage.new("RGB", (6, 4)).save(image_path)

        @img_module.convert_input_output(output_type="pt")
        def dummy_func(image, mode="nearest"):
            return image, mode

        image, mode = dummy_func(str(image_path), mode="mode.png")

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 4, 6)
        assert mode == "mode.png"

    def test_convert_input_output_default_does_not_convert_keyword_file(self, img_module, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "bilinear").touch()

        @img_module.convert_input_output(output_type="pt")
        def dummy_func(image, mode="nearest"):
            return image, mode

        image = torch.zeros(3, 4, 6)
        result, mode = dummy_func(image, mode="bilinear")

        assert result is image
        assert mode == "bilinear"

    def test_show_4d_tensor(self, img_module):
        img_module._output_image = torch.rand(4, 3, 16, 16)
        result = img_module.show(display=False)
        assert isinstance(result, PILImage.Image)

    def test_show_unsupported_backend_raises(self, img_module, sample_tensor):
        img_module._output_image = sample_tensor
        with pytest.raises(ValueError, match="Unsupported backend"):
            img_module.show(backend="matplotlib", display=False)

    def test_save_without_output_image_raises(self, img_module, tmpdir):
        img_module._output_image = None
        with pytest.raises(ValueError, match="No pre-computed images found"):
            img_module.save(name=tmpdir.join("test_image.jpg"))

    @pytest.mark.parametrize("container", [list, tuple])
    def test_store_output_image_sequence_4957(self, img_module, container, device, dtype):
        tensors = container(torch.rand(3, 4, 4, device=device, dtype=dtype, requires_grad=True) for _ in range(2))
        img_module._store_output_image(tensors, "pt")
        result = img_module._output_image
        assert isinstance(result, container)
        assert len(result) == len(tensors)
        for cached, output in zip(result, tensors):
            assert cached.device == output.device
            assert not cached.requires_grad
            assert cached.grad_fn is None
            torch.testing.assert_close(cached, output.detach())


class TestImageModule:
    @pytest.fixture
    def image_module(self):
        return ImageModule()

    @pytest.fixture
    def sample_tensor(self):
        return torch.rand((3, 100, 100))

    def test_call_with_features_disabled(self, image_module, sample_tensor):
        image_module.disable_features = True
        mock_forward = MagicMock(return_value=sample_tensor)
        image_module.forward = mock_forward
        output = image_module(sample_tensor)
        assert output is sample_tensor
        mock_forward.assert_called_once()

    def test_call_with_features_enabled(self, image_module, sample_tensor):
        image_module.disable_features = False
        mock_forward = MagicMock(return_value=sample_tensor)
        image_module.forward = mock_forward
        output = image_module(sample_tensor)
        assert output is sample_tensor
        mock_forward.assert_called_once()


class TestLazyOutputCache(BaseTester):
    @pytest.fixture(params=["module", "core_sequential", "augmentation_sequential"])
    def module(self, request):
        class Sigmoid(ImageModule):
            def forward(self, x):
                return x.sigmoid()

        if request.param == "module":
            return Sigmoid()
        if request.param == "core_sequential":
            return ImageSequential(torch.nn.Sigmoid())
        return AugmentationImageSequential(torch.nn.Sigmoid())

    def test_forward_cache_stays_on_device_4957(self, module, device, dtype):
        image = torch.rand(2, 3, 6, 8, device=device, dtype=dtype, requires_grad=True)
        output = module(image)
        cached = module._output_image
        # Restoring an eager .cpu() must fail this assertion on CUDA/MPS.
        assert cached.device == output.device == image.device
        assert not cached.requires_grad
        assert cached.grad_fn is None
        self.assert_close(cached, output.detach())
        output.sum().backward()
        self.assert_close(image.grad, output.detach() * (1 - output.detach()))

    def test_show_save_lazy_cache_4957(self, module, device, dtype, tmp_path):
        image = torch.rand(1, 3, 6, 8, device=device, dtype=dtype, requires_grad=True)
        output = module(image)
        cached = module._output_image
        expected = (output[0].detach().cpu().permute(1, 2, 0) * 255).byte().numpy()
        if dtype != torch.bfloat16:  # NumPy does not support bfloat16.
            rendered = module.show(display=False)
            assert isinstance(rendered, PILImage.Image)
            np.testing.assert_array_equal(np.asarray(rendered), expected)
        path = tmp_path / "cached.png"
        module.save(name=str(path))
        with PILImage.open(path) as saved:
            np.testing.assert_array_equal(np.asarray(saved), expected)
        assert module._output_image is cached
        assert cached.device == image.device

    @pytest.mark.parametrize("output_type", ["numpy", "pil"])
    def test_requested_output_conversion_4957(self, module, output_type, device, dtype):
        if dtype == torch.bfloat16 and output_type == "numpy":
            pytest.skip("NumPy does not support bfloat16")
        image = torch.rand(1, 3, 6, 8, device=device, dtype=dtype, requires_grad=True)
        output = module(image, output_type=output_type)
        expected = image.sigmoid().detach().cpu()
        if output_type == "numpy":
            assert isinstance(output, np.ndarray)
            np.testing.assert_array_equal(output, expected.numpy())
        else:
            assert isinstance(output, list)
            assert len(output) == 1
            assert isinstance(output[0], PILImage.Image)
            np.testing.assert_array_equal(np.asarray(output[0]), (expected[0].permute(1, 2, 0) * 255).byte().numpy())

    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    def test_export_preserves_cache_4957(self, device, dtype):
        class CacheModule(torch.nn.Module, ImageModuleMixIn):
            def forward(self, x):
                output = x.sigmoid()
                self._store_output_image(output, "pt")
                return output

        module = CacheModule()
        image = torch.rand(1, 3, 6, 8, device=device, dtype=dtype)
        expected = module(image)
        cached = module._output_image
        exported = torch.export.export(module, (image,), strict=True)
        self.assert_close(exported.module()(image), expected)
        assert module._output_image is cached
