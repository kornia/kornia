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

from kornia.core.module import ImageModule, ImageModuleMixIn


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

    def test_show_4d_tensor(self, img_module):
        img_module._output_image = torch.rand(4, 3, 16, 16)
        result = img_module.show(display=False)
        assert isinstance(result, PILImage.Image)

    def test_show_unsupported_backend_raises(self, img_module, sample_tensor):
        img_module._output_image = sample_tensor
        with pytest.raises(ValueError, match="Unsupported backend"):
            img_module.show(backend="matplotlib", display=False)

    def test_detach_tensor_to_cpu_tensor(self, img_module, sample_tensor):
        result = img_module._detach_tensor_to_cpu(sample_tensor)
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"

    def test_detach_tensor_to_cpu_list(self, img_module):
        tensors = [torch.rand(3, 4, 4), torch.rand(3, 4, 4)]
        result = img_module._detach_tensor_to_cpu(tensors)
        assert isinstance(result, list)
        assert all(t.device.type == "cpu" for t in result)

    def test_detach_tensor_to_cpu_tuple(self, img_module):
        tensors = (torch.rand(3, 4, 4), torch.rand(3, 4, 4))
        result = img_module._detach_tensor_to_cpu(tensors)
        assert isinstance(result, tuple)


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
