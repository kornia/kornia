import os
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image as PILImage

from kornia.core.module import ImageModule, ImageModuleMixIn

# Assuming ImageModuleMixIn and ImageModule have been imported from the module


class TestImageModuleMixIn:
    @pytest.fixture
    def img_module(self):
        class DummyModule(ImageModuleMixIn):
            pass

        return DummyModule()

    @pytest.fixture
    def sample_image(self):
        # Create a sample PIL image for testing
        return PILImage.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

    @pytest.fixture
    def sample_tensor(self):
        # Create a sample tensor for testing
        return torch.rand((3, 100, 100))

    @pytest.fixture
    def sample_numpy(self):
        # Create a sample numpy array for testing
        return np.random.rand(100, 100, 3).astype(np.float32)

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


class TestImageModule:
    @pytest.fixture
    def image_module(self):
        return ImageModule()

    @pytest.fixture
    def sample_tensor(self):
        return torch.rand((3, 100, 100))

    def test_disable_features(self, image_module):
        image_module.disable_features = True
        assert image_module.disable_features is True

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
