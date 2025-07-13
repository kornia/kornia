import base64
import io
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

from kornia.mcp.utils import load_any_image


def test_load_input_image_from_numpy():
    # Test loading from numpy array
    img_np = np.random.rand(100, 100, 3)
    img_tensor = load_any_image(img_np)
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 100, 100)

def test_load_input_image_from_tensor():
    # Test loading from tensor
    img_tensor_input = torch.randn(3, 100, 100)
    img_tensor = load_any_image(img_tensor_input)
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 100, 100)

def test_load_input_image_from_tensor_gray():
    # Test loading grayscale tensor
    img_tensor_gray = torch.randn(100, 100)
    img_tensor = load_any_image(img_tensor_gray)
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (1, 100, 100)

def test_load_input_image_from_tensor_hwc():
    # Test loading HWC tensor
    img_tensor_hwc = torch.randn(100, 100, 3)
    img_tensor = load_any_image(img_tensor_hwc)
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 100, 100)

    # Test invalid tensor shape
    with pytest.raises(ValueError):
        load_any_image(torch.randn(1, 2, 3, 4))

    # Test invalid input type
    with pytest.raises(TypeError):
        load_any_image(123)

def test_load_input_image_from_file_path():
    # Test loading from file path
    with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
        # Create a temporary test image
        img = Image.fromarray((np.random.rand(100, 100, 3) * 255).astype('uint8'))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()
        tmp.write(img_bytes)
        tmp.flush()
        tmp.seek(0)

        img_tensor = load_any_image(tmp.name)
        assert isinstance(img_tensor, torch.Tensor)
        assert img_tensor.shape[0] in [1, 3]  # Either grayscale or RGB
        assert len(img_tensor.shape) == 3

def test_load_input_image_from_base64_string():
    # Test loading from base64 string
    img_np = (np.random.rand(100, 100, 3) * 255).astype('uint8')
    img = Image.fromarray(img_np)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    base64_str = base64.b64encode(buffer.getvalue()).decode()
    img_tensor = load_any_image(f'data:image/jpeg;base64,{base64_str}')
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape[0] in [1, 3]
    assert len(img_tensor.shape) == 3