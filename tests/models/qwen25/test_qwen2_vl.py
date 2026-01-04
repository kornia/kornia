import pytest
import torch

from kornia.models.qwen25 import Qwen2VLVisionTransformer

class TestQwen2VL:
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_smoke(self, batch_size, device, dtype):
        model = Qwen2VLVisionTransformer().to(device=device, dtype=dtype)
        input = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)
        
        output = model(input)
        
    
        assert output.shape[0] == batch_size
        assert output.shape[2] == 1280