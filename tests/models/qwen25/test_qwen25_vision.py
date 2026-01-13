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

"""Tests for Qwen2.5-VL Vision Encoder."""

import pytest
import torch

from kornia.models.qwen25 import Qwen2VLVisionTransformer


class TestQwen25VisionEncoder:
    """Basic tests for vision encoder architecture."""
    
    def test_model_creation(self):
        """Test that model can be instantiated."""
        model = Qwen2VLVisionTransformer()
        assert model is not None
        assert isinstance(model, Qwen2VLVisionTransformer)
    
    def test_forward_pass(self):
        """Test forward pass with random input."""
        model = Qwen2VLVisionTransformer()
        model.eval()
        
        with torch.no_grad():
            x = torch.randn(1, 3, 448, 448)
            output = model(x)
        
        # Check output shape
        assert output.dim() == 3, f"Expected (B, seq, dim), got {output.shape}"
        assert output.shape[0] == 1  # batch size
        
        # Get expected dimension from model
        expected_dim = model.merger.mlp[-1].out_features
        assert output.shape[2] == expected_dim
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_batch_processing(self, batch_size):
        """Test with different batch sizes."""
        model = Qwen2VLVisionTransformer()
        model.eval()
        
        x = torch.randn(batch_size, 3, 448, 448)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape[0] == batch_size
    
    @pytest.mark.parametrize("resolution", [224, 448])
    def test_different_resolutions(self, resolution):
        """Test with different input resolutions."""
        model = Qwen2VLVisionTransformer()
        model.eval()
        
        x = torch.randn(1, 3, resolution, resolution)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape[0] == 1
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test model on CUDA."""
        model = Qwen2VLVisionTransformer()
        model = model.to("cuda")
        model.eval()
        
        with torch.no_grad():
            x = torch.randn(1, 3, 448, 448, device="cuda")
            output = model(x)
        
        assert output.device.type == "cuda"
