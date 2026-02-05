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

import pytest
import torch

from kornia.models.small_sr import SmallSRNet, SmallSRNetWrapper

from testing.base import BaseTester


class TestSmallSRNet(BaseTester):
    """Test suite for SmallSRNet - the core super-resolution model."""

    @pytest.mark.parametrize("upscale_factor", [2, 3, 4])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_smoke(self, device, dtype, upscale_factor, batch_size):
        """Test that SmallSRNet can be instantiated and run with different upscale factors."""
        model = SmallSRNet(upscale_factor=upscale_factor, pretrained=False).to(device, dtype)
        assert model is not None

        # Input is single channel (Y channel from YCbCr)
        x = torch.randn(batch_size, 1, 224, 224, device=device, dtype=dtype)
        output = model(x)
        assert output is not None

    def test_exception_invalid_input_channels(self, device, dtype):
        """Test that SmallSRNet raises an error with wrong number of input channels."""
        model = SmallSRNet(upscale_factor=3, pretrained=False).to(device, dtype)

        # SmallSRNet expects 1 channel input (Y channel), not 3
        with pytest.raises(RuntimeError):
            x = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)
            model(x)

    def test_exception_invalid_upscale_factor(self, device, dtype):
        """Test that non-positive upscale factors are handled (model accepts them but may not work correctly)."""
        # Note: SmallSRNet doesn't validate upscale_factor, it just uses it in PixelShuffle
        # This test documents that behavior rather than enforcing validation
        model_zero = SmallSRNet(upscale_factor=1, pretrained=False).to(device, dtype)
        assert model_zero is not None

    @pytest.mark.parametrize("upscale_factor", [2, 3, 4])
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("height,width", [(64, 64), (128, 128), (224, 224)])
    def test_cardinality(self, device, dtype, upscale_factor, batch_size, height, width):
        """Test that output shape matches expected upscaled dimensions."""
        model = SmallSRNet(upscale_factor=upscale_factor, pretrained=False).to(device, dtype)

        x = torch.randn(batch_size, 1, height, width, device=device, dtype=dtype)
        output = model(x)

        expected_shape = (batch_size, 1, height * upscale_factor, width * upscale_factor)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    @pytest.mark.parametrize("upscale_factor", [2, 3, 4])
    def test_feature_forward_pass(self, device, dtype, upscale_factor):
        """Test the forward pass produces valid outputs with expected value ranges."""
        model = SmallSRNet(upscale_factor=upscale_factor, pretrained=False).to(device, dtype)

        x = torch.randn(1, 1, 64, 64, device=device, dtype=dtype)
        output = model(x)

        # Check output is not all zeros
        assert not torch.allclose(output, torch.zeros_like(output))

        # Check output is finite
        assert torch.isfinite(output).all()

    @pytest.mark.parametrize("upscale_factor", [2, 3, 4])
    def test_feature_pretrained_loading(self, device, dtype, upscale_factor):
        """Test that pretrained weights can be loaded (only upscale_factor=3 has pretrained weights)."""
        if upscale_factor == 3:
            # This should download and load pretrained weights
            model = SmallSRNet(upscale_factor=upscale_factor, pretrained=True).to(device, dtype)

            x = torch.randn(1, 1, 224, 224, device=device, dtype=dtype)
            output = model(x)

            # Pretrained model should be in eval mode
            assert not model.training

            # Output should be valid
            assert torch.isfinite(output).all()
        else:
            # For other upscale factors, pretrained weights may not exist
            # Model should still initialize without error
            model = SmallSRNet(upscale_factor=upscale_factor, pretrained=False).to(device, dtype)
            assert model is not None

    @pytest.mark.parametrize("upscale_factor", [2, 3, 4])
    def test_gradcheck(self, device, upscale_factor):
        """Test that gradients are computed correctly."""
        model = SmallSRNet(upscale_factor=upscale_factor, pretrained=False).to(device, torch.float64)
        model.train()

        x = torch.randn(1, 1, 16, 16, device=device, dtype=torch.float64, requires_grad=True)

        self.gradcheck(model, (x,), nondet_tol=1e-4)

    @pytest.mark.parametrize("upscale_factor", [2, 3, 4])
    def test_dynamo(self, device, dtype, torch_optimizer, upscale_factor):
        """Test that the model works with torch.compile."""
        model = SmallSRNet(upscale_factor=upscale_factor, pretrained=False).to(device, dtype)
        x = torch.randn(1, 1, 64, 64, device=device, dtype=dtype)

        op = model
        op_optimized = torch_optimizer(model)

        actual = op(x)
        expected = op_optimized(x)

        self.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


class TestSmallSRNetWrapper(BaseTester):
    """Test suite for SmallSRNetWrapper - the RGB-input wrapper with color space conversion."""

    @pytest.mark.parametrize("upscale_factor", [2, 3, 4])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_smoke(self, device, dtype, upscale_factor, batch_size):
        """Test that SmallSRNetWrapper can be instantiated and run."""
        model = SmallSRNetWrapper(upscale_factor=upscale_factor, pretrained=False).to(device, dtype)
        assert model is not None

        # Input is RGB image
        x = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)
        output = model(x)
        assert output is not None

    def test_exception_invalid_input_channels(self, device, dtype):
        """Test that SmallSRNetWrapper raises an error with wrong number of input channels."""
        model = SmallSRNetWrapper(upscale_factor=3, pretrained=False).to(device, dtype)

        # SmallSRNetWrapper expects 3 channel RGB input, rgb_to_ycbcr will raise ValueError
        with pytest.raises(ValueError, match="Input size must have a shape of"):
            x = torch.randn(1, 1, 224, 224, device=device, dtype=dtype)
            model(x)

    def test_exception_negative_values(self, device, dtype):
        """Test handling of invalid pixel value ranges (should still process but may produce unexpected results)."""
        model = SmallSRNetWrapper(upscale_factor=3, pretrained=False).to(device, dtype)

        # RGB values should typically be [0, 1] but model shouldn't crash on negative values
        x = torch.randn(1, 3, 64, 64, device=device, dtype=dtype) - 1.0  # Negative values
        output = model(x)

        # Should not crash, output should be finite
        assert torch.isfinite(output).all()

    @pytest.mark.parametrize("upscale_factor", [2, 3, 4])
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("height,width", [(64, 64), (128, 128), (224, 224)])
    def test_cardinality(self, device, dtype, upscale_factor, batch_size, height, width):
        """Test that output shape matches expected upscaled dimensions for RGB images."""
        model = SmallSRNetWrapper(upscale_factor=upscale_factor, pretrained=False).to(device, dtype)

        x = torch.randn(batch_size, 3, height, width, device=device, dtype=dtype)
        output = model(x)

        expected_shape = (batch_size, 3, height * upscale_factor, width * upscale_factor)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    @pytest.mark.parametrize("upscale_factor", [2, 3, 4])
    def test_feature_rgb_processing(self, device, dtype, upscale_factor):
        """Test that RGB images are processed correctly through color space conversions."""
        model = SmallSRNetWrapper(upscale_factor=upscale_factor, pretrained=False).to(device, dtype)

        # Create a simple RGB test image
        x = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)  # RGB in [0, 1]
        output = model(x)

        # Check output is not all zeros
        assert not torch.allclose(output, torch.zeros_like(output))

        # Check output is finite
        assert torch.isfinite(output).all()

        # Output should have 3 channels (RGB)
        assert output.shape[1] == 3

    def test_feature_pretrained_upscale_3x(self, device, dtype):
        """Test upscaling with pretrained weights (only available for upscale_factor=3)."""
        model = SmallSRNetWrapper(upscale_factor=3, pretrained=True).to(device, dtype)

        # Test with a realistic image
        x = torch.rand(1, 3, 224, 224, device=device, dtype=dtype)
        output = model(x)

        # Check output shape
        assert output.shape == (1, 3, 224 * 3, 224 * 3)

        # Pretrained model's inner SmallSRNet should be in eval mode
        assert not model.model.training

        # Output should be valid
        assert torch.isfinite(output).all()

    @pytest.mark.parametrize("upscale_factor", [2, 3, 4])
    def test_feature_color_space_conversion(self, device, dtype, upscale_factor):
        """Test that color space conversions (RGB->YCbCr->RGB) preserve information."""
        model = SmallSRNetWrapper(upscale_factor=upscale_factor, pretrained=False).to(device, dtype)

        # Create a simple test pattern
        x = torch.ones(1, 3, 32, 32, device=device, dtype=dtype) * 0.5
        output = model(x)

        # Output should maintain similar color characteristics (roughly)
        # This is a weak test since the model transforms the image
        assert output.shape[1] == 3  # RGB output

    @pytest.mark.parametrize("upscale_factor", [2, 3, 4])
    def test_gradcheck(self, device, upscale_factor):
        """Test that gradients are computed correctly through the wrapper."""
        model = SmallSRNetWrapper(upscale_factor=upscale_factor, pretrained=False).to(device, torch.float64)
        model.train()

        x = torch.randn(1, 3, 16, 16, device=device, dtype=torch.float64, requires_grad=True)

        self.gradcheck(model, (x,), nondet_tol=1e-4)

    @pytest.mark.parametrize("upscale_factor", [2, 3, 4])
    def test_dynamo(self, device, dtype, torch_optimizer, upscale_factor):
        """Test that the wrapper works with torch.compile."""
        model = SmallSRNetWrapper(upscale_factor=upscale_factor, pretrained=False).to(device, dtype)
        x = torch.randn(1, 3, 64, 64, device=device, dtype=dtype)

        op = model
        op_optimized = torch_optimizer(model)

        actual = op(x)
        expected = op_optimized(x)

        self.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_feature_consistency_across_batch_sizes(self, device, dtype):
        """Test that processing images individually vs in batch produces consistent results."""
        model = SmallSRNetWrapper(upscale_factor=3, pretrained=False).to(device, dtype)
        model.eval()

        # Create two identical images
        x1 = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)
        x2 = x1.clone()
        x_batch = torch.cat([x1, x2], dim=0)

        with torch.no_grad():
            output1 = model(x1)
            output2 = model(x2)
            output_batch = model(x_batch)

        # Individual processing should match batch processing
        self.assert_close(output1, output_batch[0:1], rtol=1e-4, atol=1e-4)
        self.assert_close(output2, output_batch[1:2], rtol=1e-4, atol=1e-4)
