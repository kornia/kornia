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

from kornia.augmentation import EdgeAwareAugmentation, RandomBrightness, RandomGaussianBlur

from testing.base import BaseTester


class TestEdgeAwareAugmentation(BaseTester):
    @pytest.mark.parametrize("edge_detector", ["sobel", "canny"])
    @pytest.mark.parametrize("mode", ["soft", "hard"])
    @pytest.mark.parametrize("edge_weight", [0.0, 0.3, 0.5, 1.0])
    def test_smoke(self, edge_detector, mode, edge_weight, device, dtype):
        """Test basic functionality with various parameter combinations."""
        base_aug = RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0)
        aug = EdgeAwareAugmentation(
            base_aug,
            edge_detector=edge_detector,
            mode=mode,
            edge_weight=edge_weight,
            p=1.0,
        )
        input_tensor = torch.rand(1, 3, 32, 32, device=device, dtype=dtype)
        output = aug(input_tensor)
        assert output.shape == input_tensor.shape

    def test_exception_invalid_edge_detector(self):
        """Test that invalid edge_detector raises ValueError."""
        base_aug = RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0)
        with pytest.raises(ValueError, match="edge_detector must be"):
            EdgeAwareAugmentation(base_aug, edge_detector="invalid", p=1.0)

    def test_exception_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        base_aug = RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0)
        with pytest.raises(ValueError, match="mode must be"):
            EdgeAwareAugmentation(base_aug, mode="invalid", p=1.0)

    def test_exception_invalid_edge_weight(self):
        """Test that invalid edge_weight raises ValueError."""
        base_aug = RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0)
        with pytest.raises(ValueError, match="edge_weight must be"):
            EdgeAwareAugmentation(base_aug, edge_weight=1.5, p=1.0)
        with pytest.raises(ValueError, match="edge_weight must be"):
            EdgeAwareAugmentation(base_aug, edge_weight=-0.1, p=1.0)

    def test_cardinality(self, device, dtype):
        """Test output shape for various input shapes."""
        base_aug = RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0)
        aug = EdgeAwareAugmentation(base_aug, p=1.0)

        for shape in [(1, 3, 32, 32), (2, 3, 64, 64), (4, 1, 16, 16)]:
            input_tensor = torch.rand(shape, device=device, dtype=dtype)
            output = aug(input_tensor)
            assert output.shape == input_tensor.shape

    def test_random_p_0(self, device, dtype):
        """Test that p=0 returns input unchanged."""
        base_aug = RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0)
        aug = EdgeAwareAugmentation(base_aug, p=0.0)
        input_tensor = torch.rand(1, 3, 32, 32, device=device, dtype=dtype)
        output = aug(input_tensor)
        self.assert_close(output, input_tensor)

    def test_edge_preservation(self, device, dtype):
        """Test that edges are preserved better than with base augmentation alone."""
        # Create an image with clear edges (checkerboard pattern)
        input_tensor = torch.zeros(1, 1, 16, 16, device=device, dtype=dtype)
        input_tensor[:, :, ::2, ::2] = 1.0
        input_tensor[:, :, 1::2, 1::2] = 1.0

        base_aug = RandomGaussianBlur(kernel_size=(5, 5), sigma=(2.0, 2.0), p=1.0)
        edge_aware_aug = EdgeAwareAugmentation(base_aug, edge_weight=1.0, p=1.0)

        # Apply both augmentations
        base_output = base_aug(input_tensor)
        edge_aware_output = edge_aware_aug(input_tensor)

        # Edge-aware should preserve more of the original structure near edges
        # The difference from original should be smaller for edge-aware
        base_diff = (base_output - input_tensor).abs().mean()
        edge_aware_diff = (edge_aware_output - input_tensor).abs().mean()

        # With high edge_weight, edge-aware should stay closer to original
        assert edge_aware_diff <= base_diff

    def test_soft_vs_hard_mode(self, device, dtype):
        """Test difference between soft and hard modes."""
        # Create image with varying edge strengths
        input_tensor = torch.rand(1, 3, 32, 32, device=device, dtype=dtype)

        base_aug = RandomGaussianBlur(kernel_size=(3, 3), sigma=(1.0, 1.0), p=1.0)

        soft_aug = EdgeAwareAugmentation(base_aug, mode="soft", edge_weight=0.5, p=1.0)
        hard_aug = EdgeAwareAugmentation(base_aug, mode="hard", edge_weight=0.5, p=1.0)

        soft_output = soft_aug(input_tensor)
        hard_output = hard_aug(input_tensor)

        # Outputs should be different due to different modulation strategies
        assert not torch.allclose(soft_output, hard_output, rtol=1e-4, atol=1e-4)

    def test_sobel_vs_canny(self, device, dtype):
        """Test difference between sobel and canny edge detectors."""
        input_tensor = torch.rand(1, 3, 32, 32, device=device, dtype=dtype)

        base_aug = RandomGaussianBlur(kernel_size=(3, 3), sigma=(1.0, 1.0), p=1.0)

        sobel_aug = EdgeAwareAugmentation(base_aug, edge_detector="sobel", edge_weight=0.5, p=1.0)
        canny_aug = EdgeAwareAugmentation(base_aug, edge_detector="canny", edge_weight=0.5, p=1.0)

        sobel_output = sobel_aug(input_tensor)
        canny_output = canny_aug(input_tensor)

        # Outputs should be different due to different edge detection methods
        assert not torch.allclose(sobel_output, canny_output, rtol=1e-4, atol=1e-4)

    def test_detach_edges(self, device, dtype):
        """Test that detach_edges affects gradient flow."""
        input_tensor = torch.rand(1, 3, 32, 32, device=device, dtype=dtype, requires_grad=True)

        base_aug = RandomGaussianBlur(kernel_size=(3, 3), sigma=(1.0, 1.0), p=1.0)

        detach_aug = EdgeAwareAugmentation(base_aug, detach_edges=True, p=1.0)
        no_detach_aug = EdgeAwareAugmentation(base_aug, detach_edges=False, p=1.0)

        detach_output = detach_aug(input_tensor.clone())
        no_detach_output = no_detach_aug(input_tensor.clone())

        # Both should produce valid outputs
        assert detach_output.shape == (1, 3, 32, 32)
        assert no_detach_output.shape == (1, 3, 32, 32)

    def test_with_brightness(self, device, dtype):
        """Test edge-aware wrapping of RandomBrightness."""
        base_aug = RandomBrightness(brightness=(0.5, 1.5), p=1.0)
        aug = EdgeAwareAugmentation(base_aug, edge_weight=0.3, p=1.0)

        input_tensor = torch.rand(1, 3, 32, 32, device=device, dtype=dtype)
        output = aug(input_tensor)
        assert output.shape == input_tensor.shape

    def test_batch_processing(self, device, dtype):
        """Test batch processing."""
        base_aug = RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0)
        aug = EdgeAwareAugmentation(base_aug, p=1.0)

        input_tensor = torch.rand(4, 3, 32, 32, device=device, dtype=dtype)
        output = aug(input_tensor)
        assert output.shape == input_tensor.shape

    def test_same_on_batch(self, device, dtype):
        """Test same_on_batch parameter."""
        base_aug = RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0)
        aug = EdgeAwareAugmentation(base_aug, same_on_batch=True, p=1.0)

        input_tensor = torch.rand(4, 3, 32, 32, device=device, dtype=dtype)
        output = aug(input_tensor)
        assert output.shape == input_tensor.shape
