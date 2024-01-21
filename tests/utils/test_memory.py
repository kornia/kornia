import torch

import kornia
from testing.base import BaseTester


class TestBatchedForward(BaseTester):
    def test_runbatch(self, device):
        patches = torch.rand(34, 1, 32, 32)
        sift = kornia.feature.SIFTDescriptor(32)
        desc_batched = kornia.utils.memory.batched_forward(sift, patches, device, 32)
        desc = sift(patches)
        assert torch.allclose(desc, desc_batched)

    def test_runone(self, device):
        patches = torch.rand(16, 1, 32, 32)
        sift = kornia.feature.SIFTDescriptor(32)
        desc_batched = kornia.utils.memory.batched_forward(sift, patches, device, 32)
        desc = sift(patches)
        assert torch.allclose(desc, desc_batched)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 3, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(
            kornia.utils.memory.batched_forward, (kornia.feature.BlobHessian(), img, device, 2), nondet_tol=1e-4
        )
