import torch
from kornia.losses.mutual_information import mutual_information_loss

class TestMutualInformationBatch:
    def test_batch_consistency(self):
        """
        Verifies that:
        Loss(Batch of 4) == Mean(Loss(Image 1), Loss(Image 2), ...)
        """
        B, C, H, W = 4, 1, 32, 32
        device = torch.device('cpu')
        
        # 1. Create random batch
        torch.manual_seed(0)  # Fix seed for reproducibility
        img1 = torch.rand(B, C, H, W, device=device)
        img2 = torch.rand(B, C, H, W, device=device)

        # 2. Compute Batch Loss (The "Vectorized" way)
        # This will likely return a scalar computed on the flattened batch (Incorrect behavior)
        loss_batch = mutual_information_loss(img1, img2, num_bins=64)

        # 3. Compute Iterative Loss (The "Slow but Correct" way)
        losses = []
        for i in range(B):
            l = mutual_information_loss(img1[i:i+1], img2[i:i+1], num_bins=64)
            losses.append(l)
        
        loss_iterative_avg = torch.stack(losses).mean()

        # 4. Compare
        # We expect this to FAIL if the batch bug exists
        print(f"\nBatch Loss: {loss_batch.item()}")
        print(f"Iterative Loss: {loss_iterative_avg.item()}")
        
        assert torch.allclose(loss_batch, loss_iterative_avg, atol=1e-4), \
            f"Batch mismatch! Batch: {loss_batch.item()}, Iterative: {loss_iterative_avg.item()}"