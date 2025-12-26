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

import torch

from kornia.losses.mutual_information import mutual_information_loss


class TestMutualInformationBatch:
    def test_batch_consistency(self):
        """
        Verifies that:
        Loss(Batch of 4) == Mean(Loss(Image 1), Loss(Image 2), ...)
        """
        torch.manual_seed(0)  # Fix seed for reproducibility
        for i in range(5):
            # 1. Create random batch with n_dims = i+1
            dims = torch.randint(low=1, high=8, size=(i + 1,))
            dims = tuple(map(int, dims))
            device = torch.device("cpu")
            for _ in range(10):
                img1 = torch.rand(dims, device=device)
                img2 = torch.rand(dims, device=device)

                # flatten batch dims
                unique_batch_dim_1 = img1.reshape((-1,) + img1.shape[-1:])
                unique_batch_dim_2 = img2.reshape((-1,) + img1.shape[-1:])

                # 2. Compute Batch Loss (The "Vectorized" way)
                # This will likely return a scalar computed on the flattened batch (Incorrect behavior)
                loss_batch = mutual_information_loss(img1, img2, num_bins=64)

                # 3. Compute Iterative Loss (The "Slow but Correct" way)
                losses = []
                for i in range(unique_batch_dim_1.shape[0]):
                    loss = mutual_information_loss(unique_batch_dim_1[i], unique_batch_dim_2[i], num_bins=64)
                    losses.append(loss)

                loss_iterative = torch.stack(losses)

                # 4. Compare
                print(f"{loss_batch.shape=},{loss_iterative.shape}")
                assert loss_batch.shape == dims[:-1], "The shape of the batched losses is wrong."
                assert torch.allclose(loss_batch.flatten(), loss_iterative, atol=1e-4), (
                    f"Batch mismatch! Batch: {loss_batch}, Iterative: {loss_iterative}"
                )
