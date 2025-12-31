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

from kornia.losses.mutual_information import mutual_information_loss, normalized_mutual_information_loss


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
                normalized_loss_batch = normalized_mutual_information_loss(img1, img2, num_bins=64)

                # 3. Compute Iterative Loss (The "Slow but Correct" way)
                losses = []
                normalized_losses = []
                for i in range(unique_batch_dim_1.shape[0]):
                    loss = mutual_information_loss(unique_batch_dim_1[i], unique_batch_dim_2[i], num_bins=64)
                    normalized_loss = normalized_mutual_information_loss(
                        unique_batch_dim_1[i], unique_batch_dim_2[i], num_bins=64
                    )
                    losses.append(loss)
                    normalized_losses.append(normalized_loss)

                loss_iterative = torch.stack(losses)
                normalized_loss_iterative = torch.stack(normalized_losses)

                # 4. Compare
                print(f"{loss_batch.shape=},{loss_iterative.shape}")
                assert loss_batch.shape == dims[:-1], "The shape of the batched losses for mi is wrong."
                assert normalized_loss_batch.shape == dims[:-1], "The shape of the batched losses for nmi is wrong."

                assert torch.allclose(loss_batch.flatten(), loss_iterative, atol=1e-4), (
                    f"Batch mismatch for mi! Batch: {loss_batch}, Iterative: {loss_iterative}"
                )
                assert torch.allclose(normalized_loss_batch.flatten(), normalized_loss_iterative, atol=1e-4), (
                    f"Batch mismatch for nmi! Batch: {normalized_loss_batch}, Iterative: {normalized_loss_iterative}"
                )


## value ranges test

# "We check the expected behaviour for mutual information loss (MI loss)
#  and the normalized mutual information loss (NMI)\n",
#     "For two sample data of the same shape X,Y, these are respectively:\n",
#     "- $MI(X,Y)= H(X) + H(Y) - H(X,Y)$\n",
#     "- $NMI(X,Y)= (H(X) + H(Y)) / H(X,Y)$\n",
#     "Where H(X) is the Shannon entropy for X and H(X,Y) is the Shannon entropy for $X\\oplus Y$\n",
#     "\n",
#     "Observe the two extreme situations:\n",
#     "- if $Y=f(X)$ and $X=g(Y)$ for some functions $f$ and $g$, then $H(X)=H(X,Y)=H(Y)$\n",
#     "- if $X$ and $Y$ are independent, then $H(X,Y)=H(X) + H(Y)$\n",
#     "\n",
#     "These are the key observations for the output expectations in the loop below."


def relative_mi(img_1, img_2, window_radius):
    """Should theoretically be 0 if img_1 and img_2 are independent and 1 if img_1 = f(img_2), f one to one."""
    numerator = mutual_information_loss(img_1, img_2, window_radius=window_radius)
    denominator = mutual_information_loss(img_2, img_2, window_radius=window_radius)
    return numerator / denominator


def sampling_function(n_samples):
    data = torch.rand(n_samples)
    return 400 * torch.sin(data * torch.pi)


def value_ranges_check(n_samples=10000, num_bins=64):
    img_1 = sampling_function(n_samples)
    img_2 = 50 * img_1 + 1
    img_3 = sampling_function(n_samples)

    for radius in [1 / 2, 1, 2, 3]:
        # relative MI, expect 1
        assert torch.allclose(relative_mi(img_1, img_2, window_radius=radius), torch.ones(1)), (
            "Wrong MI behaviour, correlated case."
        )
        # relative MI, expect 0
        print(relative_mi(img_1, img_3, window_radius=radius))
        assert torch.allclose(relative_mi(img_1, img_3, window_radius=radius), torch.zeros(1), atol=0.2), (
            "Wrong MI behaviour, uncorrelated case."
        )

        assert torch.allclose(relative_mi(img_2, img_3, window_radius=radius), torch.zeros(1), atol=0.2), (
            "Wrong MI behaviour, uncorrelated case."
        )

        # NMI, expect -2
        assert torch.allclose(
            normalized_mutual_information_loss(img_1, img_2, window_radius=radius, num_bins=num_bins),
            -2 * torch.ones(1),
            atol=0.2 * radius + 0.15,
        ), "Wrong NMI behaviour, uncorrelated case."

        # NMI, expect -1
        assert torch.allclose(
            normalized_mutual_information_loss(img_1, img_3, window_radius=radius, num_bins=num_bins),
            -torch.ones(1),
            atol=0.1,
        ), "Wrong NMI behaviour, correlated case."
        assert torch.allclose(
            normalized_mutual_information_loss(img_2, img_3, window_radius=radius, num_bins=num_bins),
            -torch.ones(1),
            atol=0.1,
        ), "Wrong NMI behaviour, correlated case."


def test_value_ranges():
    for _ in range(10):
        value_ranges_check()


def test_differentiability():
    for _ in range(10):
        img_1 = sampling_function(10000)
        img_2 = sampling_function(10000)
        param = torch.tensor(1 / 2.0, requires_grad=True)
        mi = mutual_information_loss(img_1 + param * img_2, img_2)
        mi.backward()
        # negative gradient, order of magnitude 1/2
        assert param.grad > -1 and -1 / 10 > param.grad, f"Differentiability issue for mi, {param.grad=}."
        param = torch.tensor(1 / 2.0, requires_grad=True)
        nmi = normalized_mutual_information_loss(img_1 + param * img_2, img_2)
        nmi.backward()
        # negative gradient, order of magnitude 1/20
        assert param.grad > -1 / 10 and -1 / 100 > param.grad, f"Differentiability issue for nmi, {param.grad=}."
