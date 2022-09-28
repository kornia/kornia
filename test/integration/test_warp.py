from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import kornia


class MyHomography(nn.Module):
    def __init__(self, init_homo: torch.Tensor) -> None:
        super().__init__()
        self.homo = nn.Parameter(init_homo.clone().detach())

    def forward(self) -> torch.Tensor:
        return torch.unsqueeze(self.homo, dim=0)


class TestWarping:
    # optimization
    lr = 1e-3
    num_iterations = 100

    def test_smoke(self, device):

        img_src_t: torch.Tensor = torch.rand(1, 3, 120, 120).to(device)
        img_dst_t: torch.Tensor = torch.rand(1, 3, 120, 120).to(device)

        init_homo: torch.Tensor = torch.from_numpy(
            np.array([[0.0415, 1.2731, -1.1731], [-0.9094, 0.5072, 0.4272], [0.0762, 1.3981, 1.0646]])
        ).float()

        height, width = img_dst_t.shape[-2:]
        warper = kornia.geometry.transform.HomographyWarper(height, width)
        dst_homo_src = MyHomography(init_homo=init_homo).to(device)

        learning_rate = self.lr
        optimizer = optim.Adam(dst_homo_src.parameters(), lr=learning_rate)

        for _ in range(self.num_iterations):
            # warp the reference image to the destiny with current homography
            img_src_to_dst = warper(img_src_t, dst_homo_src())

            # compute the photometric loss
            loss = F.l1_loss(img_src_to_dst, img_dst_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            assert not bool(torch.isnan(dst_homo_src.homo.grad).any())
