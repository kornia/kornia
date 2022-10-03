from typing import List

import torch
import torch.nn as nn

from kornia.filters.dexined import DexiNed


class EdgeDetector(nn.Module):
    r"""Detect edges in a given image using a CNN.

    By default, it uses the method described in :cite:`xsoria2020dexined`.

    Return:
        A tensor of shape :math:`(B,1,H,W)`.

    Example:
        >>> img = torch.rand(1, 3, 320, 320)
        >>> detect = EdgeDetector()
        >>> out = detect(img)
        >>> out.shape
        torch.Size([1, 1, 320, 320])
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = DexiNed()

    def load(self, path_file: str) -> None:
        self.model.load_from_file(path_file)

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def postprocess(self, data: List[torch.Tensor]) -> torch.Tensor:
        # input are intermediate layer -- for inference we need only last.
        return data[-1]  # Bx1xHxW

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        img = self.preprocess(image)
        out = self.model(img)
        return self.postprocess(out)
