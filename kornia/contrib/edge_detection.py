from typing import List

from kornia.core import Module, Tensor
from kornia.filters.dexined import DexiNed
from kornia.testing import KORNIA_CHECK_SHAPE


class EdgeDetector(Module):
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
        self.model = DexiNed(pretrained=True)

    def load(self, path_file: str) -> None:
        self.model.load_from_file(path_file)

    def preprocess(self, image: Tensor) -> Tensor:
        return image

    def postprocess(self, data: List[Tensor]) -> Tensor:
        # input are intermediate layer -- for inference we need only last.
        return data[-1]  # Bx1xHxW

    def forward(self, image: Tensor) -> Tensor:
        KORNIA_CHECK_SHAPE(image, ["B", "3", "H", "W"])
        img = self.preprocess(image)
        out = self.model(img)
        return self.postprocess(out)
