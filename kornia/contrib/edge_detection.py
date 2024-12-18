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

from kornia.core import Module, Tensor
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.filters.dexined import DexiNed


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

    def postprocess(self, data: Tensor) -> Tensor:
        return data

    def forward(self, image: Tensor) -> Tensor:
        KORNIA_CHECK_SHAPE(image, ["B", "3", "H", "W"])
        img = self.preprocess(image)
        out = self.model(img)
        return self.postprocess(out)
