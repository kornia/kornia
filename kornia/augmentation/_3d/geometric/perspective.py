from typing import Dict, Optional, Union, cast

import torch

from kornia.augmentation import random_generator as rg
from kornia.augmentation._3d.base import AugmentationBase3D
from kornia.constants import Resample
from kornia.geometry import get_perspective_transform3d, warp_perspective3d


class RandomPerspective3D(AugmentationBase3D):
    r"""Apply andom perspective transformation to 3D volumes (5D tensor).

    Args:
        p: probability of the image being perspectively transformed.
        distortion_scale: it controls the degree of distortion and ranges from 0 to 1.
        resample:
        return_transform: if ``True`` return the matrix describing the transformation applied to each.
        same_on_batch: apply the same transformation across the batch.
        align_corners: interpolation flag.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
          to the batch form (False).

    Shape:
        - Input: :math:`(C, D, H, W)` or :math:`(B, C, D, H, W)`, Optional: :math:`(B, 4, 4)`
        - Output: :math:`(B, C, D, H, W)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 4, 4)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs= torch.tensor([[[
        ...    [[1., 0., 0.],
        ...     [0., 1., 0.],
        ...     [0., 0., 1.]],
        ...    [[1., 0., 0.],
        ...     [0., 1., 0.],
        ...     [0., 0., 1.]],
        ...    [[1., 0., 0.],
        ...     [0., 1., 0.],
        ...     [0., 0., 1.]]
        ... ]]])
        >>> aug = RandomPerspective3D(0.5, p=1., align_corners=True)
        >>> aug(inputs)
        tensor([[[[[0.3976, 0.5507, 0.0000],
                   [0.0901, 0.3668, 0.0000],
                   [0.0000, 0.0000, 0.0000]],
        <BLANKLINE>
                  [[0.2651, 0.4657, 0.0000],
                   [0.1390, 0.5174, 0.0000],
                   [0.0000, 0.0000, 0.0000]],
        <BLANKLINE>
                  [[0.0000, 0.1153, 0.0000],
                   [0.0000, 0.0000, 0.0000],
                   [0.0000, 0.0000, 0.0000]]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.rand(1, 3, 32, 32, 32)
        >>> aug = RandomPerspective3D(0.5, p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        distortion_scale: Union[torch.Tensor, float] = 0.5,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        return_transform: bool = False,
        same_on_batch: bool = False,
        align_corners: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch, keepdim=keepdim)
        self.flags = dict(resample=Resample.get(resample), align_corners=align_corners)
        self._param_generator = cast(rg.PerspectiveGenerator3D, rg.PerspectiveGenerator3D(distortion_scale))

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return get_perspective_transform3d(params["start_points"], params["end_points"]).to(input)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        transform = cast(torch.Tensor, transform)
        return warp_perspective3d(
            input,
            transform,
            (input.shape[-3], input.shape[-2], input.shape[-1]),
            flags=self.flags["resample"].name.lower(),
            align_corners=self.flags["align_corners"],
        )
