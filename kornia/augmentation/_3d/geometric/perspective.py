from typing import Any, Dict, Optional, Union

from kornia.augmentation import random_generator as rg
from kornia.augmentation._3d.geometric.base import GeometricAugmentationBase3D
from kornia.constants import Resample
from kornia.core import Tensor
from kornia.geometry import get_perspective_transform3d, warp_perspective3d


class RandomPerspective3D(GeometricAugmentationBase3D):
    r"""Apply andom perspective transformation to 3D volumes (5D tensor).

    Args:
        p: probability of the image being perspectively transformed.
        distortion_scale: it controls the degree of distortion and ranges from 0 to 1.
        resample: resample mode from "nearest" (0) or "bilinear" (1).
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
        >>> import torch
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
        distortion_scale: Union[Tensor, float] = 0.5,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.flags = {"resample": Resample.get(resample), "align_corners": align_corners}
        self._param_generator = rg.PerspectiveGenerator3D(distortion_scale)

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        return get_perspective_transform3d(params["start_points"], params["end_points"]).to(input)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        if not isinstance(transform, Tensor):
            raise TypeError(f"Expected the transform to be a Tensor. Gotcha {type(transform)}")

        return warp_perspective3d(
            input,
            transform,
            (input.shape[-3], input.shape[-2], input.shape[-1]),
            flags=flags["resample"].name.lower(),
            align_corners=flags["align_corners"],
        )
