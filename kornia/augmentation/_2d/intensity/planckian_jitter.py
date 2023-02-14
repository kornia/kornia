from typing import Any, Dict, List, Optional, Union

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core import Tensor, stack


class RandomPlanckianJitter(IntensityAugmentationBase2D):
    r"""Apply planckian jitter transformation to input tensor.

    .. image:: _static/img/RandomPlanckianJitter.png

    This is physics based color augmentation, that creates realistic
    variations in chromaticity, this can simulate the illumination
    changes in the scene.

    See :cite:`zini2022planckian` for more details.

    Args:
        mode: 'blackbody' or 'CIED'.
        select_from: choose a list of jitters to apply from. `blackbody` range [0-24], `CIED` range [0-22]
        same_on_batch: apply the same transformation across the batch.
        p: probability that the random erasing operation will be performed.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
            to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        Input tensor must be float and normalized into [0, 1].

    Examples:

        To apply planckian jitter based on mode

        >>> rng = torch.manual_seed(0)
        >>> input = torch.randn(1, 3, 2, 2)
        >>> aug = RandomPlanckianJitter(mode='CIED')
        >>> aug(input)
        tensor([[[[ 1.0000, -0.2389],
                  [-1.7740,  0.4628]],
        <BLANKLINE>
                 [[-1.0845, -1.3986],
                  [ 0.4033,  0.8380]],
        <BLANKLINE>
                 [[-0.9228, -0.5175],
                  [-0.7654,  0.2335]]]])

        To apply planckian jitter on image(s) from list of interested jitters

        >>> rng = torch.manual_seed(0)
        >>> input = torch.randn(2, 3, 2, 2)
        >>> aug = RandomPlanckianJitter(mode='blackbody', select_from=[23, 24, 1, 2])
        >>> aug(input)
        tensor([[[[-1.1258, -1.1524],
                  [-0.2506, -0.4339]],
        <BLANKLINE>
                 [[ 0.8487,  0.6920],
                  [-0.3160, -2.1152]],
        <BLANKLINE>
                 [[ 0.4681, -0.1577],
                  [ 1.4437,  0.2660]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[ 0.2465,  1.0000],
                  [-0.2125, -0.1653]],
        <BLANKLINE>
                 [[ 0.9318,  1.0000],
                  [ 1.0000,  0.0537]],
        <BLANKLINE>
                 [[ 0.2426, -0.1621],
                  [-0.3302, -0.9093]]]])
    """

    def __init__(
        self,
        mode: str = "blackbody",
        select_from: Optional[Union[int, List[int]]] = None,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)

        self._param_generator = rg.PlanckianJitterGenerator(mode=mode, select_from=select_from)

    def apply_transform(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        coeffs = params["coeffs"]

        r_w = coeffs[:, 0][..., None, None]
        b_w = coeffs[:, 1][..., None, None]

        r = input[..., 0, :, :] * r_w
        g = input[..., 1, :, :]
        b = input[..., 2, :, :] * b_w

        output = stack([r, g, b], -3)

        return output.clamp(max=1.0)
