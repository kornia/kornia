from typing import Callable, Tuple, Union, Optional, Dict, cast
from collections import OrderedDict

import torch
from torch.distributions import Uniform

from kornia.enhance import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    sharpness,
    solarize,
    equalize,
    posterize,
    invert2d,
)
from kornia.geometry import (
    translate,
    shear,
    rotate
)
from kornia.augmentation import AugmentationBase2D
from ..utils import _adapted_rsampling
from .utils import _cutout

IMAGENET_POLICY = OrderedDict(
    Sharpness=(sharpness, 0.1, 0.8),
    Solarize=(lambda inp, threshold: solarize(inp, threshold, None), 0, 1),
    SolarizeAdd=(lambda inp, additions: solarize(inp, 0.5, additions), 0., 0.5),
    Equalize=(equalize, None, None),
    Posterize=(posterize, 4, 8),
    Contrast=(adjust_contrast, 0.3, 1.1),
    Brightness=(adjust_brightness, -0.6, 0.6),
    Color=(adjust_saturation, 0.3, 1.0),
    Rotate=(lambda inp, angle: rotate(inp, angle, align_corners=True), -30, 30),
    ShearX=(lambda inp, shearX: shear(inp, torch.stack([shearX, torch.zeros_like(shearX)], dim=-1), True), -0.3, 0.3),
    ShearY=(lambda inp, shearY: shear(inp, torch.stack([torch.zeros_like(shearY), shearY], dim=-1), True), -0.3, 0.3),
    TranslateX=(lambda inp, transX: translate(
        inp, torch.stack([transX * inp.size(-2), torch.zeros_like(transX)], dim=-1), True), -0.3, 0.3),
    TranslateY=(lambda inp, transY: translate(
        inp, torch.stack([torch.zeros_like(transY), transY * inp.size(-1)], dim=-1), True), -0.3, 0.3),
    Invert=(invert2d, None, None),
    # TODO: Implement below
    AutoContrast=(lambda input: input, None, None),
    Cutout=(_cutout, 0., .3),
)


class RandAugment(AugmentationBase2D):
    r"""Applies the RandAugment policy to `image`.

    RandAugment is from the paper https://arxiv.org/abs/1909.13719.

    Args:
        N (int): the number of augmentation transformations to apply sequentially to an image.
            Usually best values will be in the range [1, 3]. Default: 2.
        M (int): shared magnitude across all augmentation operations.
            Usually best values are in the range [5, 30]. Default: [5, 30].
        policy (str or OrderedDict): policy to use. If `imagenet`, it will load pre-defined policy.
            If OrderedDict, it will be validated and loaded straight away. Default: 'imagenet'.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        p (float): probability for applying an augmentation. This param controls the augmentation probabilities
                   element-wisely for a batch.
        p_batch (float): probability for applying an augmentation to a batch. This param controls the augmentation
                         probabilities batch-wisely.

    Returns:
        torch.Tensor: The augmented version of `image`.

    Note:
        To maximize the efficiency, same policy will be applied in each batch even if same_on_batch == False.

    Example:
        >>> _ = torch.manual_seed(0)
        >>> randaug = RandAugment()
        >>> input = torch.randn(2, 3, 3, 5)
        >>> randaug(input)
        tensor([[[[0.0571, 0.0571, 0.0571, 0.0571, 0.0571],
                  [0.0571, 0.0571, 0.0571, 0.0000, 0.0571],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
        <BLANKLINE>
                 [[0.0571, 0.0571, 0.0000, 0.0571, 0.0571],
                  [0.0571, 0.0000, 0.0571, 0.0571, 0.0571],
                  [0.0000, 0.0571, 0.0000, 0.0000, 0.0571]],
        <BLANKLINE>
                 [[0.0571, 0.0000, 0.0000, 0.0000, 0.0571],
                  [0.0000, 0.0000, 0.0000, 0.0571, 0.0000],
                  [0.0571, 0.0571, 0.0571, 0.0000, 0.0571]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[0.3210, 0.5333, 0.3210, 0.3210, 0.9579],
                  [0.1087, 0.1087, 0.7456, 0.5333, 0.5333],
                  [0.5333, 0.5333, 0.7456, 0.5333, 0.3210]],
        <BLANKLINE>
                 [[0.5333, 0.7456, 0.7456, 0.5333, 0.5333],
                  [0.9579, 0.1087, 0.3210, 0.5333, 0.5333],
                  [0.5333, 0.3210, 0.3210, 0.5333, 0.1087]],
        <BLANKLINE>
                 [[0.3210, 0.9579, 0.7456, 0.9579, 0.5333],
                  [0.7456, 0.7456, 0.9579, 0.5333, 0.7456],
                  [0.7456, 0.1087, 0.5333, 0.3210, 0.1087]]]])
    """

    def __init__(self, N: int = 2, M: Tuple[int, int] = [5, 30], policy: Union[str, OrderedDict] = 'imagenet',
                 same_on_batch: bool = False, p: float = 0.8, p_batch: float = 1.) -> None:
        super(RandAugment, self).__init__(return_transform=False, same_on_batch=same_on_batch, p=p, p_batch=p_batch)
        self._MAX_M_ = 30
        self.N = N
        self.M = Uniform(M[0], M[1])
        self.load_policy(policy)

    def load_policy(self, policy: Union[str, OrderedDict] = 'imagenet') -> None:
        if isinstance(policy, (str)):
            if policy == 'imagenet':
                self.policy = IMAGENET_POLICY
            else:
                raise ValueError(f"Policy for {policy} is not yet defined.")
        elif isinstance(policy, (OrderedDict)):
            # TODO: validate policy format
            self.policy = policy
        else:
            raise ValueError(f"Policy must be either a string or an OrderedDict of augmentations. Got {policy}.")

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        m = _adapted_rsampling((batch_shape[0],), self.M, self.same_on_batch)
        policy_idx = torch.randperm(len(self.policy.keys()))[:self.N]
        return dict(
            policy_idx=policy_idx,
            m=m
        )

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        policy = [list(self.policy.items())[i] for i in params['policy_idx']]
        for name, (func, min_val, max_val) in policy:
            if min_val is None and max_val is None:
                input = func(input)
            else:
                val = (params['m'] / self._MAX_M_) * (max_val - min_val) + min_val
                input = func(input, val)
        return input
