from typing import Tuple, Union, Dict

import torch
from torch.distributions import Uniform

from kornia.augmentation import AugmentationBase2D
from ..utils import _adapted_rsampling
from .utils import POLICY_FUNCS

# {policy_name: (min_val, max_val)}
IMAGENET_RANDAUG_POLICY = dict(
    sharpness=(0.1, 0.8),
    solarize=(0, 1),
    solarizeAdd=(0., 0.5),
    equalize=(None, None),
    posterize=(4, 8),
    contrast=(0.3, 1.1),
    brightness=(-0.6, 0.6),
    color=(0.3, 1.0),
    rotate=(-30, 30),
    shearX=(-0.3, 0.3),
    shearY=(-0.3, 0.3),
    translateX=(-0.3, 0.3),
    translateY=(-0.3, 0.3),
    invert=(None, None),
    autocontrast=(None, None),
    cutout=(0., .3),
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
        tensor([[[[0.0000, 0.0000, 0.0000, 0.0000, 0.6351],
                  [0.5178, 0.0000, 0.0000, 0.2412, 0.0000],
                  [0.2619, 0.2306, 0.0897, 0.9261, 0.8357]],
        <BLANKLINE>
                 [[0.0000, 0.0000, 0.0000, 0.4240, 0.5938],
                  [0.4481, 0.0000, 0.0000, 1.0000, 0.5614],
                  [0.0000, 0.0000, 0.1373, 1.0000, 1.0000]],
        <BLANKLINE>
                 [[0.7081, 0.0000, 0.0000, 0.0236, 0.0000],
                  [0.1859, 0.3290, 0.0841, 0.4795, 0.3301],
                  [0.0000, 0.5930, 0.0000, 0.0393, 0.3913]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[1.0000, 0.0000, 0.0000, 0.0000, 0.7129],
                  [0.8619, 0.1452, 0.0000, 0.0000, 0.4437],
                  [0.0000, 0.0000, 0.6077, 1.0000, 1.0000]],
        <BLANKLINE>
                 [[0.0000, 0.0000, 1.0000, 0.0000, 0.0000],
                  [0.7514, 0.9072, 1.0000, 0.0000, 1.0000],
                  [0.0000, 0.2740, 0.0000, 0.0000, 0.9545]],
        <BLANKLINE>
                 [[0.3222, 1.0000, 0.6464, 0.0000, 0.0000],
                  [1.0000, 0.0000, 0.7892, 1.0000, 0.0000],
                  [0.0000, 0.0812, 0.3608, 0.2017, 0.0628]]]])
    """

    def __init__(self, N: int = 2, M: Tuple[int, int] = [5, 30], policy: Union[str, dict] = 'imagenet',
                 same_on_batch: bool = False, p: float = 0.8, p_batch: float = 1.) -> None:
        super(RandAugment, self).__init__(return_transform=False, same_on_batch=same_on_batch, p=p, p_batch=p_batch)
        self._MAX_M_ = 30
        self.N = N
        self.M = Uniform(M[0], M[1])
        self.load_policy(policy)

    def load_policy(self, policy: Union[str, dict] = 'imagenet') -> None:
        if isinstance(policy, (str)):
            if policy == 'imagenet':
                self.policy = IMAGENET_RANDAUG_POLICY
            else:
                raise ValueError(f"Policy for {policy} is not yet defined.")
        elif isinstance(policy, (dict)):
            # TODO: validate policy format
            self.policy = policy
        else:
            raise ValueError(f"Policy must be either a string or an dict of augmentations. Got {policy}.")

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        m = _adapted_rsampling((batch_shape[0],), self.M, self.same_on_batch)
        policy_idx = torch.randperm(len(self.policy.keys()))[:self.N]
        return dict(
            policy_idx=policy_idx,
            m=m
        )

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        policy = [list(self.policy.items())[i] for i in params['policy_idx']]
        for name, (min_val, max_val) in policy:
            func = POLICY_FUNCS[name]
            if min_val is None and max_val is None:
                input = func(input)
            else:
                # Ref: https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py#261
                val = (params['m'] / self._MAX_M_) * (max_val - min_val) + min_val
                input = func(input, val)
        return input
