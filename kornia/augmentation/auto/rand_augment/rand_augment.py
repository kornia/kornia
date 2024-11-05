from typing import Dict, Iterator, List, Optional, Tuple, Union, cast

import torch

from kornia.augmentation.auto.base import SUBPOLICY_CONFIG, PolicyAugmentBase
from kornia.augmentation.auto.operations import OperationBase
from kornia.augmentation.auto.operations.policy import PolicySequential
from kornia.augmentation.container.params import ParamItem
from kornia.core import Module, Tensor

from . import ops

default_policy: List[SUBPOLICY_CONFIG] = [
    [("auto_contrast", 0, 1)],
    [("equalize", 0, 1)],
    [("invert", 0, 1)],
    [("rotate", -30.0, 30.0)],
    [("posterize", 0.0, 4)],
    [("solarize", 0.0, 1.0)],
    [("solarize_add", 0.0, 0.43)],
    [("color", 0.1, 1.9)],
    [("contrast", 0.1, 1.9)],
    [("brightness", 0.1, 1.9)],
    [("sharpness", 0.1, 1.9)],
    [("shear_x", -0.3, 0.3)],
    [("shear_y", -0.3, 0.3)],
    # (CutoutAbs, 0, 40),
    [("translate_x", -0.1, 0.1)],
    [("translate_y", -0.1, 0.1)],
]


class RandAugment(PolicyAugmentBase):
    """Apply RandAugment :cite:`cubuk2020randaugment` augmentation strategies.

    Args:
        n: the number of augmentations to apply sequentially.
        m: magnitude for all the augmentations, ranged from [0, 30].
        policy: candidate transformations. If None, a default candidate list will be used.
        transformation_matrix_mode: computation mode for the chained transformation matrix, via `.transform_matrix`
                                    attribute.
                                    If `silent`, transformation matrix will be computed silently and the non-rigid
                                    modules will be ignored as identity transformations.
                                    If `rigid`, transformation matrix will be computed silently and the non-rigid
                                    modules will trigger errors.
                                    If `skip`, transformation matrix will be totally ignored.

    Examples:
        >>> import kornia.augmentation as K
        >>> in_tensor = torch.rand(5, 3, 30, 30)
        >>> aug = K.AugmentationSequential(RandAugment(n=2, m=10))
        >>> aug(in_tensor).shape
        torch.Size([5, 3, 30, 30])
    """

    def __init__(
        self,
        n: int,
        m: int,
        policy: Optional[List[SUBPOLICY_CONFIG]] = None,
        transformation_matrix_mode: str = "silent",
    ) -> None:
        if m <= 0 or m >= 30:
            raise ValueError(f"Expect `m` in [0, 30]. Got {m}.")

        if policy is None:
            _policy = default_policy
        else:
            _policy = policy

        super().__init__(_policy, transformation_matrix_mode=transformation_matrix_mode)
        self.n = n
        self.m = m

    def rand_selector(self, n: int) -> Tensor:
        perm = torch.randperm(len(self._modules))
        idx = perm[:n]
        return idx

    def compose_subpolicy_sequential(self, subpolicy: SUBPOLICY_CONFIG) -> PolicySequential:
        if len(subpolicy) != 1:
            raise RuntimeError(f"Each policy must have only one operation for RandAugment. Got {len(subpolicy)}.")
        name, low, high = subpolicy[0]
        return PolicySequential(*[getattr(ops, name)(low, high)])

    def get_forward_sequence(self, params: Optional[List[ParamItem]] = None) -> Iterator[Tuple[str, Module]]:
        if params is None:
            idx = self.rand_selector(
                self.n,
            )
            return self.get_children_by_indices(idx)

        return self.get_children_by_params(params)

    def forward_parameters(self, batch_shape: torch.Size) -> List[ParamItem]:
        named_modules: Iterator[Tuple[str, Module]] = self.get_forward_sequence()
        params: List[ParamItem] = []
        mod_param: Union[Dict[str, Tensor], List[ParamItem]]
        m = torch.tensor([self.m / 30] * batch_shape[0])

        for name, module in named_modules:
            # The Input PolicySequential only got one child.
            op = cast(PolicySequential, module)[0]
            op = cast(OperationBase, op)
            mag = None
            if op.magnitude_range is not None:
                minval, maxval = op.magnitude_range
                mag = m * float(maxval - minval) + minval
            mod_param = op.forward_parameters(batch_shape, mag=mag)
            # Compose it
            param = ParamItem(name, [ParamItem(next(iter(module.named_children()))[0], mod_param)])
            params.append(param)

        return params
