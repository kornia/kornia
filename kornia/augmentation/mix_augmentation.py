from typing import Callable, Tuple, Union, List, Optional, Dict, cast

import torch
import torch.nn as nn
from torch.nn.functional import pad

from kornia.constants import Resample, BorderType
from . import functional as F
from . import random_generator as rg
from .base import MixAugmentationBase
from .utils import (
    _infer_batch_shape
)


class RandomMixUp(MixAugmentationBase):
    r"""Apply MixUp augmentation to a batch of tensor images.

    Implemention for `mixup: BEYOND EMPIRICAL RISK MINIMIZATION` :cite:`zhang2018mixup`.

    The function returns (inputs, labels), in which the inputs is the tensor that contains the mixup images
    while the labels is a :math:`(B, 3)` tensor that contains (label_batch, label_permuted_batch, lambda) for
    each image.

    The implementation is on top of the following repository:
    `https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py
    <https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py>`_.

    The loss and accuracy are computed as:

    .. code-block:: python

        def loss_mixup(y, logits):
            criterion = F.cross_entropy
            loss_a = criterion(logits, y[:, 0].long(), reduction='none')
            loss_b = criterion(logits, y[:, 1].long(), reduction='none')
            return ((1 - y[:, 2]) * loss_a + y[:, 2] * loss_b).mean()

    .. code-block:: python

        def acc_mixup(y, logits):
            pred = torch.argmax(logits, dim=1).to(y.device)
            return (1 - y[:, 2]) * pred.eq(y[:, 0]).float() + y[:, 2] * pred.eq(y[:, 1]).float()

    Args:
        p (float): probability for applying an augmentation to a batch. This param controls the augmentation
                   probabilities batch-wisely.
        lambda_val (float or torch.Tensor, optional): min-max value of mixup strength. Default is 0-1.
        same_on_batch (bool): apply the same transformation across the batch.
            This flag will not maintain permutation order. Default: False.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False

    Inputs:
        - Input image tensors, shape of :math:`(B, C, H, W)`.
        - Label: raw labels, shape of :math:`(B)`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
        - Adjusted image, shape of :math:`(B, C, H, W)`.
        - Raw labels, permuted labels and lambdas for each mix, shape of :math:`(B, 3)`.

    Note:
        This implementation would randomly mixup images in a batch. Ideally, the larger batch size would be preferred.

    Examples:
        >>> rng = torch.manual_seed(1)
        >>> input = torch.rand(2, 1, 3, 3)
        >>> label = torch.tensor([0, 1])
        >>> mixup = RandomMixUp()
        >>> mixup(input, label)
        (tensor([[[[0.7576, 0.2793, 0.4031],
                  [0.7347, 0.0293, 0.7999],
                  [0.3971, 0.7544, 0.5695]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[0.4388, 0.6387, 0.5247],
                  [0.6826, 0.3051, 0.4635],
                  [0.4550, 0.5725, 0.4980]]]]), tensor([[0.0000, 0.0000, 0.1980],
                [1.0000, 1.0000, 0.4162]]))
    """

    def __init__(self, lambda_val: Optional[Union[torch.Tensor, Tuple[float, float]]] = None,
                 same_on_batch: bool = False, p: float = 1.0, keepdim: bool = False) -> None:
        super(RandomMixUp, self).__init__(p=1., p_batch=p, same_on_batch=same_on_batch, keepdim=keepdim)
        if lambda_val is None:
            self.lambda_val = torch.tensor([0, 1.])
        else:
            self.lambda_val = \
                cast(torch.Tensor, lambda_val) if isinstance(lambda_val, torch.Tensor) else torch.tensor(lambda_val)

    def __repr__(self) -> str:
        repr = f"lambda_val={self.lambda_val}"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_mixup_generator(batch_shape[0], self.p, self.lambda_val, same_on_batch=self.same_on_batch)

    def apply_transform(self, input: torch.Tensor, label: torch.Tensor,  # type: ignore
                        params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        return F.apply_mixup(input, label, params)


class RandomCutMix(MixAugmentationBase):
    r"""Apply CutMix augmentation to a batch of tensor images.

    Implemention for `CutMix: Regularization Strategy to Train Strong Classifiers with
    Localizable Features` :cite:`yun2019cutmix`.

    The function returns (inputs, labels), in which the inputs is the tensor that contains the mixup images
    while the labels is a :math:`(\text{num_mixes}, B, 3)` tensor that contains (label_permuted_batch, lambda)
    for each cutmix.

    The implementation referred to the following repository: `https://github.com/clovaai/CutMix-PyTorch
    <https://github.com/clovaai/CutMix-PyTorch>`_.

    The onehot label may be computed as:

    .. code-block:: python

        def onehot(size, target):
            vec = torch.zeros(size, dtype=torch.float32)
            vec[target] = 1.
            return vec

    .. code-block:: python

        def cutmix_label(labels, out_labels, size):
            lb_onehot = onehot(size, labels)
            for out_label in out_labels:
                label_permuted_batch, lam = out_label[:, 0], out_label[:, 1]
                label_permuted_onehot = onehot(size, label_permuted_batch)
                lb_onehot = lb_onehot * lam + label_permuted_onehot * (1. - lam)
            return lb_onehot

    Args:
        height (int): the width of the input image.
        width (int): the width of the input image.
        p (float): probability for applying an augmentation to a batch. This param controls the augmentation
                   probabilities batch-wisely.
        num_mix (int): cut mix times. Default is 1.
        beta (float or torch.Tensor, optional): hyperparameter for generating cut size from beta distribution.
            If None, it will be set to 1.
        cut_size ((float, float) or torch.Tensor, optional): controlling the minimum and maximum cut ratio from [0, 1].
            If None, it will be set to [0, 1], which means no restriction.
        same_on_batch (bool): apply the same transformation across the batch.
            This flag will not maintain permutation order. Default: False.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False

    Inputs:
        - Input image tensors, shape of :math:`(B, C, H, W)`.
        - Raw labels, shape of :math:`(B)`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
        - Adjusted image, shape of :math:`(B, C, H, W)`.
        - Raw labels, permuted labels and lambdas for each mix, shape of :math:`(B, num_mix, 3)`.

    Note:
        This implementation would randomly cutmix images in a batch. Ideally, the larger batch size would be preferred.

    Examples:
        >>> rng = torch.manual_seed(3)
        >>> input = torch.rand(2, 1, 3, 3)
        >>> input[0] = torch.ones((1, 3, 3))
        >>> label = torch.tensor([0, 1])
        >>> cutmix = RandomCutMix(3, 3)
        >>> cutmix(input, label)
        (tensor([[[[0.8879, 0.4510, 1.0000],
                  [0.1498, 0.4015, 1.0000],
                  [1.0000, 1.0000, 1.0000]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[1.0000, 1.0000, 0.7995],
                  [1.0000, 1.0000, 0.0542],
                  [0.4594, 0.1756, 0.9492]]]]), tensor([[[0.0000, 1.0000, 0.4444],
                 [1.0000, 0.0000, 0.4444]]]))
    """

    def __init__(self, height: int, width: int, num_mix: int = 1,
                 cut_size: Optional[Union[torch.Tensor, Tuple[float, float]]] = None,
                 beta: Optional[Union[torch.Tensor, float]] = None, same_on_batch: bool = False,
                 p: float = 1., keepdim: bool = False) -> None:
        super(RandomCutMix, self).__init__(p=1., p_batch=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.height = height
        self.width = width
        self.num_mix = num_mix
        if beta is None:
            self.beta = torch.tensor(1.)
        else:
            self.beta = cast(torch.Tensor, beta) if isinstance(beta, torch.Tensor) else torch.tensor(beta)
        if cut_size is None:
            self.cut_size = torch.tensor([0., 1.])
        else:
            self.cut_size = \
                cast(torch.Tensor, cut_size) if isinstance(cut_size, torch.Tensor) else torch.tensor(cut_size)

    def __repr__(self) -> str:
        repr = f"num_mix={self.num_mix}, beta={self.beta}, cut_size={self.cut_size}, "
        f"height={self.height}, width={self.width}"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_cutmix_generator(batch_shape[0], width=self.width, height=self.height, p=self.p,
                                          cut_size=self.cut_size, num_mix=self.num_mix, beta=self.beta,
                                          same_on_batch=self.same_on_batch)

    def apply_transform(self, input: torch.Tensor, label: torch.Tensor,  # type: ignore
                        params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        return F.apply_cutmix(input, label, params)
