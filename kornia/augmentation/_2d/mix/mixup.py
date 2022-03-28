from typing import Dict, Optional, Tuple, Union, cast

import torch

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.mix.base import MixAugmentationBase


class RandomMixUp(MixAugmentationBase):
    r"""Apply MixUp augmentation to a batch of tensor images.

    .. image:: _static/img/RandomMixUp.png

    Implementation for `mixup: BEYOND EMPIRICAL RISK MINIMIZATION` :cite:`zhang2018mixup`.

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

    def __init__(
        self,
        lambda_val: Optional[Union[torch.Tensor, Tuple[float, float]]] = None,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=1.0, p_batch=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator = cast(rg.MixupGenerator, rg.MixupGenerator(lambda_val, p=p))

    def apply_transform(  # type: ignore
        self, input: torch.Tensor, label: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_permute = input.index_select(dim=0, index=params["mixup_pairs"].to(input.device))
        labels_permute = label.index_select(dim=0, index=params["mixup_pairs"].to(label.device))

        lam = params["mixup_lambdas"].view(-1, 1, 1, 1).expand_as(input).to(label.device)
        inputs = input * (1 - lam) + input_permute * lam
        out_labels = torch.stack(
            [
                label.to(input.dtype),
                labels_permute.to(input.dtype),
                params["mixup_lambdas"].to(label.device, input.dtype),
            ],
            dim=-1,
        ).to(label.device)
        return inputs, out_labels
