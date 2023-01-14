from typing import Any, Dict, List, Optional, Tuple, Union

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.mix.base import MixAugmentationBaseV2
from kornia.constants import DataKey, DType
from kornia.core import Tensor, stack, zeros


class RandomMixUpV2(MixAugmentationBaseV2):
    r"""Apply MixUp augmentation to a batch of tensor images.

    .. image:: _static/img/RandomMixUpV2.png

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
        p: probability for applying an augmentation to a batch. This param controls the augmentation
                   probabilities batch-wisely.
        lambda_val: min-max value of mixup strength. Default is 0-1.
        same_on_batch: apply the same transformation across the batch.
            This flag will not maintain permutation order.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False).

    Inputs:
        - Input image tensors, shape of :math:`(B, C, H, W)`.
        - Label: raw labels, shape of :math:`(B)`.

    Returns:
        Tuple[Tensor, Tensor]:
        - Adjusted image, shape of :math:`(B, C, H, W)`.
        - Raw labels, permuted labels and lambdas for each mix, shape of :math:`(B, 3)`.

    Note:
        This implementation would randomly mixup images in a batch. Ideally, the larger batch size would be preferred.

    Examples:
        >>> rng = torch.manual_seed(1)
        >>> input = torch.rand(2, 1, 3, 3)
        >>> label = torch.tensor([0, 1])
        >>> mixup = RandomMixUpV2(data_keys=["input", "class"])
        >>> mixup(input, label)
        [tensor([[[[0.7576, 0.2793, 0.4031],
                  [0.7347, 0.0293, 0.7999],
                  [0.3971, 0.7544, 0.5695]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[0.4388, 0.6387, 0.5247],
                  [0.6826, 0.3051, 0.4635],
                  [0.4550, 0.5725, 0.4980]]]]), tensor([[0.0000, 0.0000, 0.1980],
                [1.0000, 1.0000, 0.4162]])]
    """

    def __init__(
        self,
        lambda_val: Optional[Union[Tensor, Tuple[float, float]]] = None,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
        data_keys: List[Union[str, int, DataKey]] = [DataKey.INPUT],
    ) -> None:
        super().__init__(p=1.0, p_batch=p, same_on_batch=same_on_batch, keepdim=keepdim, data_keys=data_keys)
        self._param_generator = rg.MixupGenerator(lambda_val, p=p)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], maybe_flags: Optional[Dict[str, Any]] = None
    ) -> Tensor:
        input_permute = input.index_select(dim=0, index=params["mixup_pairs"].to(input.device))

        lam = params["mixup_lambdas"].view(-1, 1, 1, 1).expand_as(input).to(input.device)
        inputs = input * (1 - lam) + input_permute * lam
        return inputs

    def apply_non_transform_class(
        self, input: Tensor, params: Dict[str, Tensor], maybe_flags: Optional[Dict[str, Any]] = None
    ) -> Tensor:
        out_labels = stack(
            [
                input.to(device=input.device, dtype=DType.to_torch(int(params["dtype"].item()))),
                input.to(device=input.device, dtype=DType.to_torch(int(params["dtype"].item()))),
                zeros((len(input),), device=input.device, dtype=DType.to_torch(int(params["dtype"].item()))),
            ],
            -1,
        )
        return out_labels

    def apply_transform_class(
        self, input: Tensor, params: Dict[str, Tensor], maybe_flags: Optional[Dict[str, Any]] = None
    ) -> Tensor:
        labels_permute = input.index_select(dim=0, index=params["mixup_pairs"].to(input.device))

        out_labels = stack(
            [
                input.to(device=input.device, dtype=DType.to_torch(int(params["dtype"].item()))),
                labels_permute.to(device=input.device, dtype=DType.to_torch(int(params["dtype"].item()))),
                params["mixup_lambdas"].to(device=input.device, dtype=DType.to_torch(int(params["dtype"].item()))),
            ],
            -1,
        )
        return out_labels
