from typing import cast, Dict, Optional, Tuple, Union

import torch

from kornia.geometry.bbox import bbox_to_mask, infer_bbox_shape

from . import random_generator as rg
from .base import MixAugmentationBase
from .utils import _shape_validation


class RandomMixUp(MixAugmentationBase):
    r"""Apply MixUp augmentation to a batch of tensor images.

    .. image:: _static/img/RandomMixUp.png

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

    def __init__(
        self,
        lambda_val: Optional[Union[torch.Tensor, Tuple[float, float]]] = None,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super(RandomMixUp, self).__init__(p=1.0, p_batch=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.lambda_val = lambda_val

    def __repr__(self) -> str:
        repr = f"lambda_val={self.lambda_val}"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        if self.lambda_val is None:
            lambda_val = torch.tensor([0.0, 1.0], device=self.device, dtype=self.dtype)
        else:
            lambda_val = (
                cast(torch.Tensor, self.lambda_val)
                if isinstance(self.lambda_val, torch.Tensor)
                else torch.tensor(self.lambda_val, device=self.device, dtype=self.dtype)
            )
        return rg.random_mixup_generator(batch_shape[0], self.p, lambda_val, same_on_batch=self.same_on_batch)

    def apply_transform(  # type: ignore
        self, input: torch.Tensor, label: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_permute = input.index_select(dim=0, index=params['mixup_pairs'].to(input.device))
        labels_permute = label.index_select(dim=0, index=params['mixup_pairs'].to(label.device))

        lam = params['mixup_lambdas'].view(-1, 1, 1, 1).expand_as(input).to(label.device)
        inputs = input * (1 - lam) + input_permute * lam
        out_labels = torch.stack(
            [
                label.to(input.dtype),
                labels_permute.to(input.dtype),
                params['mixup_lambdas'].to(label.device, input.dtype),
            ],
            dim=-1,
        ).to(label.device)
        return inputs, out_labels


class RandomCutMix(MixAugmentationBase):
    r"""Apply CutMix augmentation to a batch of tensor images.

    .. image:: _static/img/RandomCutMix.png

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
            Beta cannot be set to 0 after torch 1.8.0. If None, it will be set to 1.
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

    def __init__(
        self,
        height: int,
        width: int,
        num_mix: int = 1,
        cut_size: Optional[Union[torch.Tensor, Tuple[float, float]]] = None,
        beta: Optional[Union[torch.Tensor, float]] = None,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super(RandomCutMix, self).__init__(p=1.0, p_batch=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.height = height
        self.width = width
        self.num_mix = num_mix
        self.beta = beta
        self.cut_size = cut_size

    def __repr__(self) -> str:
        repr = (
            f"num_mix={self.num_mix}, beta={self.beta}, cut_size={self.cut_size}, "
            f"height={self.height}, width={self.width}"
        )
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        if self.beta is None:
            beta = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        else:
            beta = (
                cast(torch.Tensor, self.beta)
                if isinstance(self.beta, torch.Tensor)
                else torch.tensor(self.beta, device=self.device, dtype=self.dtype)
            )
        if self.cut_size is None:
            cut_size = torch.tensor([0.0, 1.0], device=self.device, dtype=self.dtype)
        else:
            cut_size = (
                cast(torch.Tensor, self.cut_size)
                if isinstance(self.cut_size, torch.Tensor)
                else torch.tensor(self.cut_size, device=self.device, dtype=self.dtype)
            )
        return rg.random_cutmix_generator(
            batch_shape[0],
            width=self.width,
            height=self.height,
            p=self.p,
            cut_size=cut_size,
            num_mix=self.num_mix,
            beta=beta,
            same_on_batch=self.same_on_batch,
        )

    def apply_transform(  # type: ignore
        self, input: torch.Tensor, label: torch.Tensor, params: Dict[str, torch.Tensor]  # type: ignore
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        height, width = input.size(2), input.size(3)
        num_mixes = params['mix_pairs'].size(0)
        batch_size = params['mix_pairs'].size(1)

        _shape_validation(params['mix_pairs'], [num_mixes, batch_size], 'mix_pairs')
        _shape_validation(params['crop_src'], [num_mixes, batch_size, 4, 2], 'crop_src')

        out_inputs = input.clone()
        out_labels = []
        for pair, crop in zip(params['mix_pairs'], params['crop_src']):
            input_permute = input.index_select(dim=0, index=pair.to(input.device))
            labels_permute = label.index_select(dim=0, index=pair.to(label.device))
            w, h = infer_bbox_shape(crop)
            lam = w.to(input.dtype) * h.to(input.dtype) / (width * height)  # width_beta * height_beta
            # compute mask to match input shape
            mask = bbox_to_mask(crop, width, height).bool().unsqueeze(dim=1).repeat(1, input.size(1), 1, 1)
            out_inputs[mask] = input_permute[mask]
            out_labels.append(
                torch.stack([label.to(input.dtype), labels_permute.to(input.dtype), lam.to(label.device)], dim=1)
            )

        return out_inputs, torch.stack(out_labels, dim=0)
