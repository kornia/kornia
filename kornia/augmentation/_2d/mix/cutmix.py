import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.mix.base import MixAugmentationBase, MixAugmentationBaseV2
from kornia.augmentation.utils import _shape_validation
from kornia.constants import DataKey
from kornia.core import Tensor
from kornia.geometry.bbox import bbox_to_mask, infer_bbox_shape


class RandomCutMix(MixAugmentationBase):
    r"""Apply CutMix augmentation to a batch of tensor images.

    .. image:: _static/img/RandomCutMix.png

    Implementation for `CutMix: Regularization Strategy to Train Strong Classifiers with
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
        height: the width of the input image.
        width: the width of the input image.
        p: probability for applying an augmentation to a batch. This param controls the augmentation
                   probabilities batch-wisely.
        num_mix: cut mix times. Default is 1.
        beta: hyperparameter for generating cut size from beta distribution.
            Beta cannot be set to 0 after torch 1.8.0. If None, it will be set to 1.
        cut_size: controlling the minimum and maximum cut ratio from [0, 1].
            If None, it will be set to [0, 1], which means no restriction.
        same_on_batch: apply the same transformation across the batch.
            This flag will not maintain permutation order. Default: False.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False

    Inputs:
        - Input image tensors, shape of :math:`(B, C, H, W)`.
        - Raw labels, shape of :math:`(B)`.

    Returns:
        Tuple[Tensor, Tensor]:
        - Adjusted image, shape of :math:`(B, C, H, W)`.
        - Raw labels, permuted labels and lambdas for each mix, shape of :math:`(B, num_mix, 3)`.

    Note:
        This implementation would randomly cutmix images in a batch. Ideally, the larger batch size would be preferred.

    Examples:
        >>> rng = torch.manual_seed(3)
        >>> input = torch.rand(2, 1, 3, 3)
        >>> input[0] = torch.ones((1, 3, 3))
        >>> label = torch.tensor([0, 1])
        >>> cutmix = RandomCutMix()
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
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_mix: int = 1,
        cut_size: Optional[Union[Tensor, Tuple[float, float]]] = None,
        beta: Optional[Union[Tensor, float]] = None,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=1.0, p_batch=p, same_on_batch=same_on_batch, keepdim=keepdim)
        if height is not None or width is not None:
            warnings.warn(
                "height and width can be inferred automatically now. "
                "The height and width arguments will be removed finally.",
                category=DeprecationWarning,
            )
        self._param_generator = cast(rg.CutmixGenerator, rg.CutmixGenerator(cut_size, beta, num_mix, p=p))

    def apply_transform(  # type: ignore
        self, input: Tensor, label: Tensor, params: Dict[str, Tensor]  # type: ignore
    ) -> Tuple[Tensor, Tensor]:
        height, width = input.size(2), input.size(3)
        num_mixes = params["mix_pairs"].size(0)
        batch_size = params["mix_pairs"].size(1)

        _shape_validation(params["mix_pairs"], [num_mixes, batch_size], "mix_pairs")
        _shape_validation(params["crop_src"], [num_mixes, batch_size, 4, 2], "crop_src")

        out_inputs = input.clone()
        out_labels = []
        for pair, crop in zip(params["mix_pairs"], params["crop_src"]):
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


class RandomCutMixV2(MixAugmentationBaseV2):
    r"""Apply CutMix augmentation to a batch of tensor images.

    .. image:: _static/img/RandomCutMix.png

    Implementation for `CutMix: Regularization Strategy to Train Strong Classifiers with
    Localizable Features` :cite:`yun2019cutmix`.

    The function returns (inputs, labels), in which the inputs is the tensor that contains the mixup images
    while the labels is a :math:`(\text{num_mixes}, B, 3)` tensor that contains (label_permuted_batch, lambda)
    for each cutmix.

    The implementation referred to the following repository: `https://github.com/clovaai/CutMix-PyTorch
    <https://github.com/clovaai/CutMix-PyTorch>`_.

    Args:
        height: the width of the input image.
        width: the width of the input image.
        p: probability for applying an augmentation to a batch. This param controls the augmentation
                   probabilities batch-wisely.
        num_mix: cut mix times. Default is 1.
        beta: hyperparameter for generating cut size from beta distribution.
            Beta cannot be set to 0 after torch 1.8.0. If None, it will be set to 1.
        cut_size: controlling the minimum and maximum cut ratio from [0, 1].
            If None, it will be set to [0, 1], which means no restriction.
        same_on_batch: apply the same transformation across the batch.
            This flag will not maintain permutation order. Default: False.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False

    Inputs:
        - Input image tensors, shape of :math:`(B, C, H, W)`.
        - Raw labels, shape of :math:`(B)`.

    Returns:
        Tuple[Tensor, Tensor]:
        - Adjusted image, shape of :math:`(B, C, H, W)`.
        - Raw labels, permuted labels and lambdas for each mix, shape of :math:`(B, num_mix, 3)`.

    Note:
        This implementation would randomly cutmix images in a batch. Ideally, the larger batch size would be preferred.

    Examples:
        >>> rng = torch.manual_seed(3)
        >>> input = torch.rand(2, 1, 3, 3)
        >>> input[0] = torch.ones((1, 3, 3))
        >>> label = torch.tensor([0, 1])
        >>> cutmix = RandomCutMixV2(data_keys=["input", "class"])
        >>> cutmix(input, label)
        [tensor([[[[0.8879, 0.4510, 1.0000],
                  [0.1498, 0.4015, 1.0000],
                  [1.0000, 1.0000, 1.0000]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[1.0000, 1.0000, 0.7995],
                  [1.0000, 1.0000, 0.0542],
                  [0.4594, 0.1756, 0.9492]]]]), tensor([[[0.0000, 1.0000, 0.4444],
                 [1.0000, 0.0000, 0.4444]]])]
    """

    def __init__(
        self,
        num_mix: int = 1,
        cut_size: Optional[Union[Tensor, Tuple[float, float]]] = None,
        beta: Optional[Union[Tensor, float]] = None,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
        data_keys: List[Union[str, int, DataKey]] = [DataKey.INPUT],
    ) -> None:
        super().__init__(p=1.0, p_batch=p, same_on_batch=same_on_batch, keepdim=keepdim, data_keys=data_keys)
        self._param_generator = cast(rg.CutmixGenerator, rg.CutmixGenerator(cut_size, beta, num_mix, p=p))

    def apply_transform_class(
        self, label: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]
    ) -> Tensor:
        height, width = params["image_shape"]

        out_labels = []
        for pair, crop in zip(params["mix_pairs"], params["crop_src"]):
            labels_permute = label.index_select(dim=0, index=pair.to(label.device))
            w, h = infer_bbox_shape(crop)
            lam = w.to(label.dtype) * h.to(label.dtype) / (width * height)  # width_beta * height_beta
            out_labels.append(
                torch.stack([label.to(label.dtype), labels_permute.to(label.dtype), lam.to(label.device)], dim=1)
            )

        return torch.stack(out_labels, dim=0)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]
    ) -> Tensor:
        height, width = input.size(2), input.size(3)

        out_inputs = input.clone()
        for pair, crop in zip(params["mix_pairs"], params["crop_src"]):
            input_permute = input.index_select(dim=0, index=pair.to(input.device))
            # compute mask to match input shape
            mask = bbox_to_mask(crop, width, height).bool().unsqueeze(dim=1).repeat(1, input.size(1), 1, 1)
            out_inputs[mask] = input_permute[mask]

        return out_inputs
