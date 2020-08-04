from typing import Callable, Tuple, Union, List, Optional, Dict, cast

import torch
import torch.nn as nn
from torch.nn.functional import pad

from kornia.augmentation.augmentation import AugmentationBase
from kornia.constants import Resample, BorderType
import kornia.augmentation.functional as F
import kornia.augmentation.random_generator as rg
from kornia.augmentation.utils import (
    _adapted_uniform,
    _infer_batch_shape
)


class AugmentationWithLabelBase(AugmentationBase):
    r"""AugmentationWithLabelBase base class for customized augmentation implementations. For any augmentation,
    the implementation of "generate_parameters" and "apply_transform" are required while the
    "compute_transformation" is only required when passing "return_transform" as True.

    Args:
        return_transform (bool): if ``True`` return the matrix describing the geometric transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.

    """
    def apply_transform(self, input: torch.Tensor, label: torch.Tensor,     # type: ignore
                        params: Dict[str, torch.Tensor]) -> (torch.Tensor, torch.Tensor):   # type: ignore
        raise NotImplementedError

    def forward(self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],  # type: ignore
                label: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None,
                return_transform: Optional[bool] = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore
        if return_transform is None:
            return_transform = self.return_transform
        if params is None:
            batch_shape = self.infer_batch_shape(input)
            self._params = self.generate_parameters(batch_shape)
        else:
            self._params = params

        if isinstance(input, tuple):
            output = self.apply_transform(input[0], label, self._params)
            transformation_matrix = self.compute_transformation(input[0], self._params)
            if return_transform:
                return output, input[1] @ transformation_matrix
            else:
                return output, input[1]

        output = self.apply_transform(input, label, self._params)
        if return_transform:
            transformation_matrix = self.compute_transformation(input, self._params)
            return output, transformation_matrix
        return output


class RandomMixUp(AugmentationWithLabelBase):
    """
    Implemention for `mixup: BEYOND EMPIRICAL RISK MINIMIZATION <https://arxiv.org/pdf/1710.09412.pdf>`.
    The function returns (inputs, labels), in which the inputs is the tensor that contains the mixup images
    while the labels is a :math:`(B, 3)` tensor that contains (label_a, label_b, lambda) for each image.
    The implementation is on top of `https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py`.
    The loss and accuracy are computed as:
        ```
        def loss_mixup(y, logits):
            criterion = F.cross_entropy
            loss_a = criterion(logits, y[:, 0].long(), reduction='none')
            loss_b = criterion(logits, y[:, 1].long(), reduction='none')
            return ((1 - y[:, 2]) * loss_a + y[:, 2] * loss_b).mean()
        
        def acc_mixup(y, logits):
            pred = torch.argmax(logits, dim=1).to(y.device)
            return (1 - y[:, 2]) * pred.eq(y[:, 0]).float() + y[:, 2] * pred.eq(y[:, 1]).float()
        ```

    Args:
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated

    Shape:
        - Input: :math:`(B, C, H, W)`, :math:`(B,)`
        - Output: :math:`(B, C, H, W)`, :math:`(B, 3)`

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
                  [0.4550, 0.5725, 0.4980]]]]), tensor([[0.0000, 0.0000, 0.6556],
                [1.0000, 1.0000, 0.3138]]))
    """
    def __init__(self, return_transform: bool = False) -> None:
        super(RandomMixUp, self).__init__(return_transform)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(return_transform={self.return_transform})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_mixup_generator(batch_shape[0])

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_intensity_transformation(input, params)

    def apply_transform(self, input: torch.Tensor, label: torch.Tensor,  # type: ignore
                        params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        return F.apply_mixup(input, label, params)
