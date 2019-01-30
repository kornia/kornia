from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Based on
# https://github.com/tensorflow/models/blob/master/research/struct2depth/model.py#L625-L641


class DiceLoss(nn.Module):
    r"""Criterion that computes Dice Coefficient loss.
    """
    def __init__(self,
            smooth: Optional[float] = 1.0,
            reduction: Optional[str] = 'none') -> None:
        super(DiceLoss, self).__init__()
        self.smooth: float = smooth
        self.reduction: str = reduction
        self.eps: float = 1e-6

    @staticmethod
    def make_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        r"""Converts an integer label tensor a one-hot labels tensor.                                                                                        
        Args:
            labels (torch.Tensor) : tensor with labels of shape :math:`(N, H, W)`,
                                    where N is batch siz. Each value is an integer
                                    representing correct classification.       
            num_classes (int): number of classes in labels.                                        
        Returns:
            torch.Tensor: the labels in one hot tensor.
        """
        if not torch.is_tensor(labels):
            raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                            .format(type(labels)))
        if not len(labels.shape) == 3:
            raise ValueError("Invalid depth shape, we expect BxHxW. Got: {}"
                             .format(labels.shape))
        if not labels.dtype == torch.int64:
            raise ValueError(
                "labels must be of the same dtype torch.int64. Got: {}" .format(
                    labels.dtype))
        if num_classes < 1:
            raise ValueError("The number of classes must be bigger than one. Got: {}"
                             .format(num_classes))
        batch_size, height, width = labels.shape
        one_hot = torch.zeros(
            batch_size, num_classes, height, width, device=labels.device)       
        return one_hot.scatter_(1, labels.unsqueeze(1), 1.0)                    

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, shape.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = self.make_one_hot(target, num_classes=input.shape[1])
        target_one_hot = target_one_hot.to(input.dtype)

        # compute the actual dice score
        numerator = torch.sum(input_soft * target_one_hot, dim=(1, 2, 3))
        denominator = torch.sum(input_soft + target_one_hot, dim=(1, 2, 3))

        dice_score_tmp = (2. * numerator + self.smooth) / \
                         (denominator + self.smooth + self.eps)
        
        # reducte loss
        dice_score = -1                                                        
        if self.reduction == 'none':                                           
            dice_score = dice_score_tmp                                        
        elif self.reduction == 'mean':                                         
            dice_score = torch.mean(dice_score_tmp)                            
        elif self.reduction == 'sum':
            dice_score = torch.sum(dice_score_tmp)                             
        else:                                                                  
            raise NotImplementedError("Invalid reduction type: {}."
                                      .format(self.reduction))                 
        return 1. - dice_score                                                 


######################
# functional interface
######################


def dice_loss(input: torch.Tensor, target: torch.Tensor, smooth: Optional[float] = 1.0, reduction: Optional[str] = 'none') -> torch.Tensor:
    r"""Computes image-aware depth smoothness loss.

    See :class:`~torchgeometry.losses.DiceLoss` for details.
    """
    return DiceLoss()(input, target)
