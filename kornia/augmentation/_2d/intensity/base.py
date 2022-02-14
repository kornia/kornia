from typing import Dict

import torch

from kornia.augmentation._2d.base import AugmentationBase2D


class IntensityAugmentationBase2D(AugmentationBase2D):
    r"""IntensityAugmentationBase2D base class for customized intensity augmentation implementations.

    For any augmentation, the implementation of "generate_parameters" and "apply_transform" are required while the
    "compute_transformation" is only required when passing "return_transform" as True.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        return_transform: if ``True`` return the matrix describing the geometric transformation applied to each
          input tensor. If ``False`` and the input is a tuple the applied transformation  won't be concatenated.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.
    """

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.identity_matrix(input)
