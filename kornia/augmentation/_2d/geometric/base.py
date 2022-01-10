import warnings
from typing import Dict, Optional, Tuple, cast

import torch

from kornia.augmentation._2d.base import AugmentationBase2D
from kornia.augmentation.base import TensorWithTransformMat
from kornia.augmentation.utils import _transform_output_shape
from kornia.utils.helpers import _torch_inverse_cast


class GeometricAugmentationBase2D(AugmentationBase2D):
    r"""GeometricAugmentationBase2D base class for customized geometric augmentation implementations.

    For any augmentation, the implementation of "generate_parameters" and "apply_transform" are required while the
    "compute_transformation" is only required when passing "return_transform" as True.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        return_transform: if ``True`` return the matrix describing the geometric transformation applied to each
          input tensor. If ``False`` and the input is a tuple the applied transformation won't be concatenated.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.
    """

    def inverse_transform(
        self,
        input: torch.Tensor,
        transform: Optional[torch.Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """By default, the exact transformation as ``apply_transform`` will be used."""
        raise NotImplementedError

    def compute_inverse_transformation(self, transform: torch.Tensor):
        """Compute the inverse transform of given transformation matrices."""
        return _torch_inverse_cast(transform)

    def get_transformation_matrix(
        self, input: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        if params is not None:
            transform = self.compute_transformation(input[params['batch_prob']], params)
        elif not hasattr(self, "_transform_matrix"):
            params = self.forward_parameters(input.shape)
            transform = self.identity_matrix(input)
            transform[params['batch_prob']] = self.compute_transformation(input[params['batch_prob']], params)
        else:
            transform = self._transform_matrix
        return torch.as_tensor(transform, device=input.device, dtype=input.dtype)

    def inverse(
        self,
        input: TensorWithTransformMat,
        params: Optional[Dict[str, torch.Tensor]] = None,
        size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(input, (list, tuple)):
            input, transform = input
        else:
            transform = self.get_transformation_matrix(input, params)
        if params is not None:
            transform = self.identity_matrix(input)
            transform[params['batch_prob']] = self.compute_transformation(input[params['batch_prob']], params)

        ori_shape = input.shape
        in_tensor = self.transform_tensor(input)
        batch_shape = input.shape
        if params is None:
            params = self._params
        if size is None and "forward_input_shape" in params:
            # Majorly for cropping functions
            size = params['forward_input_shape'].numpy().tolist()
            size = (size[-2], size[-1])
        if 'batch_prob' not in params:
            params['batch_prob'] = torch.tensor([True] * batch_shape[0])
            warnings.warn("`batch_prob` is not found in params. Will assume applying on all data.")
        output = input.clone()
        to_apply = params['batch_prob']
        # if no augmentation needed
        if torch.sum(to_apply) == 0:
            output = in_tensor
        # if all data needs to be augmented
        elif torch.sum(to_apply) == len(to_apply):
            transform = self.compute_inverse_transformation(transform)
            output = self.inverse_transform(in_tensor, transform, size, **kwargs)
        else:
            transform[to_apply] = self.compute_inverse_transformation(transform[to_apply])
            output[to_apply] = self.inverse_transform(in_tensor[to_apply], transform[to_apply], size, **kwargs)
        return cast(torch.Tensor, _transform_output_shape(output, ori_shape)) if self.keepdim else output
