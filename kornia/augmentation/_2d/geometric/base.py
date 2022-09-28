from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple, cast

from torch import Tensor, as_tensor

from kornia.augmentation._2d.base import AugmentationBase2D
from kornia.augmentation.utils import override_parameters
from kornia.utils.helpers import _torch_inverse_cast


class GeometricAugmentationBase2D(AugmentationBase2D):
    r"""GeometricAugmentationBase2D base class for customized geometric augmentation implementations.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.
    """

    def inverse_transform(
        self, input: Tensor, flags: dict[str, Any], transform: Tensor | None = None, size: tuple[int, int] | None = None
    ) -> Tensor:
        """By default, the exact transformation as ``apply_transform`` will be used."""
        raise NotImplementedError

    def compute_inverse_transformation(self, transform: Tensor):
        """Compute the inverse transform of given transformation matrices."""
        return _torch_inverse_cast(transform)

    def get_transformation_matrix(
        self, input: Tensor, params: dict[str, Tensor] | None = None, flags: dict[str, Any] | None = None
    ) -> Tensor:
        flags = self.flags if flags is None else flags
        if params is not None:
            transform = self.compute_transformation(input[params['batch_prob']], params=params, flags=flags)

        elif self.transform_matrix is None:
            params = self.forward_parameters(input.shape)
            transform = self.identity_matrix(input)
            transform[params['batch_prob']] = self.compute_transformation(
                input[params['batch_prob']], params=params, flags=flags
            )
        else:
            transform = self.transform_matrix
        return as_tensor(transform, device=input.device, dtype=input.dtype)

    def inverse(
        self, input: Tensor, params: dict[str, Tensor] | None = None, size: tuple[int, int] | None = None, **kwargs
    ) -> Tensor:
        """Perform inverse operations.

        Args:
            input: the input tensor.
            params: the corresponding parameters for an operation.
                If None, a new parameter suite will be generated.
            size: input size during the forward step to restore the original shape.
            **kwargs: key-value pairs to override the parameters and flags.
        """
        input_shape = input.shape
        in_tensor = self.transform_tensor(input)
        batch_shape = input.shape

        if len(kwargs.keys()) != 0:
            _src_params = self._params if params is None else params
            params = override_parameters(_src_params, kwargs, in_place=False)
            flags = override_parameters(self.flags, kwargs, in_place=False)
        else:
            flags = self.flags

        if params is not None:
            transform = self.identity_matrix(in_tensor)
            transform[params['batch_prob']] = self.compute_transformation(
                in_tensor[params['batch_prob']], params=params, flags=flags
            )
        else:
            # Avoid recompute.
            transform = self.get_transformation_matrix(in_tensor, params=params, flags=flags)
            params = self._params

        if size is None and "forward_input_shape" in params:
            # Majorly for cropping functions
            size = params['forward_input_shape'].numpy().tolist()
            size = (size[-2], size[-1])
        if 'batch_prob' not in params:
            params['batch_prob'] = as_tensor([True] * batch_shape[0])
            warnings.warn("`batch_prob` is not found in params. Will assume applying on all data.")
        output = in_tensor.clone()
        to_apply = params['batch_prob']
        # if no augmentation needed
        if not to_apply.any():
            output = in_tensor
        # if all data needs to be augmented
        elif to_apply.all():
            transform = self.compute_inverse_transformation(transform)
            output = self.inverse_transform(in_tensor, flags=flags, transform=transform, size=size)
        else:
            transform[to_apply] = self.compute_inverse_transformation(transform[to_apply])
            output[to_apply] = self.inverse_transform(
                in_tensor[to_apply], transform=transform[to_apply], size=size, flags=flags
            )
        return cast(Tensor, self.transform_output_tensor(output, input_shape)) if self.keepdim else output
