from typing import Tuple, Union, Optional

import torch
import torch.nn as nn

from . import functional as F
from . import random_generator as rg
from .random_generator import AugParamDict
from .utils import (
    _infer_batch_shape,
    _infer_batch_shape3d,
    _transform_input,
    _transform_input3d,
    _validate_input_dtype,
)


class AugmentationBase(nn.Module):
    r"""AugmentationBase base class for customized augmentation implementations. For any augmentation,
    the implementation of "generate_parameters" and "apply_transform" are required while the
    "compute_transformation" is only required when passing "return_transform" as True.

    Args:
        p (float): probability for applying an augmentation. This param controls the augmentation probabilities
                   element-wisely for a batch.
        return_transform (bool): if ``True`` return the matrix describing the geometric transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False
    """

    def __init__(self, p: float, return_transform: bool = False, same_on_batch: bool = False) -> None:
        super(AugmentationBase, self).__init__()
        self.p = p
        self.return_transform = return_transform
        self.same_on_batch = same_on_batch

    def __repr__(self) -> str:
        return f"p={self.p}, return_transform={self.return_transform}, same_on_batch={self.same_on_batch}"

    def infer_batch_shape(self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Size:
        raise NotImplementedError

    def transform_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """Standardize input tensors"""
        raise NotImplementedError

    def identity_matrix(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def generate_parameters(self, batch_shape: torch.Size) -> AugParamDict:
        raise NotImplementedError

    def compute_transformation(self, input: torch.Tensor, params: AugParamDict) -> torch.Tensor:
        raise NotImplementedError

    def apply_transform(self, input: torch.Tensor, params: AugParamDict) -> torch.Tensor:
        raise NotImplementedError

    def __forward_parameters__(self, batch_shape: torch.Size) -> AugParamDict:
        _params = self.generate_parameters(batch_shape)
        _params.update({'flags': _params['flags'] if 'flags' in _params else {}})
        _params.update({'params': _params['params'] if 'params' in _params else {}})
        batch_prob = rg.random_prob_generator(batch_shape[0], self.p, self.same_on_batch)
        # select the params to be augmented
        for k, v in _params['params'].items():
            _params['params'].update({k: v[batch_prob]})
        return AugParamDict(dict(batch_prob=batch_prob, params=_params['params'], flags=_params['flags']))

    def __forward_transformation_matrix__(self, input: torch.Tensor, params: AugParamDict):
        to_apply = params['batch_prob']
        identity_matrix = self.identity_matrix(input)
        identity_matrix[to_apply] = self.compute_transformation(input[to_apply], params)
        return identity_matrix

    def __forward_input__(self, input: torch.Tensor, params: AugParamDict) -> torch.Tensor:
        """ Forward the transformation.

        Args:
            params (Dict[str, torch.Tensor]): A dict with the following keys.
                - params['batch_prob']: element-wise boolean mask to determin if to apply the augmentation or not.
                - params['params']: element-wise param for each element in a batch.
                - params['flags']: overall config flags for the augmentation
        """
        to_apply = params['batch_prob']

        to_pass_params = params.copy()

        if len(to_apply.unique()) == 1 and not to_apply.unique()[0]:
            # if no data needs to be transformed
            return input
        elif len(to_apply.unique()) == 1 and to_apply.unique()[0]:
            # if all data needs to be transformed
            return self.apply_transform(input, to_pass_params)
        else:
            # select the inputs to be augmented
            transformed[to_apply] = self.apply_transform(input[to_apply], params['params'], params['flags'])
        return transformed

    def forward_transform(self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                          params: Optional[AugParamDict] = None,  # type: ignore
                          return_transform: Optional[bool] = None
                          ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore
        if isinstance(input, tuple):
            in_tensor = self.transform_tensor(input[0])
            in_transformation = input[1]
        else:
            in_tensor = self.transform_tensor(input)
            in_transformation = None

        output = self.__forward_input__(in_tensor, self._params)
        if return_transform:
            transformation_matrix = self.__forward_transformation_matrix__(in_tensor, self._params)
            out_t = transformation_matrix if in_transformation is None else transformation_matrix @ in_transformation
            return output, out_t

        if in_transformation is not None:
            return output, in_transformation
        return output

    def forward(self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                params: Optional[AugParamDict] = None,  # type: ignore
                return_transform: Optional[bool] = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore
        if return_transform is None:
            return_transform = self.return_transform
        if params is None:
            batch_shape = self.infer_batch_shape(input)
            params = self.__forward_parameters__(batch_shape)
        self._params = params

        return self.forward_transform(input, self._params, return_transform)


class AugmentationBase2D(AugmentationBase):

    def infer_batch_shape(self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Size:
        return _infer_batch_shape(input)

    def transform_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """Convert any incoming (D, H, W), (C, D, H, W) and (B, C, D, H, W) into (B, C, D, H, W)"""
        _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])
        return _transform_input(input)

    def identity_matrix(self, input) -> torch.Tensor:
        return F.compute_intensity_transformation(input)


class AugmentationBase3D(AugmentationBase):

    def infer_batch_shape(self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Size:
        return _infer_batch_shape3d(input)

    def transform_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """Convert any incoming (D, H, W), (C, D, H, W) and (B, C, D, H, W) into (B, C, D, H, W)"""
        _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])
        return _transform_input3d(input)

    def identity_matrix(self, input) -> torch.Tensor:
        return F.compute_intensity_transformation3d(input)
