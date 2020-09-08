from typing import Tuple, Union, Optional, cast
import logging

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


class _BasicAugmentationBase(nn.Module):
    r"""_BasicAugmentationBase base class for customized augmentation implementations.

    Plain augmentation base class without the functionality of transformation matrix calculations.

    Args:
        p (float): probability for applying an augmentation.
        p_mode ('batch' or 'element'): control the behaviour of probablities generation. Default: 'element'.
            If 'batch', param p will control if to augment the whole batch or not.
            If 'element', param p will control if to augment a batch element-wisely.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
    """

    def __init__(self, p: float = 0.5, same_on_batch: bool = False, p_mode: str = 'element') -> None:
        super(_BasicAugmentationBase, self).__init__()
        self.p = p
        self.same_on_batch = same_on_batch
        if p_mode not in ['batch', 'element']:
            raise ValueError(f"`p_mode` must be either `batch` or `element`. Got {p_mode}.")
        self.p_mode = p_mode

    def __repr__(self) -> str:
        return f"p={self.p}, same_on_batch={self.same_on_batch}"

    def __infer_input__(
        self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        in_tensor = self.transform_tensor(input)
        return in_tensor

    def transform_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """Standardize input tensors."""
        raise NotImplementedError

    def generate_parameters(self, batch_shape: torch.Size) -> AugParamDict:
        raise NotImplementedError

    def apply_transform(self, input: torch.Tensor, params: AugParamDict) -> torch.Tensor:
        raise NotImplementedError

    def __get_param_dict__(self, batch_shape: torch.Size, to_apply: Optional[torch.Tensor] = None) -> AugParamDict:
        _params = self.generate_parameters(torch.Size((torch.sum(to_apply), *batch_shape[1:])))
        if _params is None:
            _params = {"params": {}, "flags": {}}
        assert 'batch_prob' not in _params, f"Unexpected `batch_prob` found in params. Got {_params}."
        _params.update({'flags': _params['flags'] if 'flags' in _params else {}})
        _params.update({'params': _params['params'] if 'params' in _params else {}})
        return AugParamDict(dict(batch_prob=to_apply, params=_params['params'], flags=_params['flags']))

    def __forward_parameters__(self, batch_shape: torch.Size, p: float, same_on_batch: bool, p_mode: str) -> AugParamDict:
        if p == 1:
            batch_prob = torch.tensor([True]).repeat(batch_shape[0])
        elif p == 0:
            batch_prob = torch.tensor([False]).repeat(batch_shape[0])
        else:
            if p_mode == 'element':
                batch_prob = rg.random_prob_generator(batch_shape[0], p, same_on_batch)
            elif p_mode == 'batch':
                batch_prob = rg.random_prob_generator(1, p, same_on_batch).repeat(batch_shape[0])
            else:
                raise ValueError(f"`p_mode` must be either `batch` or `element`. Got {p_mode}.")
        # selectively param gen
        return self.__get_param_dict__(batch_shape, batch_prob)

    def apply_func(self, input: torch.Tensor, params: AugParamDict,
                   ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        input = self.transform_tensor(input)
        return self.apply_transform(input, params)

    def forward(self, input: torch.Tensor, params: Optional[AugParamDict] = None,  # type: ignore
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore
        if params is None:
            batch_shape = self.infer_batch_shape(input)
            params = self.__forward_parameters__(batch_shape, self.p, self.same_on_batch, self.p_mode)
        self._params = params

        return self.apply_func(input, self._params)


class _AugmentationBase(_BasicAugmentationBase):
    r"""_AugmentationBase base class for customized augmentation implementations.

    Advanced augmentation base class with the functionality of transformation matrix calculations.

    Args:
        p (float): probability for applying an augmentation. This param controls the augmentation probabilities
                   element-wisely for a batch.
        p_mode ('batch' or 'element'): control the behaviour of probablities generation. Default: 'element'.
            If 'batch', param p will control if to augment the whole batch or not.
            If 'element', param p will control if to augment a batch element-wisely.
        return_transform (bool): if ``True`` return the matrix describing the geometric transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
    """

    def __init__(self, p: float, return_transform: bool = False, same_on_batch: bool = False,
                 p_mode: str = 'element') -> None:
        super(_AugmentationBase, self).__init__(p, same_on_batch=same_on_batch, p_mode=p_mode)
        self.p = p
        self.return_transform = return_transform

    def __repr__(self) -> str:
        return super().__repr__() + f", return_transform={self.return_transform}"

    def identity_matrix(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_transformation(self, input: torch.Tensor, params: AugParamDict) -> torch.Tensor:
        raise NotImplementedError

    def __infer_input__(
        self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(input, tuple):
            in_tensor = self.transform_tensor(input[0])
            in_transformation = input[1]
            return (in_tensor, in_transformation)
        else:
            in_tensor = self.transform_tensor(input)
            return in_tensor, None

    def apply_func(self, in_tensor: torch.Tensor, in_transform: Optional[torch.Tensor],
                   params: AugParamDict, return_transform: bool = False
                   ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        to_apply = params['batch_prob']

        # if no augmentation needed
        if torch.sum(to_apply) == 0:
            output = in_tensor
            if return_transform:
                trans_matrix = self.identity_matrix(in_tensor)
        # if all data needs to be augmented
        elif torch.sum(to_apply) == len(to_apply):
            output = self.apply_transform(in_tensor, params)
            if return_transform:
                trans_matrix = self.compute_transformation(in_tensor, params)
        else:
            output = in_tensor.clone()
            output[to_apply] = self.apply_transform(in_tensor[to_apply], params)
            if return_transform:
                trans_matrix = self.identity_matrix(in_tensor)
                trans_matrix[to_apply] = self.compute_transformation(in_tensor[to_apply], params)

        if return_transform:
            out_transformation = trans_matrix if in_transform is None else trans_matrix @ in_transform
            return output, out_transformation

        if in_transform is not None:
            return output, in_transform

        return output

    def forward(self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                params: Optional[AugParamDict] = None,  # type: ignore
                return_transform: Optional[bool] = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore
        in_tensor, in_transform = self.__infer_input__(input)
        batch_shape = in_tensor.shape
        if return_transform is None:
            return_transform = self.return_transform
        if params is None:
            params = self.__forward_parameters__(batch_shape, self.p, self.same_on_batch, self.p_mode)
        if 'batch_prob' not in params:
            params['batch_prob'] = torch.tensor([True] * batch_shape[0])
            logging.warning(f"`batch_prob` is not found in params. Will assume applying on all data.")

        self._params = params
        return self.apply_func(in_tensor, in_transform, self._params, return_transform)


class AugmentationBase2D(_AugmentationBase):
    r"""AugmentationBase2D base class for customized augmentation implementations.

    For any augmentation, the implementation of "generate_parameters" and "apply_transform" are required while the
    "compute_transformation" is only required when passing "return_transform" as True.

    Args:
        p (float): probability for applying an augmentation. This param controls the augmentation probabilities
                   element-wisely for a batch.
        p_mode ('batch' or 'element'): control the behaviour of probablities generation. Default: 'element'.
            If 'batch', param p will control if to augment the whole batch or not.
            If 'element', param p will control if to augment a batch element-wisely.
        return_transform (bool): if ``True`` return the matrix describing the geometric transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
    """

    def transform_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """Convert any incoming (H, W), (C, H, W) and (B, C, H, W) into (B, C, H, W)."""
        _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])
        return _transform_input(input)

    def identity_matrix(self, input) -> torch.Tensor:
        """Return 3x3 identity matrix."""
        return F.compute_intensity_transformation(input)


class AugmentationBase3D(_AugmentationBase):
    r"""AugmentationBase3D base class for customized augmentation implementations.

    For any augmentation, the implementation of "generate_parameters" and "apply_transform" are required while the
    "compute_transformation" is only required when passing "return_transform" as True.

    Args:
        p (float): probability for applying an augmentation. This param controls the augmentation probabilities
                   element-wisely for a batch.
        return_transform (bool): if ``True`` return the matrix describing the geometric transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False
    """

    def transform_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """Convert any incoming (D, H, W), (C, D, H, W) and (B, C, D, H, W) into (B, C, D, H, W)."""
        _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])
        return _transform_input3d(input)

    def identity_matrix(self, input) -> torch.Tensor:
        """Return 4x4 identity matrix."""
        return F.compute_intensity_transformation3d(input)


class MixAugmentationBase(_BasicAugmentationBase):
    r"""MixAugmentationBase base class for customized augmentation implementations.

    For any augmentation, the implementation of "generate_parameters" and "apply_transform" are required.
    "apply_transform" will need to handle the probabilities internally.

    Args:
        p (float): probability for applying an augmentation. This param controls the augmentation probabilities
                   element-wisely for a batch.
        same_on_batch (bool): apply the same transformation across the batch. Default: False
    """

    def __init__(self, p: float, same_on_batch: bool = False) -> None:
        # TODO: Have another p for batch mode?
        super(MixAugmentationBase, self).__init__(p, same_on_batch=same_on_batch, p_mode='element')

    def transform_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """Convert any incoming (H, W), (C, H, W) and (B, C, H, W) into (B, C, H, W)."""
        _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])
        return _transform_input(input)

    def apply_transform(self, input: torch.Tensor, label: torch.Tensor,     # type: ignore
                        params: AugParamDict) -> Tuple[torch.Tensor, torch.Tensor]:   # type: ignore
        raise NotImplementedError

    def forward(self, input: torch.Tensor, label: torch.Tensor,  # type: ignore
                params: Optional[AugParamDict] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        in_tensor = self.__infer_input__(input)
        if params is None:
            batch_shape = in_tensor.shape
            params = self.__forward_parameters__(batch_shape, 1., self.same_on_batch, self.p_mode)
        self._params = params

        return self.apply_transform(in_tensor, label, self._params)
