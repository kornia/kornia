from typing import Tuple, Union, Optional, cast

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


class BasicAugmentationBase(nn.Module):

    def __init__(self, p: float, same_on_batch: bool = False) -> None:
        super(BasicAugmentationBase, self).__init__()
        self.p = p
        self.same_on_batch = same_on_batch

    def __repr__(self) -> str:
        return f"p={self.p}, same_on_batch={self.same_on_batch}"

    def infer_batch_shape(self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Size:
        raise NotImplementedError

    def transform_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """Standardize input tensors"""
        raise NotImplementedError

    def generate_parameters(self, batch_shape: torch.Size) -> AugParamDict:
        raise NotImplementedError

    def apply_transform(self, input: torch.Tensor, params: AugParamDict) -> torch.Tensor:
        raise NotImplementedError

    def __forward_parameters__(self, batch_shape: torch.Size) -> AugParamDict:
        _params = self.generate_parameters(batch_shape)
        if _params is None:
            _params = {"batch_prob": None, "params": {}, "flags": {}}
        _params.update({'flags': _params['flags'] if 'flags' in _params else {}})
        _params.update({'params': _params['params'] if 'params' in _params else {}})
        _params.update({'batch_prob': _params['batch_prob'] if 'batch_prob' in _params else None})
        return AugParamDict(dict(batch_prob=_params['batch_prob'], params=_params['params'], flags=_params['flags']))

    def forward(self, input: torch.Tensor, params: Optional[AugParamDict] = None,  # type: ignore
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore
        if params is None:
            batch_shape = self.infer_batch_shape(input)
            params = self.__forward_parameters__(batch_shape)
        self._params = params

        input = self.transform_tensor(input)

        return self.apply_transform(input, self._params)


class AugmentationBase(BasicAugmentationBase):
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
        super(AugmentationBase, self).__init__(p=p, same_on_batch=same_on_batch)
        self.return_transform = return_transform

    def __repr__(self) -> str:
        return super().__repr__() + f", return_transform={self.return_transform}"

    def identity_matrix(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_transformation(self, input: torch.Tensor, params: AugParamDict) -> torch.Tensor:
        raise NotImplementedError

    def __forward_input__(
        self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(input, tuple):
            in_tensor = self.transform_tensor(input[0])
            in_transformation = input[1]
            return (in_tensor, in_transformation)
        else:
            in_tensor = self.transform_tensor(input)
            return in_tensor

    def __forward_parameters__(self, batch_shape: torch.Size) -> AugParamDict:
        _params = super().__forward_parameters__(batch_shape)
        if _params["batch_prob"] is None:
            batch_prob = rg.random_prob_generator(batch_shape[0], self.p, self.same_on_batch)
        else:
            batch_prob = _params['batch_prob']
        return AugParamDict(dict(batch_prob=batch_prob, params=_params['params'], flags=_params['flags']))

    def ___forward_transform___(self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                                params: AugParamDict = None, return_transform: bool = None# type: ignore
                                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore
        in_transformation: Optional[torch.Tensor]
        if isinstance(input, tuple):
            in_tensor = input[0]
            in_transformation = input[1]
        else:
            in_tensor = input[0]
            in_transformation = None

        output = self.apply_transform(in_tensor, self._params)
        if return_transform:
            transformation_matrix = self.compute_transformation(in_tensor, self._params)
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

        input = self.__forward_input__(input)

        return self.___forward_transform___(input, self._params, return_transform)


class AugmentationBase2D(AugmentationBase):

    def infer_batch_shape(self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Size:
        return _infer_batch_shape(input)

    def transform_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """Convert any incoming (D, H, W), (C, D, H, W) and (B, C, D, H, W) into (B, C, D, H, W)"""
        _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])
        return _transform_input(input)

    def identity_matrix(self, input) -> torch.Tensor:
        return F.compute_intensity_transformation(input)

    def forward(self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                params: Optional[AugParamDict] = None,  # type: ignore
                return_transform: Optional[bool] = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore
        if return_transform is None:
            return_transform = self.return_transform
        if params is None:
            batch_shape = self.infer_batch_shape(input)
            params = self.__forward_parameters__(batch_shape)
        params = cast(AugParamDict, params)
        self._params = params

        to_apply = self._params['batch_prob']

        in_target = self.__forward_input__(input)

        # Situations that do not need to use to_apply
        # Smart selective forward
        selective_params: AugParamDict = dict(params)  # type: ignore
        for k, v in selective_params['params'].items():
            selective_params['params'].update({k: v[to_apply]})

        if not isinstance(in_target, tuple):
            in_target = (in_target, self.identity_matrix(in_target))

        # No need to go through the process if no data needs to apply augmentation
        if to_apply.unique() == torch.tensor([False]):
            partial_out = in_target
        else:
            partial_out = self.___forward_transform___(  # type: ignore
                (in_target[0][to_apply], in_target[1][to_apply]), selective_params, return_transform)

        if len(to_apply.unique()) == 1:
            # if True, avoid ragged tensor if methods like cropping is used
            # if False, avoid zero-dim in batch dimension
            out_tensor = partial_out[0]
        else:
            out_tensor = in_target[0].clone()
            out_tensor[to_apply] = partial_out[0]

        if isinstance(input, tuple) or return_transform:
            if len(to_apply.unique()) == 1:
                # if True, avoid ragged tensor if methods like cropping is used
                # if False, avoid zero-dim in batch dimension
                out_transformation = partial_out[1]
            else:
                out_transformation = in_target[1].clone()
                out_transformation[to_apply] = partial_out[1]
            return (out_tensor, out_transformation)

        return out_tensor


class AugmentationBase3D(AugmentationBase2D):

    def infer_batch_shape(self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Size:
        return _infer_batch_shape3d(input)

    def transform_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """Convert any incoming (D, H, W), (C, D, H, W) and (B, C, D, H, W) into (B, C, D, H, W)"""
        _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])
        return _transform_input3d(input)

    def identity_matrix(self, input) -> torch.Tensor:
        return F.compute_intensity_transformation3d(input)


class MixAugmentationBase(BasicAugmentationBase):
    r"""MixAugmentationBase base class for customized augmentation implementations. For any augmentation,
    the implementation of "generate_parameters" and "apply_transform" are required while the
    "compute_transformation" is only required when passing "return_transform" as True.

    In "apply_transform", both input and label tensors are required.

    Args:
        return_transform (bool): if ``True`` return the matrix describing the geometric transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.

    """

    def infer_batch_shape(self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Size:
        return _infer_batch_shape(input)

    def transform_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """Convert any incoming (D, H, W), (C, D, H, W) and (B, C, D, H, W) into (B, C, D, H, W)"""
        _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])
        return _transform_input(input)

    def apply_transform(self, input: torch.Tensor, label: torch.Tensor,     # type: ignore
                        params: AugParamDict) -> Tuple[torch.Tensor, torch.Tensor]:   # type: ignore
        raise NotImplementedError

    def forward(self, input: torch.Tensor, label: torch.Tensor,  # type: ignore
                params: Optional[AugParamDict] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        if params is None:
            batch_shape = self.infer_batch_shape(input)
            params = self.__forward_parameters__(batch_shape)
        self._params = params

        input = self.transform_tensor(input)

        return self.apply_transform(input, label, self._params)
