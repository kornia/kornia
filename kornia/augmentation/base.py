from typing import Tuple, Union, Optional, Dict, cast
import warnings

import torch
import torch.nn as nn

from torch.distributions import Bernoulli

from . import functional as F
from .utils import (
    _transform_input,
    _transform_input3d,
    _transform_output_shape,
    _validate_input_dtype,
    _adapted_sampling
)


class _BasicAugmentationBase(nn.Module):
    r"""_BasicAugmentationBase base class for customized augmentation implementations.

    Plain augmentation base class without the functionality of transformation matrix calculations.
    By default, the random computations will be happened on CPU with ``torch.get_default_dtype()``.
    To change this behaviour, please use ``set_rng_device_and_dtype``.

    Args:
        p (float): probability for applying an augmentation. This param controls the augmentation
                   probabilities element-wisely.
        p_batch (float): probability for applying an augmentation to a batch. This param controls the augmentation
                         probabilities batch-wisely.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.
    """

    def __init__(self, p: float = 0.5, p_batch: float = 1., same_on_batch: bool = False,
                 keepdim: bool = False) -> None:
        super(_BasicAugmentationBase, self).__init__()
        self.p = p
        self.p_batch = p_batch
        self.same_on_batch = same_on_batch
        self.keepdim = keepdim
        self._params = None  # type: Union[None,Dict[str, torch.Tensor]]
        if p != 0. or p != 1.:
            self._p_gen = Bernoulli(self.p)
        if p_batch != 0. or p_batch != 1.:
            self._p_batch_gen = Bernoulli(self.p_batch)
        self.set_rng_device_and_dtype(torch.device('cpu'), torch.get_default_dtype())

    def __repr__(self) -> str:
        return f"p={self.p}, p_batch={self.p_batch}, same_on_batch={self.same_on_batch}"

    def __unpack_input__(
        self, input: torch.Tensor
    ) -> torch.Tensor:
        return input

    def __check_batching__(self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
        """Check if a transformation matrix is returned,
        it has to be in the same batching mode as output."""
        raise NotImplementedError

    def transform_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """Standardize input tensors."""
        raise NotImplementedError

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def set_rng_device_and_dtype(self, device: torch.device, dtype: torch.dtype) -> None:
        """Change the random generation device and dtype.

        Note:
            The generated random numbers are not reproducible across different devices and dtypes.
        """
        self.device = device
        self.dtype = dtype

    def __selective_param_gen__(
            self, batch_shape: torch.Size, to_apply: torch.Tensor) -> Dict[str, torch.Tensor]:
        _params = self.generate_parameters(
            torch.Size((int(to_apply.sum().item()), *batch_shape[1:])))
        if _params is None:
            _params = {}
        _params['batch_prob'] = to_apply
        return _params

    def __forward_parameters__(
            self, batch_shape: torch.Size, p: float, p_batch: float, same_on_batch: bool) -> Dict[str, torch.Tensor]:
        batch_prob: torch.Tensor
        if p_batch == 1:
            batch_prob = torch.tensor([True])
        elif p_batch == 0:
            batch_prob = torch.tensor([False])
        else:
            batch_prob = _adapted_sampling((1,), self._p_batch_gen, same_on_batch).bool()

        if batch_prob.sum().item() == 1:
            elem_prob: torch.Tensor
            if p == 1:
                elem_prob = torch.tensor([True] * batch_shape[0])
            elif p == 0:
                elem_prob = torch.tensor([False] * batch_shape[0])
            else:
                elem_prob = _adapted_sampling((batch_shape[0],), self._p_gen, same_on_batch).bool()
            batch_prob = batch_prob * elem_prob
        else:
            batch_prob = batch_prob.repeat(batch_shape[0])
        # selectively param gen
        return self.__selective_param_gen__(batch_shape, batch_prob)

    def apply_func(self, input: torch.Tensor, params: Dict[str, torch.Tensor],
                   ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        input = self.transform_tensor(input)
        return self.apply_transform(input, params)

    def forward(self, input: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None,  # type: ignore
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore
        in_tensor = self.__unpack_input__(input)
        self.__check_batching__(input)
        ori_shape = in_tensor.shape
        in_tensor = self.transform_tensor(in_tensor)
        batch_shape = in_tensor.shape
        if params is None:
            params = self.__forward_parameters__(batch_shape, self.p, self.p_batch, self.same_on_batch)
        self._params = params

        output = self.apply_func(input, self._params)
        return _transform_output_shape(output, ori_shape) if self.keepdim else output


class _AugmentationBase(_BasicAugmentationBase):
    r"""_AugmentationBase base class for customized augmentation implementations.

    Advanced augmentation base class with the functionality of transformation matrix calculations.

    Args:
        p (float): probability for applying an augmentation. This param controls the augmentation probabilities
                   element-wisely for a batch.
        p_batch (float): probability for applying an augmentation to a batch. This param controls the augmentation
                         probabilities batch-wisely.
        return_transform (bool): if ``True`` return the matrix describing the geometric transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.
    """

    def __init__(self, return_transform: bool = False, same_on_batch: bool = False, p: float = 0.5,
                 p_batch: float = 1., keepdim: bool = False) -> None:
        super(_AugmentationBase, self).__init__(p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        self.p = p
        self.p_batch = p_batch
        self.return_transform = return_transform

    def __repr__(self) -> str:
        return super().__repr__() + f", return_transform={self.return_transform}"

    def identity_matrix(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def __unpack_input__(  # type: ignore
        self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(input, tuple):
            in_tensor = input[0]
            in_transformation = input[1]
            return in_tensor, in_transformation
        else:
            in_tensor = input
            return in_tensor, None

    def apply_func(self, in_tensor: torch.Tensor, in_transform: Optional[torch.Tensor],  # type: ignore
                   params: Dict[str, torch.Tensor], return_transform: bool = False
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
            try:
                output[to_apply] = self.apply_transform(in_tensor[to_apply], params)
            except Exception as e:
                raise ValueError(f"{e}, {to_apply}")
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
                params: Optional[Dict[str, torch.Tensor]] = None,  # type: ignore
                return_transform: Optional[bool] = None,
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore
        in_tensor, in_transform = self.__unpack_input__(input)
        self.__check_batching__(input)
        ori_shape = in_tensor.shape
        in_tensor = self.transform_tensor(in_tensor)
        batch_shape = in_tensor.shape

        if return_transform is None:
            return_transform = self.return_transform
        if params is None:
            params = self.__forward_parameters__(batch_shape, self.p, self.p_batch, self.same_on_batch)
        if 'batch_prob' not in params:
            params['batch_prob'] = torch.tensor([True] * batch_shape[0])
            warnings.warn("`batch_prob` is not found in params. Will assume applying on all data.")

        self._params = params
        output = self.apply_func(in_tensor, in_transform, self._params, return_transform)
        return _transform_output_shape(output, ori_shape) if self.keepdim else output


class AugmentationBase2D(_AugmentationBase):
    r"""AugmentationBase2D base class for customized augmentation implementations.

    For any augmentation, the implementation of "generate_parameters" and "apply_transform" are required while the
    "compute_transformation" is only required when passing "return_transform" as True.

    Args:
        p (float): probability for applying an augmentation. This param controls the augmentation probabilities
                   element-wisely for a batch.
        p_batch (float): probability for applying an augmentation to a batch. This param controls the augmentation
                         probabilities batch-wisely.
        return_transform (bool): if ``True`` return the matrix describing the geometric transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.
    """

    def __check_batching__(self, input: Union[torch.Tensor,
                                              Tuple[torch.Tensor, torch.Tensor]]):
        if isinstance(input, tuple):
            inp, mat = input
            if len(inp.shape) == 4:
                assert len(mat.shape) == 3, 'Input tensor is in batch mode ' \
                                            'but transformation matrix is not'
                assert mat.shape[0] == inp.shape[0], f'In batch dimension, input has {inp.shape[0]}' \
                                                     f'but transformation matrix has {mat.shape[0]}'
            elif len(inp.shape) == 3 or len(inp.shape) == 2:
                assert len(mat.shape) == 2, 'Input tensor is in non-batch mode ' \
                                            'but transformation matrix is not'
            else:
                raise ValueError(f'Unrecognized output shape. Expected 2, 3, or 4, got {len(inp.shape)}')

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
        p_batch (float): probability for applying an augmentation to a batch. This param controls the augmentation
                         probabilities batch-wisely.
        return_transform (bool): if ``True`` return the matrix describing the geometric transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
    """

    def __check_batching__(self, input: Union[torch.Tensor,
                                              Tuple[torch.Tensor, torch.Tensor]]):
        if isinstance(input, tuple):
            inp, mat = input
            if len(inp.shape) == 5:
                assert len(mat.shape) == 3, 'Input tensor is in batch mode ' \
                                            'but transformation matrix is not'
                assert mat.shape[0] == inp.shape[0], f'In batch dimension, input has {inp.shape[0]}' \
                                                     f'but transformation matrix has {mat.shape[0]}'
            elif len(inp.shape) == 3 or len(inp.shape) == 4:
                assert len(mat.shape) == 2, 'Input tensor is in non-batch mode ' \
                                            'but transformation matrix is not'
            else:
                raise ValueError(f'Unrecognized output shape. Expected 3, 4 or 5, got {len(inp.shape)}')

    def transform_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """Convert any incoming (D, H, W), (C, D, H, W) and (B, C, D, H, W) into (B, C, D, H, W)."""
        _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])
        return _transform_input3d(input)

    def identity_matrix(self, input) -> torch.Tensor:
        """Return 4x4 identity matrix."""
        return F.compute_intensity_transformation3d(input)


class MixAugmentationBase(_BasicAugmentationBase):
    r"""MixAugmentationBase base class for customized mix augmentation implementations.

    For any augmentation, the implementation of "generate_parameters" and "apply_transform" are required.
    "apply_transform" will need to handle the probabilities internally.

    Args:
        p (float): probability for applying an augmentation.
            This param controls if to apply the augmentation for the batch.
        p_batch (float): probability for applying an augmentation to a batch. This param controls the augmentation
                         probabilities batch-wisely.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.
    """

    def __init__(self, p: float, p_batch: float, same_on_batch: bool = False, keepdim: bool = False) -> None:
        super(MixAugmentationBase, self).__init__(p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)

    def __check_batching__(self, input: Union[torch.Tensor,
                                              Tuple[torch.Tensor, torch.Tensor]]):
        if isinstance(input, tuple):
            inp, mat = input
            if len(inp.shape) == 4:
                assert len(mat.shape) == 3, 'Input tensor is in batch mode ' \
                                            'but transformation matrix is not'
                assert mat.shape[0] == inp.shape[0], f'In batch dimension, input has {inp.shape[0]}' \
                                                     f'but transformation matrix has {mat.shape[0]}'
            elif len(inp.shape) == 3 or len(inp.shape) == 2:
                assert len(mat.shape) == 2, 'Input tensor is in non-batch mode ' \
                                            'but transformation matrix is not'
            else:
                raise ValueError(f'Unrecognized output shape. Expected 2, 3, or 4, got {len(inp.shape)}')

    def transform_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """Convert any incoming (H, W), (C, H, W) and (B, C, H, W) into (B, C, H, W)."""
        _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])
        return _transform_input(input)

    def apply_transform(self, input: torch.Tensor, label: torch.Tensor,     # type: ignore
                        params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:   # type: ignore
        raise NotImplementedError

    def apply_func(self, in_tensor: torch.Tensor, label: torch.Tensor,  # type: ignore
                   params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        to_apply = params['batch_prob']

        # if no augmentation needed
        if torch.sum(to_apply) == 0:
            output = in_tensor
        # if all data needs to be augmented
        elif torch.sum(to_apply) == len(to_apply):
            output, label = self.apply_transform(in_tensor, label, params)
        else:
            raise ValueError(
                "Mix augmentations must be performed batch-wisely. Element-wise augmentation is not supported.")

        return output, label

    def forward(self, input: torch.Tensor, label: torch.Tensor,  # type: ignore
                params: Optional[Dict[str, torch.Tensor]] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        in_tensor = self.__unpack_input__(input)
        ori_shape = in_tensor.shape
        in_tensor = self.transform_tensor(in_tensor)
        if params is None:
            batch_shape = in_tensor.shape
            params = self.__forward_parameters__(batch_shape, self.p, self.p_batch, self.same_on_batch)
        self._params = params

        output = self.apply_func(in_tensor, label, self._params)
        return _transform_output_shape(output, ori_shape) if self.keepdim else output  # type: ignore
