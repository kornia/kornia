from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Bernoulli

from kornia.augmentation.random_generator import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_sampling, _transform_output_shape

TensorWithTransformMat = Union[Tensor, Tuple[Tensor, Tensor]]


class _BasicAugmentationBase(nn.Module):
    r"""_BasicAugmentationBase base class for customized augmentation implementations.

    Plain augmentation base class without the functionality of transformation matrix calculations.
    By default, the random computations will be happened on CPU with ``torch.get_default_dtype()``.
    To change this behaviour, please use ``set_rng_device_and_dtype``.

    For automatically generating the corresponding ``__repr__`` with full customized parameters, you may need to
    implement ``_param_generator`` by inheriting ``RandomGeneratorBase`` for generating random parameters and
    put all static parameters inside ``self.flags``. You may take the advantage of ``PlainUniformGenerator`` to
    generate simple uniform parameters with less boilerplate code.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities element-wise.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it to
          the batch form ``False``.
    """

    def __init__(
        self, p: float = 0.5, p_batch: float = 1.0, same_on_batch: bool = False, keepdim: bool = False
    ) -> None:
        super().__init__()
        self.p = p
        self.p_batch = p_batch
        self.same_on_batch = same_on_batch
        self.keepdim = keepdim
        self._params: Dict[str, Tensor] = {}
        if p != 0.0 or p != 1.0:
            self._p_gen = Bernoulli(self.p)
        if p_batch != 0.0 or p_batch != 1.0:
            self._p_batch_gen = Bernoulli(self.p_batch)
        self._param_generator: Optional[RandomGeneratorBase] = None
        self.flags: Dict[str, Any] = {}
        self.set_rng_device_and_dtype(torch.device('cpu'), torch.get_default_dtype())

    def __repr__(self) -> str:
        txt = f"p={self.p}, p_batch={self.p_batch}, same_on_batch={self.same_on_batch}"
        if isinstance(self._param_generator, RandomGeneratorBase):
            txt = f"{str(self._param_generator)}, {txt}"
        for k, v in self.flags.items():
            if isinstance(v, Enum):
                txt += f", {k}={v.name.lower()}"
            else:
                txt += f", {k}={v}"
        return f"{self.__class__.__name__}({txt})"

    def __unpack_input__(self, input: Tensor) -> Tensor:
        return input

    def __check_batching__(self, input: Tensor):
        """Check if a transformation matrix is returned, it has to be in the same batching mode as output."""
        raise NotImplementedError

    def transform_tensor(self, input: Tensor) -> Tensor:
        """Standardize input tensors."""
        raise NotImplementedError

    def transform_output_tensor(self, output: Tensor, output_shape: Tuple) -> Tensor:
        """Standardize output tensors."""
        return _transform_output_shape(output, output_shape) if self.keepdim else output

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, Tensor]:
        if self._param_generator is not None:
            return self._param_generator(batch_shape, self.same_on_batch)
        return {}

    def apply_transform(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        raise NotImplementedError

    def set_rng_device_and_dtype(self, device: torch.device, dtype: torch.dtype) -> None:
        """Change the random generation device and dtype.

        Note:
            The generated random numbers are not reproducible across different devices and dtypes.
        """
        self.device = device
        self.dtype = dtype
        if self._param_generator is not None:
            self._param_generator.set_rng_device_and_dtype(device, dtype)

    def __batch_prob_generator__(
        self, batch_shape: torch.Size, p: float, p_batch: float, same_on_batch: bool
    ) -> Tensor:
        batch_prob: Tensor
        if p_batch == 1:
            batch_prob = torch.tensor([True])
        elif p_batch == 0:
            batch_prob = torch.tensor([False])
        else:
            batch_prob = _adapted_sampling((1,), self._p_batch_gen, same_on_batch).bool()

        if batch_prob.sum().item() == 1:
            elem_prob: Tensor
            if p == 1:
                elem_prob = torch.tensor([True] * batch_shape[0])
            elif p == 0:
                elem_prob = torch.tensor([False] * batch_shape[0])
            else:
                elem_prob = _adapted_sampling((batch_shape[0],), self._p_gen, same_on_batch).bool()
            batch_prob = batch_prob * elem_prob
        else:
            batch_prob = batch_prob.repeat(batch_shape[0])
        return batch_prob

    def forward_parameters(self, batch_shape) -> Dict[str, Tensor]:
        to_apply = self.__batch_prob_generator__(batch_shape, self.p, self.p_batch, self.same_on_batch)
        _params = self.generate_parameters(torch.Size((int(to_apply.sum().item()), *batch_shape[1:])))
        if _params is None:
            _params = {}
        _params['batch_prob'] = to_apply
        # Added another input_size parameter for geometric transformations
        # This might be needed for correctly inversing.
        input_size = torch.tensor(batch_shape, dtype=torch.long)
        _params.update({'forward_input_shape': input_size})
        return _params

    def apply_func(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        return self.apply_transform(input, params, flags)

    def _deepcopy_param(self, param: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in param.items():
            # NOTE: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol
            if isinstance(v, Tensor):
                out.update({k: v.clone()})
            else:
                out.update({k: v})
        return param

    def _override_parameters(
        self, params: Dict[str, Any], params_override: Optional[Dict[str, Any]] = None,
        if_none_exist: str = 'ignore', in_place: bool = False
    ) -> Dict[str, Any]:
        """Override params dict w.r.t params_override.

        Args:
            params: source parameters.
            params_override: key-values to override the source parameters.
            if_none_exist: behaviour if the key in `params_override` does not exist in `params`.
                'raise' | 'ignore'.
            in_place: if to override in-place or not.
        """

        if params_override is None:
            return params
        out = params if in_place else self._deepcopy_param(params)
        for k, v in params_override.items():
            if k in params_override:
                out[k] = v
            else:
                if if_none_exist == 'ignore':
                    pass
                elif if_none_exist == 'raise':
                    raise RuntimeError(f"Param `{k}` not existed in `{params_override}`.")
                else:
                    raise ValueError(f"`{if_none_exist}` is not a valid option.")
        return out

    def forward(  # type: ignore
        self, input: Tensor, params: Optional[Dict[str, Tensor]] = None, **kwargs
    ) -> Tensor:
        """Perform forward operations.

        Args:
            input: the input tensor.
            params: the corresponding parameters for an operation.
                If None, a new parameter suite will be generated.
            **kwargs: key-value pairs to override the parameters and flags.
        """
        in_tensor = self.__unpack_input__(input)
        self.__check_batching__(input)
        input_shape = in_tensor.shape
        in_tensor = self.transform_tensor(in_tensor)
        batch_shape = in_tensor.shape
        if params is None:
            params = self.forward_parameters(batch_shape)
            params = self._override_parameters(params, kwargs, in_place=True)
        else:
            params = self._override_parameters(params, kwargs, in_place=False)

        if 'batch_prob' not in params:
            params['batch_prob'] = torch.tensor([True] * batch_shape[0])

        self._params = params
        _flags = self._override_parameters(self.flags, kwargs, in_place=False)

        output = self.apply_func(in_tensor, self._params, _flags)
        return self.transform_output_tensor(output, input_shape) if self.keepdim else output


class _AugmentationBase(_BasicAugmentationBase):
    r"""_AugmentationBase base class for customized augmentation implementations.

    Advanced augmentation base class with the functionality of transformation matrix calculations.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.
    """

    def __init__(
        self,
        return_transform: Optional[bool] = None,
        same_on_batch: bool = False,
        p: float = 0.5,
        p_batch: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        self.p = p
        self.p_batch = p_batch
        self.return_transform = return_transform
        self._transform_matrix: Tensor
        if return_transform is not None:
            raise ValueError(
                "`return_transform` is deprecated. Please access the transformation matrix with "
                "`.transform_matrix`. For chained matrices, please use `AugmentationSequential`.",
            )

    @property
    def transform_matrix(self,) -> Tensor:
        return self._transform_matrix

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({super().__repr__()})"

    def identity_matrix(self, input: Tensor) -> Tensor:
        raise NotImplementedError

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        raise NotImplementedError

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        raise NotImplementedError

    def apply_func(  # type: ignore
        self,
        in_tensor: Tensor,
        params: Dict[str, Tensor],
        flags: Optional[Dict[str, Any]] = None
    ) -> Tensor:
        if flags is None:
            flags = self.flags
        to_apply = params['batch_prob']

        # if no augmentation needed
        if not to_apply.any():
            output = in_tensor
            trans_matrix = self.identity_matrix(in_tensor)
        # if all data needs to be augmented
        elif to_apply.all():
            trans_matrix = self.compute_transformation(in_tensor, params=params, flags=flags)
            output = self.apply_transform(in_tensor, params=params, flags=flags, transform=trans_matrix)
        else:
            output = in_tensor.clone()
            trans_matrix = self.identity_matrix(in_tensor)
            trans_matrix[to_apply] = self.compute_transformation(
                in_tensor[to_apply], params=params, flags=flags)
            output[to_apply] = self.apply_transform(
                in_tensor[to_apply], params=params, flags=flags, transform=trans_matrix[to_apply])

        self._transform_matrix = trans_matrix

        return output
