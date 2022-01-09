from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
from torch.distributions import Bernoulli

from kornia.augmentation.random_generator import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_sampling, _transform_output_shape

TensorWithTransformMat = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


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
        self._params: Dict[str, torch.Tensor] = {}
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

    def __unpack_input__(self, input: torch.Tensor) -> torch.Tensor:
        return input

    def __check_batching__(self, input: TensorWithTransformMat):
        """Check if a transformation matrix is returned, it has to be in the same batching mode as output."""
        raise NotImplementedError

    def transform_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """Standardize input tensors."""
        raise NotImplementedError

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        if self._param_generator is not None:
            return self._param_generator(batch_shape, self.same_on_batch)
        return {}

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
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
    ) -> torch.Tensor:
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
        return batch_prob

    def forward_parameters(self, batch_shape) -> Dict[str, torch.Tensor]:
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

    def apply_func(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> TensorWithTransformMat:
        input = self.transform_tensor(input)
        return self.apply_transform(input, params)

    def forward(  # type: ignore
        self, input: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None  # type: ignore
    ) -> TensorWithTransformMat:
        in_tensor = self.__unpack_input__(input)
        self.__check_batching__(input)
        ori_shape = in_tensor.shape
        in_tensor = self.transform_tensor(in_tensor)
        batch_shape = in_tensor.shape
        if params is None:
            params = self.forward_parameters(batch_shape)
        self._params = params

        output = self.apply_func(input, self._params)
        return _transform_output_shape(output, ori_shape) if self.keepdim else output


class _AugmentationBase(_BasicAugmentationBase):
    r"""_AugmentationBase base class for customized augmentation implementations.

    Advanced augmentation base class with the functionality of transformation matrix calculations.

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

    def __init__(
        self,
        return_transform: bool = None,
        same_on_batch: bool = False,
        p: float = 0.5,
        p_batch: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        self.p = p
        self.p_batch = p_batch
        self.return_transform = return_transform

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({super().__repr__()}, return_transform={self.return_transform})"

    def identity_matrix(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def __unpack_input__(  # type: ignore
        self, input: TensorWithTransformMat
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(input, tuple):
            in_tensor = input[0]
            in_transformation = input[1]
            return in_tensor, in_transformation
        in_tensor = input
        return in_tensor, None

    def apply_func(  # type: ignore
        self,
        in_tensor: torch.Tensor,
        in_transform: Optional[torch.Tensor],  # type: ignore
        params: Dict[str, torch.Tensor],
        return_transform: bool = False,
    ) -> TensorWithTransformMat:
        to_apply = params['batch_prob']

        # if no augmentation needed
        if torch.sum(to_apply) == 0:
            output = in_tensor
            trans_matrix = self.identity_matrix(in_tensor)
        # if all data needs to be augmented
        elif torch.sum(to_apply) == len(to_apply):
            trans_matrix = self.compute_transformation(in_tensor, params)
            output = self.apply_transform(in_tensor, params, trans_matrix)
        else:
            output = in_tensor.clone()
            trans_matrix = self.identity_matrix(in_tensor)
            trans_matrix[to_apply] = self.compute_transformation(in_tensor[to_apply], params)
            output[to_apply] = self.apply_transform(in_tensor[to_apply], params, trans_matrix[to_apply])

        self._transform_matrix = trans_matrix

        if return_transform:
            out_transformation = trans_matrix if in_transform is None else trans_matrix @ in_transform
            return output, out_transformation

        if in_transform is not None:
            return output, in_transform

        return output

    def forward(  # type: ignore
        self,
        input: TensorWithTransformMat,
        params: Optional[Dict[str, torch.Tensor]] = None,  # type: ignore
        return_transform: Optional[bool] = None,
    ) -> TensorWithTransformMat:
        in_tensor, in_transform = self.__unpack_input__(input)
        self.__check_batching__(input)
        ori_shape = in_tensor.shape
        in_tensor = self.transform_tensor(in_tensor)
        batch_shape = in_tensor.shape

        if return_transform is None:
            return_transform = self.return_transform
        return_transform = cast(bool, return_transform)
        if params is None:
            params = self.forward_parameters(batch_shape)
        if 'batch_prob' not in params:
            params['batch_prob'] = torch.tensor([True] * batch_shape[0])
            # TODO(jian): we cannot throw a warning every time.
            # warnings.warn("`batch_prob` is not found in params. Will assume applying on all data.")

        self._params = params
        output = self.apply_func(in_tensor, in_transform, self._params, return_transform)
        return _transform_output_shape(output, ori_shape) if self.keepdim else output
