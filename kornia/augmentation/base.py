from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch.distributions import Bernoulli

from kornia.augmentation.random_generator import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_sampling, _transform_output_shape, override_parameters
from kornia.core import Module, Tensor, tensor
from kornia.geometry.boxes import Boxes

TensorWithTransformMat = Union[Tensor, Tuple[Tensor, Tensor]]


# Trick mypy into not applying contravariance rules to inputs by defining
# forward as a value, rather than a function.  See also
# https://github.com/python/mypy/issues/8795
# Based on the trick that torch.nn.Module does for the forward method
def _apply_transform_unimplemented(self, *input: Any) -> Tensor:
    r"""Defines the computation performed at every call.

    Should be overridden by all subclasses.
    """
    raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"apply_tranform\" function")


class _BasicAugmentationBase(Module):
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

    apply_transform: Callable[..., Tensor] = _apply_transform_unimplemented

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

    def transform_tensor(self, input: Tensor) -> Tensor:
        """Standardize input tensors."""
        raise NotImplementedError

    def transform_output_tensor(self, output: Tensor, output_shape: Tuple[int, ...]) -> Tensor:
        """Standardize output tensors."""
        return _transform_output_shape(output, output_shape) if self.keepdim else output

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, Tensor]:
        if self._param_generator is not None:
            return self._param_generator(batch_shape, self.same_on_batch)
        return {}

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
            batch_prob = tensor([True])
        elif p_batch == 0:
            batch_prob = tensor([False])
        else:
            batch_prob = _adapted_sampling((1,), self._p_batch_gen, same_on_batch).bool()

        if batch_prob.sum().item() == 1:
            elem_prob: Tensor
            if p == 1:
                elem_prob = tensor([True] * batch_shape[0])
            elif p == 0:
                elem_prob = tensor([False] * batch_shape[0])
            else:
                elem_prob = _adapted_sampling((batch_shape[0],), self._p_gen, same_on_batch).bool()
            batch_prob = batch_prob * elem_prob
        else:
            batch_prob = batch_prob.repeat(batch_shape[0])
        return batch_prob

    def _process_kwargs_to_params_and_flags(
        self, params: Dict[str, Tensor], flags: Dict[str, Any], **kwargs
    ) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:

        # NOTE: determine how to save self._params
        save_kwargs = kwargs["save_kwargs"] if "save_kwargs" in kwargs else False

        if save_kwargs:
            params = override_parameters(params, kwargs, in_place=True)
            self._params = params
        else:
            self._params = params
            params = override_parameters(params, kwargs, in_place=False)

        flags = override_parameters(self.flags, kwargs, in_place=False)
        return params, flags

    def forward_parameters(self, batch_shape) -> Dict[str, Tensor]:
        to_apply = self.__batch_prob_generator__(batch_shape, self.p, self.p_batch, self.same_on_batch)
        _params = self.generate_parameters(torch.Size((int(to_apply.sum().item()), *batch_shape[1:])))
        if _params is None:
            _params = {}
        _params['batch_prob'] = to_apply
        # Added another input_size parameter for geometric transformations
        # This might be needed for correctly inversing.
        input_size = tensor(batch_shape, dtype=torch.long)
        _params.update({'forward_input_shape': input_size})
        return _params

    def apply_func(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        return self.apply_transform(input, params, flags)

    def forward(self, input: Tensor, params: Optional[Dict[str, Tensor]] = None, **kwargs) -> Tensor:
        """Perform forward operations.

        Args:
            input: the input tensor.
            params: the corresponding parameters for an operation.
                If None, a new parameter suite will be generated.
            **kwargs: key-value pairs to override the parameters and flags.

        Note:
            By default, all the overwriting parameters in kwargs will not be recorded
            as in ``self._params``. If you wish it to be recorded, you may pass
            ``save_kwargs=True`` additionally.
        """
        in_tensor = self.__unpack_input__(input)
        input_shape = in_tensor.shape
        in_tensor = self.transform_tensor(in_tensor)
        batch_shape = in_tensor.shape
        if params is None:
            params = self.forward_parameters(batch_shape)

        if 'batch_prob' not in params:
            params['batch_prob'] = tensor([True] * batch_shape[0])

        params, flags = self._process_kwargs_to_params_and_flags(params, self.flags, **kwargs)

        output = self.apply_func(in_tensor, params, flags)
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

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({super().__repr__()})"

    def identity_matrix(self, input: Tensor) -> Tensor:
        raise NotImplementedError

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        # NOTE: apply_transform receives the whole tensor, but returns only altered elements.
        raise NotImplementedError

    def apply_non_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        # For the images where batch_prob == False.
        return input

    def transform_input(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        to_apply = params['batch_prob']
        ori_shape = input.shape
        in_tensor = self.transform_tensor(input)
        output = in_tensor
        if sum(to_apply) != len(to_apply):
            output = self.apply_non_transform(in_tensor, params, flags)
        if sum(to_apply) != 0:
            applied = self.apply_transform(in_tensor, params, flags)
            output = self.apply_non_transform(in_tensor, params, flags)
            output = output.index_put((to_apply,), self.apply_non_transform(applied, params, flags))
        output = _transform_output_shape(output, ori_shape) if self.keepdim else output
        return output

    def transform_mask(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        to_apply = params['batch_prob']
        output = input
        if sum(to_apply) != len(to_apply):
            output = self.apply_non_transform_mask(input, params, flags)
        if sum(to_apply) != 0:
            output = self.apply_transform_mask(input, params, flags)
        return output

    def transform_boxes(
        self,
        input: Union[Tensor, Boxes],
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Boxes:
        # input is BxNx4x2 or Boxes.
        if isinstance(input, Tensor):
            if not (len(input.shape) == 4 and input.shape[2:] == torch.Size([4, 2])):
                raise RuntimeError(f"Only BxNx4x2 tensor is supported. Got {input.shape}.")
            input = Boxes(input, False, mode="vertices_plus")
        to_apply = params['batch_prob']
        output = input
        if sum(to_apply) != len(to_apply):
            output = self.apply_non_transform_boxes(input, params, flags)
        if sum(to_apply) != 0:
            output = self.apply_transform_boxes(output, params, flags)
        return output

    def transform_keypoint(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        to_apply = params['batch_prob']
        output = input
        if sum(to_apply) != len(to_apply):
            output = self.apply_non_transform_keypoint(input, params, flags)
        if sum(to_apply) != 0:
            output = self.apply_transform_keypoint(input, params, flags)
        return output

    def transform_class(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        to_apply = params['batch_prob']
        output = input
        if sum(to_apply) != len(to_apply):
            output = self.apply_non_transform_class(input, params, flags)
        if sum(to_apply) != 0:
            output = self.apply_transform_class(input, params, flags)
        return output

    def apply_non_transform_mask(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        raise NotImplementedError

    def apply_transform_mask(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        raise NotImplementedError

    def apply_non_transform_boxes(
        self, input: Boxes, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Boxes:
        return input

    def apply_transform_boxes(
        self, input: Boxes, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Boxes:
        raise NotImplementedError

    def apply_non_transform_keypoint(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return input

    def apply_transform_keypoint(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        raise NotImplementedError

    def apply_non_transform_class(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return input

    def apply_transform_class(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        raise NotImplementedError

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        raise NotImplementedError

    def apply_func(
        self, in_tensor: Tensor, params: Dict[str, Tensor], flags: Optional[Dict[str, Any]] = None
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
            trans_matrix = trans_matrix.index_put(
                (to_apply,), self.compute_transformation(in_tensor[to_apply], params=params, flags=flags)
            )
            output = output.index_put(
                (to_apply,),
                self.apply_transform(in_tensor[to_apply], params=params, flags=flags, transform=trans_matrix[to_apply]),
            )

        self._transform_matrix = trans_matrix

        return output
