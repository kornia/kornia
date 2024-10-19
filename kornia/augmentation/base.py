from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch.distributions import Bernoulli, Distribution, RelaxedBernoulli

from kornia.augmentation.random_generator import RandomGeneratorBase
from kornia.augmentation.utils import (
    _adapted_rsampling,
    _adapted_sampling,
    _transform_output_shape,
    override_parameters,
)
from kornia.core import ImageModule as Module
from kornia.core import Tensor, tensor, zeros
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints
from kornia.utils import is_autocast_enabled

TensorWithTransformMat = Union[Tensor, Tuple[Tensor, Tensor]]


# Trick mypy into not applying contravariance rules to inputs by defining
# forward as a value, rather than a function.  See also
# https://github.com/python/mypy/issues/8795
# Based on the trick that torch.nn.Module does for the forward method
def _apply_transform_unimplemented(self: Module, *input: Any) -> Tensor:
    r"""Defines the computation performed at every call.

    Should be overridden by all subclasses.
    """
    raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required "apply_tranform" function')


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

    # TODO: Hard to support. Many codes are not ONNX-friendly that contains lots of if-else blocks, etc.
    # Please contribute if anyone interested.
    ONNX_EXPORTABLE = False

    def __init__(
        self,
        p: float = 0.5,
        p_batch: float = 1.0,
        same_on_batch: bool = False,
        keepdim: bool = False,
    ) -> None:
        super().__init__()
        self.p = p
        self.p_batch = p_batch
        self.same_on_batch = same_on_batch
        self.keepdim = keepdim
        self._params: Dict[str, Tensor] = {}
        self._p_gen: Distribution
        self._p_batch_gen: Distribution
        if p != 0.0 or p != 1.0:
            self._p_gen = Bernoulli(self.p)
        if p_batch != 0.0 or p_batch != 1.0:
            self._p_batch_gen = Bernoulli(self.p_batch)
        self._param_generator: Optional[RandomGeneratorBase] = None
        self.flags: Dict[str, Any] = {}
        self.set_rng_device_and_dtype(torch.device("cpu"), torch.get_default_dtype())

    apply_transform: Callable[..., Tensor] = _apply_transform_unimplemented

    def to(self, *args: Any, **kwargs: Any) -> "_BasicAugmentationBase":
        r"""Set the device and dtype for the random number generator."""
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.set_rng_device_and_dtype(device, dtype)
        return super().to(*args, **kwargs)

    def __repr__(self) -> str:
        txt = f"p={self.p}, p_batch={self.p_batch}, same_on_batch={self.same_on_batch}"
        if isinstance(self._param_generator, RandomGeneratorBase):
            txt = f"{self._param_generator!s}, {txt}"
        for k, v in self.flags.items():
            if isinstance(v, Enum):
                txt += f", {k}={v.name.lower()}"
            else:
                txt += f", {k}={v}"
        return f"{self.__class__.__name__}({txt})"

    def __unpack_input__(self, input: Tensor) -> Tensor:
        return input

    def transform_tensor(
        self,
        input: Tensor,
        *,
        shape: Optional[Tensor] = None,
        match_channel: bool = True,
    ) -> Tensor:
        """Standardize input tensors."""
        raise NotImplementedError

    def validate_tensor(self, input: Tensor) -> None:
        """Check if the input tensor is formatted as expected."""
        raise NotImplementedError

    def transform_output_tensor(self, output: Tensor, output_shape: Tuple[int, ...]) -> Tensor:
        """Standardize output tensors."""
        return _transform_output_shape(output, output_shape) if self.keepdim else output

    def generate_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, Tensor]:
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
        self,
        batch_shape: Tuple[int, ...],
        p: float,
        p_batch: float,
        same_on_batch: bool,
    ) -> Tensor:
        batch_prob: Tensor
        if p_batch == 1:
            batch_prob = zeros(1) + 1
        elif p_batch == 0:
            batch_prob = zeros(1)
        elif isinstance(self._p_batch_gen, (RelaxedBernoulli,)):
            # NOTE: there is no simple way to know if the sampler has `rsample` or not
            batch_prob = _adapted_rsampling((1,), self._p_batch_gen, same_on_batch)
        else:
            batch_prob = _adapted_sampling((1,), self._p_batch_gen, same_on_batch)

        if batch_prob.sum() == 1:
            elem_prob: Tensor
            if p == 1:
                elem_prob = zeros(batch_shape[0]) + 1
            elif p == 0:
                elem_prob = zeros(batch_shape[0])
            elif isinstance(self._p_gen, (RelaxedBernoulli,)):
                elem_prob = _adapted_rsampling((batch_shape[0],), self._p_gen, same_on_batch)
            else:
                elem_prob = _adapted_sampling((batch_shape[0],), self._p_gen, same_on_batch)
            batch_prob = batch_prob * elem_prob
        else:
            batch_prob = batch_prob.repeat(batch_shape[0])
        if len(batch_prob.shape) == 2:
            return batch_prob[..., 0]
        return batch_prob

    def _process_kwargs_to_params_and_flags(
        self,
        params: Optional[Dict[str, Tensor]] = None,
        flags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
        # NOTE: determine how to save self._params
        save_kwargs = kwargs["save_kwargs"] if "save_kwargs" in kwargs else False

        params = self._params if params is None else params
        flags = self.flags if flags is None else flags

        if save_kwargs:
            params = override_parameters(params, kwargs, in_place=True)
            self._params = params
        else:
            self._params = params
            params = override_parameters(params, kwargs, in_place=False)

        flags = override_parameters(flags, kwargs, in_place=False)
        return params, flags

    def forward_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, Tensor]:
        batch_prob = self.__batch_prob_generator__(batch_shape, self.p, self.p_batch, self.same_on_batch)
        to_apply = batch_prob > 0.5
        _params = self.generate_parameters(torch.Size((int(to_apply.sum().item()), *batch_shape[1:])))
        if _params is None:
            _params = {}
        _params["batch_prob"] = batch_prob
        # Added another input_size parameter for geometric transformations
        # This might be needed for correctly inversing.
        input_size = tensor(batch_shape, dtype=torch.long)
        _params.update({"forward_input_shape": input_size})
        return _params

    def apply_func(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        return self.apply_transform(input, params, flags)

    def forward(self, input: Tensor, params: Optional[Dict[str, Tensor]] = None, **kwargs: Any) -> Tensor:
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

        if "batch_prob" not in params:
            params["batch_prob"] = tensor([True] * batch_shape[0])

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

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        # apply transform for the input image tensor
        raise NotImplementedError

    def apply_non_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        # apply additional transform for the images that are skipped from transformation
        # where batch_prob == False.
        return input

    def transform_inputs(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        params, flags = self._process_kwargs_to_params_and_flags(
            self._params if params is None else params, flags, **kwargs
        )

        batch_prob = params["batch_prob"]
        to_apply = batch_prob > 0.5  # NOTE: in case of Relaxed Distributions.
        ori_shape = input.shape
        in_tensor = self.transform_tensor(input)

        self.validate_tensor(in_tensor)
        if to_apply.all():
            output = self.apply_transform(in_tensor, params, flags, transform=transform)
        elif not to_apply.any():
            output = self.apply_non_transform(in_tensor, params, flags, transform=transform)
        else:  # If any tensor needs to be transformed.
            output = self.apply_non_transform(in_tensor, params, flags, transform=transform)
            applied = self.apply_transform(
                in_tensor[to_apply],
                params,
                flags,
                transform=transform if transform is None else transform[to_apply],
            )

            if is_autocast_enabled():
                output = output.type(input.dtype)
                applied = applied.type(input.dtype)
            output = output.index_put((to_apply,), applied)

        output = _transform_output_shape(output, ori_shape) if self.keepdim else output

        if is_autocast_enabled():
            output = output.type(input.dtype)
        return output

    def transform_masks(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        params, flags = self._process_kwargs_to_params_and_flags(
            self._params if params is None else params, flags, **kwargs
        )

        batch_prob = params["batch_prob"]
        to_apply = batch_prob > 0.5  # NOTE: in case of Relaxed Distributions.
        ori_shape = input.shape

        shape = params["forward_input_shape"]
        in_tensor = self.transform_tensor(input, shape=shape, match_channel=False)

        self.validate_tensor(in_tensor)
        if to_apply.all():
            output = self.apply_transform_mask(in_tensor, params, flags, transform=transform)
        elif not to_apply.any():
            output = self.apply_non_transform_mask(in_tensor, params, flags, transform=transform)
        else:  # If any tensor needs to be transformed.
            output = self.apply_non_transform_mask(in_tensor, params, flags, transform=transform)
            applied = self.apply_transform_mask(
                in_tensor[to_apply],
                params,
                flags,
                transform=transform if transform is None else transform[to_apply],
            )
            output = output.index_put((to_apply,), applied)
        output = _transform_output_shape(output, ori_shape, reference_shape=shape) if self.keepdim else output
        return output

    def transform_boxes(
        self,
        input: Boxes,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Boxes:
        if not isinstance(input, Boxes):
            raise RuntimeError(f"Only `Boxes` is supported. Got {type(input)}.")

        params, flags = self._process_kwargs_to_params_and_flags(
            self._params if params is None else params, flags, **kwargs
        )

        batch_prob = params["batch_prob"]
        to_apply = batch_prob > 0.5  # NOTE: in case of Relaxed Distributions.
        output: Boxes
        if to_apply.bool().all():
            output = self.apply_transform_box(input, params, flags, transform=transform)
        elif not to_apply.any():
            output = self.apply_non_transform_box(input, params, flags, transform=transform)
        else:  # If any tensor needs to be transformed.
            output = self.apply_non_transform_box(input, params, flags, transform=transform)
            applied = self.apply_transform_box(
                input[to_apply],
                params,
                flags,
                transform=transform if transform is None else transform[to_apply],
            )
            if is_autocast_enabled():
                output = output.type(input.dtype)
                applied = applied.type(input.dtype)

            output = output.index_put((to_apply,), applied)
        return output

    def transform_keypoints(
        self,
        input: Keypoints,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Keypoints:
        if not isinstance(input, Keypoints):
            raise RuntimeError(f"Only `Keypoints` is supported. Got {type(input)}.")

        params, flags = self._process_kwargs_to_params_and_flags(
            self._params if params is None else params, flags, **kwargs
        )

        batch_prob = params["batch_prob"]
        to_apply = batch_prob > 0.5  # NOTE: in case of Relaxed Distributions.
        if to_apply.all():
            output = self.apply_transform_keypoint(input, params, flags, transform=transform)
        elif not to_apply.any():
            output = self.apply_non_transform_keypoint(input, params, flags, transform=transform)
        else:  # If any tensor needs to be transformed.
            output = self.apply_non_transform_keypoint(input, params, flags, transform=transform)
            applied = self.apply_transform_keypoint(
                input[to_apply],
                params,
                flags,
                transform=transform if transform is None else transform[to_apply],
            )
            if is_autocast_enabled():
                output = output.type(input.dtype)
                applied = applied.type(input.dtype)
            output = output.index_put((to_apply,), applied)
        return output

    def transform_classes(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        params, flags = self._process_kwargs_to_params_and_flags(
            self._params if params is None else params, flags, **kwargs
        )

        batch_prob = params["batch_prob"]
        to_apply = batch_prob > 0.5  # NOTE: in case of Relaxed Distributions.
        if to_apply.all():
            output = self.apply_transform_class(input, params, flags, transform=transform)
        elif not to_apply.any():
            output = self.apply_non_transform_class(input, params, flags, transform=transform)
        else:  # If any tensor needs to be transformed.
            output = self.apply_non_transform_class(input, params, flags, transform=transform)
            applied = self.apply_transform_class(
                input[to_apply],
                params,
                flags,
                transform=transform if transform is None else transform[to_apply],
            )
            output = output.index_put((to_apply,), applied)
        return output

    def apply_non_transform_mask(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        """Process masks corresponding to the inputs that are no transformation applied."""
        raise NotImplementedError

    def apply_transform_mask(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        """Process masks corresponding to the inputs that are transformed."""
        raise NotImplementedError

    def apply_non_transform_box(
        self,
        input: Boxes,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Boxes:
        """Process boxes corresponding to the inputs that are no transformation applied."""
        return input

    def apply_transform_box(
        self,
        input: Boxes,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Boxes:
        """Process boxes corresponding to the inputs that are transformed."""
        raise NotImplementedError

    def apply_non_transform_keypoint(
        self,
        input: Keypoints,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Keypoints:
        """Process keypoints corresponding to the inputs that are no transformation applied."""
        return input

    def apply_transform_keypoint(
        self,
        input: Keypoints,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Keypoints:
        """Process keypoints corresponding to the inputs that are transformed."""
        raise NotImplementedError

    def apply_non_transform_class(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        """Process class tags corresponding to the inputs that are no transformation applied."""
        return input

    def apply_transform_class(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        """Process class tags corresponding to the inputs that are transformed."""
        raise NotImplementedError

    def apply_func(
        self,
        in_tensor: Tensor,
        params: Dict[str, Tensor],
        flags: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        if flags is None:
            flags = self.flags

        output = self.transform_inputs(in_tensor, params, flags)

        return output
