# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.distributions import Bernoulli, Distribution, RelaxedBernoulli

from kornia.augmentation.random_generator import RandomGeneratorBase
from kornia.augmentation.utils import (
    _adapted_rsampling,
    _adapted_sampling,
    _transform_output_shape,
    override_parameters,
)
from kornia.core.utils import is_autocast_enabled
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints

TensorWithTransformMat = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


# Trick mypy into not applying contravariance rules to inputs by defining
# forward as a value, rather than a function.  See also
# https://github.com/python/mypy/issues/8795
# Based on the trick that torch.nn.Module does for the forward method
def _apply_transform_unimplemented(self: nn.Module, *input: Any) -> torch.Tensor:
    r"""Define the computation performed at every call.

    Should be overridden by all subclasses.
    """
    raise NotImplementedError(f'nn.Module [{type(self).__name__}] is missing the required "apply_tranform" function')


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

    # TODO: Hard to support. Many codes are not ONNX-friendly that contains lots of if-else blocks, etc.
    # Please contribute if anyone interested.
    ONNX_EXPORTABLE = False

    # Opt-in flag: subclasses that implement ``_fast_image_only_apply`` and that
    # are safe to bypass the full forward chain (parameter generation, transform
    # matrix construction, batch-prob branching, ``_params`` caching) for the
    # simple "image-only, no replay, no transform_matrix tracking" call shape
    # set this to ``True``.  See ``forward`` for the activation conditions.
    _supports_fast_image_only_path: bool = False

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
        self._params: Dict[str, torch.Tensor] = {}
        self._p_gen: Distribution
        self._p_batch_gen: Distribution
        if p not in {0.0, 1.0}:
            self._p_gen = Bernoulli(self.p)
        if p_batch not in {0.0, 1.0}:
            self._p_batch_gen = Bernoulli(self.p_batch)
        self._param_generator: Optional[RandomGeneratorBase] = None
        self.flags: Dict[str, Any] = {}
        self.set_rng_device_and_dtype(torch.device("cpu"), torch.get_default_dtype())
        # Construction-time flag: when both p and p_batch are 1.0, every element
        # in every batch is always selected.  We cache this here so that
        # forward_parameters and transform_inputs can avoid data-dependent
        # runtime branches (e.g. batch_prob.sum().item()) that prevent
        # torch.export from tracing the module.
        self._always_apply: bool = p == 1.0 and p_batch == 1.0

    apply_transform: Callable[..., torch.Tensor] = _apply_transform_unimplemented

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

    def __unpack_input__(self, input: torch.Tensor) -> torch.Tensor:
        return input

    def transform_tensor(
        self,
        input: torch.Tensor,
        *,
        shape: Optional[torch.Tensor] = None,
        match_channel: bool = True,
    ) -> torch.Tensor:
        """Standardize input tensors."""
        raise NotImplementedError

    def validate_tensor(self, input: torch.Tensor) -> None:
        """Check if the input torch.Tensor is formatted as expected."""
        raise NotImplementedError

    def transform_output_tensor(self, output: torch.Tensor, output_shape: Tuple[int, ...]) -> torch.Tensor:
        """Standardize output tensors."""
        return _transform_output_shape(output, output_shape) if self.keepdim else output

    def generate_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
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
    ) -> torch.Tensor:
        batch_prob: torch.Tensor
        if p_batch == 1:
            batch_prob = torch.zeros(1) + 1
        elif p_batch == 0:
            batch_prob = torch.zeros(1)
        elif isinstance(self._p_batch_gen, (RelaxedBernoulli,)):
            # NOTE: there is no simple way to know if the sampler has `rsample` or not
            batch_prob = _adapted_rsampling((1,), self._p_batch_gen, same_on_batch)
        else:
            batch_prob = _adapted_sampling((1,), self._p_batch_gen, same_on_batch)

        elem_prob: torch.Tensor
        if p == 1:
            elem_prob = torch.zeros(batch_shape[0]) + 1
        elif p == 0:
            elem_prob = torch.zeros(batch_shape[0])
        elif isinstance(self._p_gen, (RelaxedBernoulli,)):
            elem_prob = _adapted_rsampling((batch_shape[0],), self._p_gen, same_on_batch)
        else:
            elem_prob = _adapted_sampling((batch_shape[0],), self._p_gen, same_on_batch)
        batch_prob = batch_prob * elem_prob
        if len(batch_prob.shape) == 2:
            return batch_prob[..., 0]
        return batch_prob

    def _process_kwargs_to_params_and_flags(
        self,
        params: Optional[Dict[str, torch.Tensor]] = None,
        flags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        # NOTE: determine how to save self._params
        save_kwargs = kwargs["save_kwargs"] if "save_kwargs" in kwargs else False

        params = self._params if params is None else params
        flags = self.flags if flags is None else flags

        if save_kwargs:
            params = override_parameters(params, kwargs, in_place=True)
            # Do not mutate self._params while being traced by torch.export /
            # torch.compile — the exporter detects attribute changes and raises.
            if not torch.compiler.is_compiling():
                self._params = params
        else:
            # Same guard: skip the cache-write under export/compile.
            if not torch.compiler.is_compiling():
                self._params = params
            params = override_parameters(params, kwargs, in_place=False)

        flags = override_parameters(flags, kwargs, in_place=False)
        return params, flags

    def forward_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        batch_prob = self.__batch_prob_generator__(batch_shape, self.p, self.p_batch, self.same_on_batch)
        to_apply = batch_prob > 0.5
        # When all elements are always selected (p == p_batch == 1.0) the number
        # of elements to generate parameters for equals the full batch size, which
        # is a static integer known at construction time.  Avoid the data-dependent
        # `.sum().item()` call so that torch.export can trace without raising
        # GuardOnDataDependentSymNode.
        if self._always_apply:
            n_apply = batch_shape[0]
        else:
            n_apply = int(to_apply.sum().item())
        _params = self.generate_parameters(torch.Size((n_apply, *batch_shape[1:])))
        if _params is None:
            _params = {}
        _params["batch_prob"] = batch_prob
        # Added another input_size parameter for geometric transformations
        # This might be needed for correctly inversing.
        input_size = torch.tensor(batch_shape, dtype=torch.long)
        _params.update({"forward_input_shape": input_size})
        return _params

    def apply_func(self, input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, Any]) -> torch.Tensor:
        return self.apply_transform(input, params, flags)

    def _fast_image_only_apply(self, input: torch.Tensor) -> Optional[torch.Tensor]:
        """Image-only fast path implementation.

        Subclasses that opt into the fast path by setting
        ``_supports_fast_image_only_path = True`` MUST override this method.  The
        contract is: produce the same numerical output as the full forward chain
        for an "image-only, no replay, no transform_matrix tracking" call.  The
        per-sample probability ``self.p`` MUST be honoured here; the activation
        gate in ``forward`` only checks ``p_batch == 1.0``.

        Returning ``None`` signals the standard forward chain should be used
        instead (e.g. for runtime configurations the fast path does not cover).

        Args:
            input: the input torch.Tensor with shape (B, C, H, W) (or (C, H, W)
                / (H, W) — caller guarantees only that it is a tensor).
        """
        raise NotImplementedError(
            f"{type(self).__name__} sets _supports_fast_image_only_path=True but does not "
            "override _fast_image_only_apply."
        )

    @torch.no_grad()
    def forward(
        self, input: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None, **kwargs: Any
    ) -> torch.Tensor:
        """Perform forward operations.

        Args:
            input: the input torch.Tensor.
            params: the corresponding parameters for an operation.
                If None, a new parameter suite will be generated.
            **kwargs: key-value pairs to override the parameters and flags.

        Note:
            By default, all the overwriting parameters in kwargs will not be recorded
            as in ``self._params``. If you wish it to be recorded, you may pass
            ``save_kwargs=True`` additionally.

        """
        # Opt-in fast path: bypass parameter generation, transform-matrix
        # construction, batch-prob branching, and ``_params`` caching for the
        # simple "image-only" call shape.  The per-sample probability ``self.p``
        # is handled inside ``_fast_image_only_apply``; only ``p_batch == 1.0``
        # is gated here so the whole batch is always considered.
        if (
            self._supports_fast_image_only_path
            and params is None
            and not kwargs
            and isinstance(input, torch.Tensor)
            and self.p_batch == 1.0
            and not self.keepdim
        ):
            output = self._fast_image_only_apply(input)
            # ``_fast_image_only_apply`` may return ``None`` to signal that the
            # current configuration is not supported and the caller should fall
            # back to the standard forward chain.
            if output is not None:
                # Populate ``_params`` with the minimum viable suite so that
                # post-forward operations like ``aug.inverse(...)`` (which
                # consume ``params['batch_prob']`` and
                # ``params['forward_input_shape']``) keep working without
                # forcing the fast path to materialise the full param dict.
                # ``batch_prob`` reflects the per-sample apply mask; for the
                # deterministic fast path (``p`` in {0, 1}) this is uniform.
                if isinstance(input, torch.Tensor) and input.dim() >= 2:
                    in_shape: Tuple[int, ...] = (
                        tuple(input.shape) if input.dim() == 4 else (1,) * (4 - input.dim()) + tuple(input.shape)
                    )
                    fill_value = bool(self.p > 0.5)
                    base_params: Dict[str, torch.Tensor] = {
                        "batch_prob": torch.full((in_shape[0],), fill_value, dtype=torch.bool),
                        "forward_input_shape": torch.tensor(in_shape, dtype=torch.long),
                    }
                    # Per-class fast paths may stash additional generated
                    # parameters (e.g. random ``thresholds`` for Solarize) so
                    # ``aug(input, params=aug._params)`` replay reproduces the
                    # exact output via the standard chain.
                    extra = getattr(self, "_fast_path_extra_params", None)
                    if extra:
                        base_params.update(extra)
                        self._fast_path_extra_params = None
                    self._params = base_params
                else:
                    self._params = {}
                # ``_transform_matrix`` is the responsibility of the per-class
                # ``_fast_image_only_apply`` implementation: it may either set
                # the matrix explicitly (preferred when cheap) or leave the
                # previous value untouched.  See the per-class overrides.
                return output

        in_tensor = self.__unpack_input__(input)
        input_shape = in_tensor.shape
        in_tensor = self.transform_tensor(in_tensor)
        batch_shape = in_tensor.shape
        if params is None:
            params = self.forward_parameters(batch_shape)

        if "batch_prob" not in params:
            params["batch_prob"] = torch.tensor([True] * batch_shape[0])

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
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # apply transform for the input image torch.Tensor
        raise NotImplementedError

    def apply_non_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # apply additional transform for the images that are skipped from transformation
        # where batch_prob == False.
        return input

    def transform_inputs(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        params, flags = self._process_kwargs_to_params_and_flags(
            self._params if params is None else params, flags, **kwargs
        )

        batch_prob = params["batch_prob"]
        to_apply = batch_prob > 0.5  # NOTE: in case of Relaxed Distributions.
        ori_shape = input.shape
        in_tensor = self.transform_tensor(input)

        self.validate_tensor(in_tensor)
        # When _always_apply is True (p == p_batch == 1.0), every element is
        # always selected.  Skip the data-dependent .all()/.any() guards so that
        # torch.export can trace the module without raising
        # GuardOnDataDependentSymNode.
        if self._always_apply:
            output = self.apply_transform(in_tensor, params, flags, transform=transform)
        elif to_apply.all():
            output = self.apply_transform(in_tensor, params, flags, transform=transform)
        elif not to_apply.any():
            output = self.apply_non_transform(in_tensor, params, flags, transform=transform)
        else:  # If any torch.Tensor needs to be transformed.
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
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
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
        else:  # If any torch.Tensor needs to be transformed.
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
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
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
        else:  # If any torch.Tensor needs to be transformed.
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
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
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
        else:  # If any torch.Tensor needs to be transformed.
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
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        params, flags = self._process_kwargs_to_params_and_flags(
            self._params if params is None else params, flags, **kwargs
        )

        batch_prob = params["batch_prob"]
        to_apply = batch_prob > 0.5  # NOTE: in case of Relaxed Distributions.
        if to_apply.all():
            output = self.apply_transform_class(input, params, flags, transform=transform)
        elif not to_apply.any():
            output = self.apply_non_transform_class(input, params, flags, transform=transform)
        else:  # If any torch.Tensor needs to be transformed.
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
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process masks corresponding to the inputs that are no transformation applied."""
        raise NotImplementedError

    def apply_transform_mask(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process masks corresponding to the inputs that are transformed."""
        raise NotImplementedError

    def apply_non_transform_box(
        self,
        input: Boxes,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> Boxes:
        """Process boxes corresponding to the inputs that are no transformation applied."""
        return input

    def apply_transform_box(
        self,
        input: Boxes,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> Boxes:
        """Process boxes corresponding to the inputs that are transformed."""
        raise NotImplementedError

    def apply_non_transform_keypoint(
        self,
        input: Keypoints,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> Keypoints:
        """Process keypoints corresponding to the inputs that are no transformation applied."""
        return input

    def apply_transform_keypoint(
        self,
        input: Keypoints,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> Keypoints:
        """Process keypoints corresponding to the inputs that are transformed."""
        raise NotImplementedError

    def apply_non_transform_class(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process class tags corresponding to the inputs that are no transformation applied."""
        return input

    def apply_transform_class(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process class tags corresponding to the inputs that are transformed."""
        raise NotImplementedError

    def apply_func(
        self,
        in_tensor: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if flags is None:
            flags = self.flags

        output = self.transform_inputs(in_tensor, params, flags)

        return output
