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

from kornia.augmentation.random_generator import RandomGeneratorBase
from kornia.augmentation.utils import (
    _transform_output_shape,
    override_parameters,
)
from kornia.core.utils import is_autocast_enabled, is_exporting
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints

TensorWithTransformMat = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

# Sentinel for ``_commit_state`` fields: distinguishes "not provided" from an explicit ``None``.
_UNSET: Any = object()


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

    # Flag for whether this augmentation supports ONNX export. Override to False in subclasses
    # that use ops the legacy tracer can't lower (e.g. ``torch.histc``,
    # ``torch.distributions.Beta``).
    # Users can introspect via ``aug.exportable``; CI iterates the known-exportable
    # subset in ``tests/augmentation/test_onnx_export.py``.
    ONNX_EXPORTABLE = True

    @property
    def exportable(self) -> bool:
        """Whether this augmentation supports ONNX export via the legacy tracer at opset 20.

        Reflects the class-level ``ONNX_EXPORTABLE`` flag. Note that ``True`` here
        means *the graph traces and exports* — it does not guarantee the resulting
        ONNX runtime output is bit-equivalent to eager. See the categorisation in
        ``tests/augmentation/test_onnx_export.py`` for the numerical-correctness
        signal per augmentation.
        """
        return bool(self.ONNX_EXPORTABLE)

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
        self._param_generator: Optional[RandomGeneratorBase] = None
        self.flags: Dict[str, Any] = {}
        self.set_rng_device_and_dtype(torch.device("cpu"), torch.get_default_dtype())

    apply_transform: Callable[..., torch.Tensor] = _apply_transform_unimplemented

    def to(self, *args: Any, **kwargs: Any) -> "_BasicAugmentationBase":
        r"""Set the device and dtype for the random number generator."""
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.set_rng_device_and_dtype(device, dtype)
        return super().to(*args, **kwargs)

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], *args: Any, **kwargs: Any) -> "_BasicAugmentationBase":
        # nn.Module.to/.cuda/.cpu/.half move children by recursing through `_apply`, not `.to`,
        # so a container like `AugmentationSequential(...).to("cuda")` never triggers our `to`
        # override and leaves parameter sampling on CPU — every forward then generates on the
        # host and copies to the device (measured ~5x slower on GPU pipelines). Mirror the device
        # and dtype of the moved tensors onto the random generator so container moves behave like
        # a direct `.to` on the augmentation.
        out = super()._apply(fn, *args, **kwargs)
        probe = fn(torch.zeros((), device=self.device, dtype=self.dtype))
        dtype = probe.dtype if probe.is_floating_point() else self.dtype
        self.set_rng_device_and_dtype(probe.device, dtype)
        return out

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
            batch_prob = torch.ones(1, device=self.device, dtype=self.dtype)
        elif p_batch == 0:
            batch_prob = torch.zeros(1, device=self.device, dtype=self.dtype)
        else:
            batch_prob = (torch.rand(1, device=self.device) < p_batch).to(self.dtype)

        # Per-sample gate. `p` and `same_on_batch` are Python values, so these branches are
        # resolved at trace time (no graph break).
        elem_prob: torch.Tensor
        if p == 1:
            elem_prob = torch.ones(batch_shape[0], device=self.device, dtype=self.dtype)
        elif p == 0:
            elem_prob = torch.zeros(batch_shape[0], device=self.device, dtype=self.dtype)
        elif same_on_batch:
            elem_prob = (torch.rand(1, device=self.device) < p).to(self.dtype).expand(batch_shape[0])
        else:
            elem_prob = (torch.rand(batch_shape[0], device=self.device) < p).to(self.dtype)

        # Branchless combine (replaces the data-dependent `if batch_prob.sum() == 1`).
        # batch_prob is (1,) in {0, 1}: a selected batch yields elem_prob, a deselected one
        # yields zeros — identical to the old branch, without the graph break.
        batch_prob = batch_prob * elem_prob

        if len(batch_prob.shape) == 2:
            return batch_prob[..., 0]
        return batch_prob

    def _commit_state(
        self,
        *,
        params: Any = _UNSET,
        transform_matrix: Any = _UNSET,
        lazy_matrix_args: Any = _UNSET,
    ) -> None:
        """Record per-call state on ``self`` for post-call retrieval.

        Covers ``._params`` (and, on the rigid/affine subclass, ``.transform_matrix`` /
        ``_lazy_matrix_args``). This is the single writer of that state. It is skipped inside a
        ``torch.export`` capture, which rejects attribute mutation in ``forward``; the captured
        image output is unchanged — only reading the state back afterwards (meaningless in an
        exported graph) is skipped. Each field is written only when explicitly passed (``None`` is
        a valid value, hence the ``_UNSET`` sentinel).
        """
        if is_exporting():
            return
        if params is not _UNSET:
            self._params = params
        if transform_matrix is not _UNSET:
            self._transform_matrix = transform_matrix
        if lazy_matrix_args is not _UNSET:
            self._lazy_matrix_args = lazy_matrix_args

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
            self._commit_state(params=params)
        else:
            self._commit_state(params=params)
            params = override_parameters(params, kwargs, in_place=False)

        flags = override_parameters(flags, kwargs, in_place=False)
        return params, flags

    def forward_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        batch_prob = self.__batch_prob_generator__(batch_shape, self.p, self.p_batch, self.same_on_batch)
        _params = self.generate_parameters(batch_shape)
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

    @staticmethod
    def _blend_by_prob(
        transformed: torch.Tensor, not_transformed: torch.Tensor, to_apply: torch.Tensor
    ) -> torch.Tensor:
        """Select transformed vs non-transformed samples element-wise by ``to_apply``.

        When the two branches share a shape this is a ``torch.where`` blend (onnx- and
        fullgraph-friendly). Shape-changing augmentations (e.g. crop/resize) whose branches
        differ in spatial size fall back to a Python branch on ``to_apply.any()``, which is
        not onnx-exportable.
        """
        if transformed.shape == not_transformed.shape and transformed.shape[0] == to_apply.shape[0]:
            to_apply_expanded = to_apply.view(-1, *([1] * (len(transformed.shape) - 1)))
            return torch.where(to_apply_expanded, transformed, not_transformed)
        return transformed if bool(to_apply.any()) else not_transformed

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

        ori_shape = input.shape
        in_tensor = self.transform_tensor(input)

        self.validate_tensor(in_tensor)

        output_transformed = self.apply_transform(in_tensor, params, flags, transform=transform)

        if self.p == 1.0 and self.p_batch == 1.0:
            # Always applied (static probabilities): the output is unconditionally the
            # transformed one. Skip the non-transform branch and the blend entirely — this
            # also makes shape-changing augmentations (e.g. Resize) fullgraph-compilable,
            # since the data-dependent shape comparison / `to_apply.any()` fallback is avoided.
            # (The `to_apply` gate is only needed on the p < 1 path below, so it is not computed
            # here.)
            output = output_transformed
        else:
            to_apply = torch.atleast_1d(params["batch_prob"] > 0.5)
            output_not_transformed = self.apply_non_transform(in_tensor, params, flags, transform=transform)
            output = self._blend_by_prob(output_transformed, output_not_transformed, to_apply)

        if is_autocast_enabled():
            output = output.type(input.dtype)

        # `_transform_output_shape` only reshapes (preserves dtype), so no second autocast cast is
        # needed after it — the cast above already restored `input.dtype`.
        output = _transform_output_shape(output, ori_shape) if self.keepdim else output
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
        to_apply = torch.atleast_1d(batch_prob > 0.5)
        ori_shape = input.shape

        shape = params["forward_input_shape"]
        in_tensor = self.transform_tensor(input, shape=shape, match_channel=False)

        self.validate_tensor(in_tensor)

        output_transformed = self.apply_transform_mask(in_tensor, params, flags, transform=transform)
        output_not_transformed = self.apply_non_transform_mask(in_tensor, params, flags, transform=transform)

        output = self._blend_by_prob(output_transformed, output_not_transformed, to_apply)

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
        to_apply = torch.atleast_1d(batch_prob > 0.5)

        output_transformed = self.apply_transform_box(input, params, flags, transform=transform)
        output_not_transformed = self.apply_non_transform_box(input, params, flags, transform=transform)

        data_transformed = output_transformed.data
        data_not_transformed = output_not_transformed.data

        if is_autocast_enabled():
            data_transformed = data_transformed.type(input.data.dtype)
            data_not_transformed = data_not_transformed.type(input.data.dtype)

        blended_data = self._blend_by_prob(data_transformed, data_not_transformed, to_apply)

        # Reuse the not-transformed Boxes container (preserves mode/_N/is_batched/etc.)
        # and swap in the blended data, same effect as the index_put on .data.
        output = output_not_transformed.clone()
        output._data = blended_data
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
        to_apply = torch.atleast_1d(batch_prob > 0.5)

        output_transformed = self.apply_transform_keypoint(input, params, flags, transform=transform)
        output_not_transformed = self.apply_non_transform_keypoint(input, params, flags, transform=transform)

        data_transformed = output_transformed.data
        data_not_transformed = output_not_transformed.data

        if is_autocast_enabled():
            data_transformed = data_transformed.type(input.data.dtype)
            data_not_transformed = data_not_transformed.type(input.data.dtype)

        blended_data = self._blend_by_prob(data_transformed, data_not_transformed, to_apply)

        output = output_not_transformed.clone()
        output._data = blended_data
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
        to_apply = torch.atleast_1d(batch_prob > 0.5)

        output_transformed = self.apply_transform_class(input, params, flags, transform=transform)
        output_not_transformed = self.apply_non_transform_class(input, params, flags, transform=transform)

        output = self._blend_by_prob(output_transformed, output_not_transformed, to_apply)

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
