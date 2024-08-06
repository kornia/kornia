from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast

import torch

import kornia.augmentation as K
from kornia.augmentation.base import _AugmentationBase
from kornia.augmentation.utils import override_parameters
from kornia.core import ImageModule, Module, Tensor, as_tensor
from kornia.core.module import ImageModuleMixIn
from kornia.utils import eye_like

from .base import ImageSequentialBase
from .params import ParamItem

__all__ = ["ImageSequential"]


class ImageModuleForSequentialMixIn(ImageModuleMixIn):
    _disable_features: bool = False

    @property
    def disable_features(self) -> bool:
        return self._disable_features

    @disable_features.setter
    def disable_features(self, value: bool = True) -> None:
        self._disable_features = value

    def disable_item_features(self, *args: Module) -> None:
        for arg in args:
            if isinstance(arg, (ImageModule,)):
                arg.disable_features = True


class ImageSequential(ImageSequentialBase, ImageModuleForSequentialMixIn):
    r"""Sequential for creating kornia image processing pipeline.

    Args:
        *args : a list of kornia augmentation and image operation modules.
        same_on_batch: apply the same transformation across the batch.
            If None, it will not overwrite the function-wise settings.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
            to the batch form (False). If None, it will not overwrite the function-wise settings.
        random_apply: randomly select a sublist (order agnostic) of args to
            apply transformation. The selection probability aligns to the ``random_apply_weights``.
            If int, a fixed number of transformations will be selected.
            If (a,), x number of transformations (a <= x <= len(args)) will be selected.
            If (a, b), x number of transformations (a <= x <= b) will be selected.
            If True, the whole list of args will be processed as a sequence in a random order.
            If False, the whole list of args will be processed as a sequence in original order.
        random_apply_weights: a list of selection weights for each operation. The length shall be as
            same as the number of operations. By default, operations are sampled uniformly.

    .. note::
        Transformation matrix returned only considers the transformation applied in ``kornia.augmentation`` module.
        Those transformations in ``kornia.geometry`` will not be taken into account.

    Examples:
        >>> _ = torch.manual_seed(77)
        >>> import kornia
        >>> input = torch.randn(2, 3, 5, 6)
        >>> aug_list = ImageSequential(
        ...     kornia.color.BgrToRgb(),
        ...     kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...     kornia.filters.MedianBlur((3, 3)),
        ...     kornia.augmentation.RandomAffine(360, p=1.0),
        ...     kornia.enhance.Invert(),
        ...     kornia.augmentation.RandomMixUpV2(p=1.0),
        ...     same_on_batch=True,
        ...     random_apply=10,
        ... )
        >>> out = aug_list(input)
        >>> out.shape
        torch.Size([2, 3, 5, 6])

        Reproduce with provided params.
        >>> out2 = aug_list(input, params=aug_list._params)
        >>> torch.equal(out, out2)
        True

    Perform ``OneOf`` transformation with ``random_apply=1`` and ``random_apply_weights`` in ``ImageSequential``.

        >>> import kornia
        >>> input = torch.randn(2, 3, 5, 6)
        >>> aug_list = ImageSequential(
        ...     kornia.color.BgrToRgb(),
        ...     kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...     kornia.filters.MedianBlur((3, 3)),
        ...     kornia.augmentation.RandomAffine(360, p=1.0),
        ...     random_apply=1,
        ...     random_apply_weights=[0.5, 0.3, 0.2, 0.5]
        ... )
        >>> out= aug_list(input)
        >>> out.shape
        torch.Size([2, 3, 5, 6])
    """

    def __init__(
        self,
        *args: Module,
        same_on_batch: Optional[bool] = None,
        keepdim: Optional[bool] = None,
        random_apply: Union[int, bool, Tuple[int, int]] = False,
        random_apply_weights: Optional[List[float]] = None,
        if_unsupported_ops: str = "raise",
        disable_item_features: bool = True,
        disable_sequential_features: bool = False,
    ) -> None:
        if disable_item_features:
            self.disable_item_features(*args)
        if disable_sequential_features:
            self.disable_features = True
        super().__init__(*args, same_on_batch=same_on_batch, keepdim=keepdim)

        self.random_apply = self._read_random_apply(random_apply, len(args))
        if random_apply_weights is not None and len(random_apply_weights) != len(self):
            raise ValueError(
                "The length of `random_apply_weights` must be as same as the number of operations."
                f"Got {len(random_apply_weights)} and {len(self)}."
            )
        self.random_apply_weights = as_tensor(random_apply_weights or torch.ones((len(self),)))
        self.if_unsupported_ops = if_unsupported_ops

    def _read_random_apply(
        self, random_apply: Union[int, bool, Tuple[int, int]], max_length: int
    ) -> Union[Tuple[int, int], bool]:
        """Process the scenarios for random apply."""
        if isinstance(random_apply, (bool,)) and random_apply is False:
            random_apply = False
        elif isinstance(random_apply, (bool,)) and random_apply is True:
            random_apply = (max_length, max_length + 1)
        elif isinstance(random_apply, (int,)):
            random_apply = (random_apply, random_apply + 1)
        elif (
            isinstance(random_apply, (tuple,))
            and len(random_apply) == 2
            and isinstance(random_apply[0], (int,))
            and isinstance(random_apply[1], (int,))
        ):
            random_apply = (random_apply[0], random_apply[1] + 1)
        elif isinstance(random_apply, (tuple,)) and len(random_apply) == 1 and isinstance(random_apply[0], (int,)):
            random_apply = (random_apply[0], max_length + 1)
        else:
            raise ValueError(f"Non-readable random_apply. Got {random_apply}.")
        if random_apply is not False and not (
            isinstance(random_apply, (tuple,))
            and len(random_apply) == 2
            and isinstance(random_apply[0], (int,))
            and isinstance(random_apply[0], (int,))
        ):
            raise AssertionError(f"Expect a tuple of (int, int). Got {random_apply}.")
        return random_apply

    def get_random_forward_sequence(self, with_mix: bool = True) -> Tuple[Iterator[Tuple[str, Module]], bool]:
        """Get a forward sequence when random apply is in need.

        Args:
            with_mix: if to require a mix augmentation for the sequence.

        Note:
            Mix augmentations (e.g. RandomMixUp) will be only applied once even in a random forward.
        """
        if isinstance(self.random_apply, tuple):
            num_samples = int(torch.randint(*self.random_apply, (1,)).item())
        else:
            raise TypeError(f"random apply should be a tuple. Gotcha {type(self.random_apply)}")

        multinomial_weights = self.random_apply_weights.clone()
        # Mix augmentation can only be applied once per forward
        mix_indices = self.get_mix_augmentation_indices(self.named_children())
        # kick out the mix augmentations
        multinomial_weights[mix_indices] = 0
        indices = torch.multinomial(
            multinomial_weights,
            num_samples,
            # enable replacement if non-mix augmentation is less than required
            replacement=num_samples > multinomial_weights.sum().item(),
        )

        mix_added = False
        if with_mix and len(mix_indices) != 0:
            # Make the selection fair.
            if (torch.rand(1) < ((len(mix_indices) + len(indices)) / len(self))).item():
                indices[-1] = torch.multinomial((~multinomial_weights.bool()).float(), 1)
                indices = indices[torch.randperm(len(indices))]
                mix_added = True

        return self.get_children_by_indices(indices), mix_added

    def get_mix_augmentation_indices(self, named_modules: Iterator[Tuple[str, Module]]) -> List[int]:
        """Get all the mix augmentations since they are label-involved.

        Special operations needed for label-involved augmentations.
        """
        # NOTE: MixV2 will not be a special op in the future.
        return [idx for idx, (_, child) in enumerate(named_modules) if isinstance(child, K.MixAugmentationBaseV2)]

    def get_forward_sequence(self, params: Optional[List[ParamItem]] = None) -> Iterator[Tuple[str, Module]]:
        if params is None:
            # Mix augmentation can only be applied once per forward
            mix_indices = self.get_mix_augmentation_indices(self.named_children())

            if self.random_apply:
                return self.get_random_forward_sequence()[0]

            if len(mix_indices) > 1:
                raise ValueError(
                    "Multiple mix augmentation is prohibited without enabling random_apply."
                    f"Detected {len(mix_indices)} mix augmentations."
                )

            return self.named_children()

        return self.get_children_by_params(params)

    def forward_parameters(self, batch_shape: torch.Size) -> List[ParamItem]:
        named_modules: Iterator[Tuple[str, Module]] = self.get_forward_sequence()

        params: List[ParamItem] = []
        mod_param: Union[Dict[str, Tensor], List[ParamItem]]
        for name, module in named_modules:
            if isinstance(module, (_AugmentationBase, K.MixAugmentationBaseV2, ImageSequentialBase)):
                mod_param = module.forward_parameters(batch_shape)
                param = ParamItem(name, mod_param)
            else:
                param = ParamItem(name, None)
            batch_shape = _get_new_batch_shape(param, batch_shape)
            params.append(param)
        return params

    def identity_matrix(self, input: Tensor) -> Tensor:
        """Return identity matrix."""
        return eye_like(3, input)

    def get_transformation_matrix(
        self,
        input: Tensor,
        params: Optional[List[ParamItem]] = None,
        recompute: bool = False,
        extra_args: Dict[str, Any] = {},
    ) -> Optional[Tensor]:
        """Compute the transformation matrix according to the provided parameters.

        Args:
            input: the input tensor.
            params: params for the sequence.
            recompute: if to recompute the transformation matrix according to the params.
                default: False.
        """
        if params is None:
            raise NotImplementedError("requires params to be provided.")
        named_modules: Iterator[Tuple[str, Module]] = self.get_forward_sequence(params)

        # Define as 1 for broadcasting
        res_mat: Optional[Tensor] = None
        for (_, module), param in zip(named_modules, params if params is not None else []):
            if isinstance(module, (K.GeometricAugmentationBase2D,)) and isinstance(param.data, dict):
                ori_shape = input.shape
                try:
                    input = module.transform_tensor(input)
                except ValueError:
                    # Ignore error for 5-dim video
                    pass
                # Standardize shape
                if recompute:
                    flags = override_parameters(module.flags, extra_args, in_place=False)
                    mat = module.generate_transformation_matrix(input, param.data, flags)
                elif module._transform_matrix is not None:
                    mat = as_tensor(module._transform_matrix, device=input.device, dtype=input.dtype)
                else:
                    raise RuntimeError(f"{module}._transform_matrix is None while `recompute=False`.")
                res_mat = mat if res_mat is None else mat @ res_mat
                input = module.transform_output_tensor(input, ori_shape)
                if module.keepdim and ori_shape != input.shape:
                    res_mat = res_mat.squeeze()
            elif isinstance(module, (ImageSequentialBase,)):
                # If not augmentationSequential
                if isinstance(module, (K.AugmentationSequential,)) and not recompute:
                    mat = as_tensor(module._transform_matrix, device=input.device, dtype=input.dtype)
                else:
                    maybe_param_data = cast(Optional[List[ParamItem]], param.data)
                    _mat = module.get_transformation_matrix(
                        input, maybe_param_data, recompute=recompute, extra_args=extra_args
                    )
                    mat = module.identity_matrix(input) if _mat is None else _mat
                res_mat = mat if res_mat is None else mat @ res_mat
        return res_mat

    # TODO: Make this as a class property to avoid running every time.
    def is_intensity_only(self, strict: bool = True) -> bool:
        """Check if all transformations are intensity-based.

        Args:
            strict: if strict is False, it will allow non-augmentation Modules to be passed.
                e.g. `kornia.enhance.AdjustBrightness` will be recognized as non-intensity module
                if strict is set to True.

        Note: patch processing would break the continuity of labels (e.g. bbounding boxes, masks).
        """
        for arg in self.children():
            if isinstance(arg, (ImageSequential,)) and not arg.is_intensity_only(strict):
                return False
            elif isinstance(arg, (ImageSequential,)):
                pass
            elif isinstance(arg, K.IntensityAugmentationBase2D):
                pass
            elif strict:
                # disallow non-registered ops if in strict mode
                # TODO: add an ops register module
                return False
        return True

    def __call__(
        self,
        *inputs: Any,
        input_names_to_handle: Optional[List[Any]] = None,
        output_type: str = "tensor",
        **kwargs: Any,
    ) -> Any:
        """Overwrites the __call__ function to handle various inputs.

        Args:
            input_names_to_handle: List of input names to convert, if None, handle all inputs.
            output_type: Desired output type ('tensor', 'numpy', or 'pil').

        Returns:
            Callable: Decorated function with converted input and output types.
        """

        # Wrap the forward method with the decorator
        if not self._disable_features:
            decorated_forward = self.convert_input_output(
                input_names_to_handle=input_names_to_handle, output_type=output_type
            )(super().__call__)
            _output_image = decorated_forward(*inputs, **kwargs)
            if output_type == "tensor":
                self._output_image = self._detach_tensor_to_cpu(_output_image)
            else:
                self._output_image = _output_image
        else:
            _output_image = super().__call__(*inputs, **kwargs)
        return _output_image


def _get_new_batch_shape(param: ParamItem, batch_shape: torch.Size) -> torch.Size:
    """Get the new batch shape if the augmentation changes the image size.

    Note:
       Augmentations that change the image size must provide the parameter `output_size`.
    """
    if param.data is None:
        return batch_shape
    if isinstance(param.data, list):
        for p in param.data:
            batch_shape = _get_new_batch_shape(p, batch_shape)
    elif "output_size" in param.data:
        if not (param.data["batch_prob"] > 0.5)[0]:
            # Augmentations that change the image size must be applied equally to all elements in batch.
            # If the augmentation is not applied, return the same batch shape.
            return batch_shape
        new_batch_shape = list(batch_shape)
        new_batch_shape[-2:] = param.data["output_size"][0]
        batch_shape = torch.Size(new_batch_shape)
    return batch_shape
