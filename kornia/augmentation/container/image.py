from itertools import zip_longest
from typing import Any, cast, Dict, Iterator, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

import kornia
from kornia.augmentation import (
    GeometricAugmentationBase2D,
    IntensityAugmentationBase2D,
    MixAugmentationBase,
    RandomCrop,
)
from kornia.augmentation.base import _AugmentationBase
from kornia.augmentation.container.base import ParamItem, SequentialBase
from kornia.augmentation.container.utils import ApplyInverseInterface, InputApplyInverse

__all__ = ["ImageSequential"]


class ImageSequential(SequentialBase):
    r"""Sequential for creating kornia image processing pipeline.

    Args:
        *args : a list of kornia augmentation and image operation modules.
        same_on_batch: apply the same transformation across the batch.
            If None, it will not overwrite the function-wise settings.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
            to the batch form (False). If None, it will not overwrite the function-wise settings.
        random_apply: randomly select a sublist (order agnostic) of args to
            apply transformation. The selection probablity aligns to the ``random_apply_weights``.
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
        >>> input, label = torch.randn(2, 3, 5, 6), torch.tensor([0, 1])
        >>> aug_list = ImageSequential(
        ...     kornia.color.BgrToRgb(),
        ...     kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...     kornia.filters.MedianBlur((3, 3)),
        ...     kornia.augmentation.RandomAffine(360, p=1.0),
        ...     kornia.enhance.Invert(),
        ...     kornia.augmentation.RandomMixUp(p=1.0),
        ...     same_on_batch=True,
        ...     random_apply=10,
        ... )
        >>> out, lab = aug_list(input, label=label)
        >>> lab
        tensor([[0.0000, 1.0000, 0.1214],
                [1.0000, 0.0000, 0.1214]])
        >>> out.shape
        torch.Size([2, 3, 5, 6])

        Reproduce with provided params.
        >>> out2, lab2 = aug_list(input, label=label, params=aug_list._params)
        >>> torch.equal(out, out2), torch.equal(lab, lab2)
        (True, True)

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
        *args: nn.Module,
        same_on_batch: Optional[bool] = None,
        return_transform: Optional[bool] = None,
        keepdim: Optional[bool] = None,
        random_apply: Union[int, bool, Tuple[int, int]] = False,
        random_apply_weights: Optional[List[float]] = None,
        if_unsupported_ops: str = "raise",
    ) -> None:
        if return_transform is not None:
            raise ValueError(
                "`return_transform` is deprecated. Please access"
                " `.transform_matrix` in `AugmentationSequential` instead."
            )
        super().__init__(*args, same_on_batch=same_on_batch, return_transform=return_transform, keepdim=keepdim)

        self.random_apply: Union[Tuple[int, int], bool] = self._read_random_apply(random_apply, len(args))
        if random_apply_weights is not None and len(random_apply_weights) != len(self):
            raise ValueError(
                "The length of `random_apply_weights` must be as same as the number of operations."
                f"Got {len(random_apply_weights)} and {len(self)}."
            )
        self.random_apply_weights = torch.as_tensor(random_apply_weights or torch.ones((len(self),)))
        self.return_label: Optional[bool] = None
        self.apply_inverse_func: Type[ApplyInverseInterface] = InputApplyInverse
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

    def get_random_forward_sequence(self, with_mix: bool = True) -> Tuple[Iterator[Tuple[str, nn.Module]], bool]:
        """Get a forward sequence when random apply is in need.

        Note:
            Mix augmentations (e.g. RandomMixUp) will be only applied once even in a random forward.
        """
        num_samples = int(torch.randint(*self.random_apply, (1,)).item())  # type: ignore
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

    def get_mix_augmentation_indices(self, named_modules: Iterator[Tuple[str, nn.Module]]) -> List[int]:
        """Get all the mix augmentations since they are label-involved.

        Special operations needed for label-involved augmentations.
        """
        indices = []
        for idx, (_, child) in enumerate(named_modules):
            if isinstance(child, (MixAugmentationBase,)):
                indices.append(idx)
        return indices

    def get_forward_sequence(self, params: Optional[List[ParamItem]] = None) -> Iterator[Tuple[str, nn.Module]]:
        if params is None:
            # Mix augmentation can only be applied once per forward
            mix_indices = self.get_mix_augmentation_indices(self.named_children())

            if self.random_apply:
                return self.get_random_forward_sequence()[0]

            if len(mix_indices) > 1:
                raise ValueError(
                    "Multiple mix augmentation is prohibited without enabling random_apply."
                    f"Detected {len(mix_indices)}."
                )

            return self.named_children()

        return self.get_children_by_params(params)

    def apply_to_input(
        self,
        input: Tensor,
        label: Optional[Tensor],
        module: Optional[nn.Module],
        param: ParamItem,
        extra_args: Dict[str, Any]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if module is None:
            module = self.get_submodule(param.name)
        return self.apply_inverse_func.apply_trans(input, label, module, param, extra_args)  # type: ignore

    def forward_parameters(self, batch_shape: torch.Size) -> List[ParamItem]:
        named_modules: Iterator[Tuple[str, nn.Module]] = self.get_forward_sequence()

        params: List[ParamItem] = []
        mod_param: Union[dict, list]
        for name, module in named_modules:
            if isinstance(module, RandomCrop):
                mod_param = module.forward_parameters_precrop(batch_shape)
                param = ParamItem(name, mod_param)
            elif isinstance(module, (_AugmentationBase, MixAugmentationBase, ImageSequential)):
                mod_param = module.forward_parameters(batch_shape)
                param = ParamItem(name, mod_param)
            else:
                param = ParamItem(name, None)
            batch_shape = _get_new_batch_shape(param, batch_shape)
            params.append(param)
        return params

    def contains_label_operations(self, params: List[ParamItem]) -> bool:
        """Check if current sequential contains label-involved operations like MixUp."""
        for param in params:
            if param.name.startswith("RandomMixUp") or param.name.startswith("RandomCutMix"):
                return True
        return False

    def __packup_output__(
        self, output: Tensor, label: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if self.return_label:
            return output, label  # type: ignore
            # Implicitly indicating the label cannot be optional since there is a mix aug
        return output

    def identity_matrix(self, input) -> Tensor:
        """Return identity matrix."""
        return kornia.eye_like(3, input)

    def get_transformation_matrix(
        self, input: Tensor, params: Optional[List[ParamItem]] = None, recompute: bool = False
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
        named_modules: Iterator[Tuple[str, nn.Module]] = self.get_forward_sequence(params)

        # Define as 1 for broadcasting
        res_mat: Optional[Tensor] = None
        for (_, module), param in zip(named_modules, params if params is not None else []):
            if isinstance(module, (_AugmentationBase,)) and not isinstance(module, (MixAugmentationBase,)):
                pdata = cast(Dict[str, Tensor], param.data)
                to_apply = pdata['batch_prob']  # type: ignore
                ori_shape = input.shape
                try:
                    input = module.transform_tensor(input)
                except ValueError:
                    # Ignore error for 5-dim video
                    pass
                # Standardize shape
                if recompute:
                    mat: Tensor = self.identity_matrix(input)
                    try:
                        mat[to_apply] = module.compute_transformation(
                            input[to_apply], pdata, module.flags)  # type: ignore
                    except:
                        xx = module.compute_transformation(
                            input[to_apply], pdata, module.flags)
                        assert False, (mat[to_apply].shape, input[to_apply].shape, xx.shape)
                    mat[to_apply] = module.compute_transformation(
                        input[to_apply], pdata, module.flags)  # type: ignore
                else:
                    mat = torch.as_tensor(module._transform_matrix, device=input.device, dtype=input.dtype)
                res_mat = mat if res_mat is None else mat @ res_mat
                input = module.transform_output_tensor(input, ori_shape)
                if module.keepdim and ori_shape != input.shape:
                    res_mat = res_mat.squeeze()
            elif isinstance(module, (ImageSequential,)):
                # If not augmentationSequential
                if isinstance(module, (kornia.augmentation.AugmentationSequential,)) and not recompute:
                    mat = torch.as_tensor(module._transform_matrix, device=input.device, dtype=input.dtype)
                else:
                    mat = module.get_transformation_matrix(input, param.data)  # type: ignore
                res_mat = mat if res_mat is None else mat @ res_mat
        return res_mat

    def is_intensity_only(self, strict: bool = True) -> bool:
        """Check if all transformations are intensity-based.

        Args:
            strict: if strict is False, it will allow non-augmentation nn.Modules to be passed.
                e.g. `kornia.enhance.AdjustBrightness` will be recognized as non-intensity module
                if strict is set to True.

        Note: patch processing would break the continuity of labels (e.g. bbounding boxes, masks).
        """
        for arg in self.children():
            if isinstance(arg, (ImageSequential,)) and not arg.is_intensity_only(strict):
                return False
            elif isinstance(arg, (ImageSequential,)):
                pass
            elif isinstance(arg, IntensityAugmentationBase2D):
                pass
            elif strict:
                # disallow non-registered ops if in strict mode
                # TODO: add an ops register module
                return False
        return True

    def inverse(
        self, input: Tensor, params: Optional[List[ParamItem]] = None,
        extra_args: Dict[str, Any] = {}
    ) -> Tensor:
        """Inverse transformation.

        Used to inverse a tensor according to the performed transformation by a forward pass, or with respect to
        provided parameters.
        """
        if params is None:
            if self._params is None:
                raise ValueError(
                    "No parameters available for inversing, please run a forward pass first "
                    "or passing valid params into this function."
                )
            params = self._params

        for (name, module), param in zip_longest(list(self.get_forward_sequence(params))[::-1], params[::-1]):
            if isinstance(module, (_AugmentationBase, ImageSequential)):
                param = params[name] if name in params else param
            else:
                param = None

            if isinstance(module, IntensityAugmentationBase2D):
                pass  # Do nothing
            elif isinstance(module, ImageSequential) and module.is_intensity_only():
                pass  # Do nothing
            elif isinstance(module, ImageSequential):
                input = module.inverse(input, param.data)
            elif isinstance(module, (GeometricAugmentationBase2D,)):
                input = self.apply_inverse_func.inverse(input, module, param, extra_args=extra_args)
            else:
                pass
                # raise NotImplementedError(f"`inverse` is not implemented for {module}.")

        return input

    def forward(  # type: ignore
        self,
        input: Tensor,
        label: Optional[Tensor] = None,
        params: Optional[List[ParamItem]] = None,
        extra_args: Dict[str, Any] = {}
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        self.clear_state()
        if params is None:
            inp = input
            _, out_shape = self.autofill_dim(inp, dim_range=(2, 4))
            params = self.forward_parameters(out_shape)
        if self.return_label is None:
            self.return_label = label is not None or self.contains_label_operations(params)
        for param in params:
            module = self.get_submodule(param.name)
            input, label = self.apply_to_input(input, label, module, param=param, extra_args=extra_args)  # type: ignore
            if isinstance(module, (_AugmentationBase, MixAugmentationBase, SequentialBase)):
                param = ParamItem(param.name, module._params)
            else:
                param = ParamItem(param.name, None)
            self.update_params(param)
        return self.__packup_output__(input, label)


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
    elif 'output_size' in param.data:
        if not param.data['batch_prob'][0]:
            # Augmentations that change the image size must be applied equally to all elements in batch.
            # If the augmentation is not applied, return the same batch shape.
            return batch_shape
        new_batch_shape = list(batch_shape)
        new_batch_shape[-2:] = param.data['output_size'][0]
        batch_shape = torch.Size(new_batch_shape)
    return batch_shape
