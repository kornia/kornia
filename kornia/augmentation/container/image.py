from itertools import zip_longest
from typing import Any, Iterator, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

import kornia
from kornia.augmentation.base import (
    _AugmentationBase,
    GeometricAugmentationBase2D,
    IntensityAugmentationBase2D,
    MixAugmentationBase,
    TensorWithTransformMat,
)

from .base import ParamItem, SequentialBase
from .utils import ApplyInverseInterface, InputApplyInverse

__all__ = ["ImageSequential"]


class ImageSequential(SequentialBase):
    r"""Sequential for creating kornia image processing pipeline.

    Args:
        *args : a list of kornia augmentation and image operation modules.
        same_on_batch: apply the same transformation across the batch.
            If None, it will not overwrite the function-wise settings.
        return_transform: if ``True`` return the matrix describing the transformation
            applied to each. If None, it will not overwrite the function-wise settings.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
            to the batch form (False). If None, it will not overwrite the function-wise settings.
        random_apply: randomly select a sublist (order agnostic) of args to
            apply transformation.
            If int, a fixed number of transformations will be selected.
            If (a,), x number of transformations (a <= x <= len(args)) will be selected.
            If (a, b), x number of transformations (a <= x <= b) will be selected.
            If True, the whole list of args will be processed as a sequence in a random order.
            If False, the whole list of args will be processed as a sequence in original order.

    .. note::
        Transformation matrix returned only considers the transformation applied in ``kornia.augmentation`` module.
        Those transformations in ``kornia.geometry`` will not be taken into account.

    Examples:
        >>> _ = torch.manual_seed(77)
        >>> import kornia
        >>> input, label = torch.randn(2, 3, 5, 6), torch.tensor([0, 1])
        >>> aug_list = ImageSequential(
        ...     kornia.color.BgrToRgb(),
        ...     kornia.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...     kornia.filters.MedianBlur((3, 3)),
        ...     kornia.augmentation.RandomAffine(360, p=1.0),
        ...     kornia.enhance.Invert(),
        ...     kornia.augmentation.RandomMixUp(p=1.0),
        ...     return_transform=True,
        ...     same_on_batch=True,
        ...     random_apply=10,
        ... )
        >>> out, lab = aug_list(input, label=label)
        >>> lab
        tensor([[0.0000, 1.0000, 0.1214],
                [1.0000, 0.0000, 0.1214]])
        >>> out[0].shape, out[1].shape
        (torch.Size([2, 3, 5, 6]), torch.Size([2, 3, 3]))

        Reproduce with provided params.
        >>> out2, lab2 = aug_list(input, label=label, params=aug_list._params)
        >>> torch.equal(out[0], out2[0]), torch.equal(out[1], out2[1]), torch.equal(lab[1], lab2[1])
        (True, True, True)
    """

    def __init__(
        self,
        *args: nn.Module,
        same_on_batch: Optional[bool] = None,
        return_transform: Optional[bool] = None,
        keepdim: Optional[bool] = None,
        random_apply: Union[int, bool, Tuple[int, int]] = False,
        if_unsupported_ops: str = 'raise'
    ) -> None:
        super().__init__(*args, same_on_batch=same_on_batch, return_transform=return_transform, keepdim=keepdim)

        self.random_apply: Union[Tuple[int, int], bool] = self._read_random_apply(random_apply, len(args))
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
        multinomial_weights = torch.ones((len(self),))
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
        input: TensorWithTransformMat,
        label: Optional[torch.Tensor],
        module: Optional[nn.Module],
        param: ParamItem,
    ) -> Tuple[TensorWithTransformMat, Optional[torch.Tensor]]:
        if module is None:
            module = self.get_submodule(param.name)
        return self.apply_inverse_func.apply_trans(input, label, module, param)  # type: ignore

    def forward_parameters(self, batch_shape: torch.Size) -> List[ParamItem]:
        named_modules: Iterator[Tuple[str, nn.Module]] = self.get_forward_sequence()

        params: List[ParamItem] = []
        mod_param: Union[dict, list]
        for name, module in named_modules:
            if isinstance(module, (_AugmentationBase, MixAugmentationBase)):
                mod_param = module.forward_parameters(batch_shape)
                param = ParamItem(name, mod_param)
            elif isinstance(module, ImageSequential):
                mod_param = module.forward_parameters(batch_shape)
                param = ParamItem(name, mod_param)
            else:
                param = ParamItem(name, None)
            params.append(param)
        return params

    def contains_label_operations(self, params: List[ParamItem]) -> bool:
        """Check if current sequential contains label-involved operations like MixUp."""
        for param in params:
            if param.name.startswith("RandomMixUp") or param.name.startswith("RandomCutMix"):
                return True
        return False

    def __packup_output__(
        self, output: TensorWithTransformMat, label: Optional[torch.Tensor] = None
    ) -> Union[TensorWithTransformMat, Tuple[TensorWithTransformMat, torch.Tensor]]:
        if self.return_label:
            return output, label  # type: ignore
            # Implicitly indicating the label cannot be optional since there is a mix aug
        return output

    def get_transformation_matrix(
        self, input: torch.Tensor, params: Optional[List[ParamItem]] = None,
    ) -> torch.Tensor:
        """Compute the transformation matrix according to the provided parameters."""
        if params is None:
            raise NotImplementedError("requires params to be provided.")
        named_modules: Iterator[Tuple[str, nn.Module]] = self.get_forward_sequence(params)

        res_mat: torch.Tensor = kornia.eye_like(3, input)
        for (_, module), param in zip(named_modules, params):
            if isinstance(module, (_AugmentationBase, MixAugmentationBase)):
                mat = module.compute_transformation(input, param.data)  # type: ignore
                res_mat = mat @ res_mat
            elif isinstance(module, (ImageSequential,)):
                mat = module.get_transformation_matrix(input, param.data)  # type: ignore
                res_mat = mat @ res_mat
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
        self,
        input: torch.Tensor,
        params: Optional[List[ParamItem]] = None,
    ) -> torch.Tensor:
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
                input = self.apply_inverse_func.inverse(input, module, param)
            else:
                pass
                # raise NotImplementedError(f"`inverse` is not implemented for {module}.")

        return input

    def forward(  # type: ignore
        self,
        input: TensorWithTransformMat,
        label: Optional[torch.Tensor] = None,
        params: Optional[List[ParamItem]] = None,
    ) -> Union[TensorWithTransformMat, Tuple[TensorWithTransformMat, torch.Tensor]]:
        self.clear_state()
        if params is None:
            if isinstance(input, (tuple, list)):
                inp = input[0]
            else:
                inp = input
            _, out_shape = self.autofill_dim(inp, dim_range=(2, 4))
            params = self.forward_parameters(out_shape)
        if self.return_label is None:
            self.return_label = label is not None or self.contains_label_operations(params)
        for param in params:
            module = self.get_submodule(param.name)
            input, label = self.apply_to_input(input, label, module, param=param)  # type: ignore
            if isinstance(module, (_AugmentationBase, MixAugmentationBase, SequentialBase)):
                param = ParamItem(param.name, module._params)
            else:
                param = ParamItem(param.name, None)
            self.update_params(param)
        return self.__packup_output__(input, label)
