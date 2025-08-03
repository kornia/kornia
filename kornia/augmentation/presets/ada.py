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

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch

from ...core import Device, Tensor  # noqa: TID252
from .. import (  # noqa: TID252
    AugmentationSequential,
    ColorJitter,
    ImageSequential,
    RandomAffine,
    RandomErasing,
    RandomGaussianNoise,
    RandomHorizontalFlip,
    RandomRotation90,
)
from ..base import _AugmentationBase  # noqa: TID252
from ..container.params import ParamItem  # noqa: TID252

_data_keys_type = List[str]
_inputs_type = Union[Tensor, Dict[str, Tensor]]


class AdaptiveDiscriminatorAugmentation(AugmentationSequential):
    r"""Implementation of Adaptive Discriminator Augmentation for GANs training as introduced in :cite:`Karras2020ada`.

    adjust a global probability p over all augmentations list to select a subset of images to augment
    based on an exponential moving average of the Discriminator's accuracy labeling real samples.


    Args:
        *args: a list of kornia augmentation modules, set to a default list if not specified.

        initial_p: initial global probability `p` for applying the augmentations

        adjustment_speed: float
            step size for updating the global probability `p`

        max_p: maximum allowed value for `p`

        target_real_acc: target Discriminator accuracy to guide `p` adjustments


        ema_lambda: EMA smoothing factor. The real accuracy EMA is what's used to determine the `p` update

        update_every: `p` update frequency (in steps)

        erasing_scale: scale range used for `RandomErasing` if default augmentations are used

        erasing_ratio: aspect ratio range used for `RandomErasing` if default augmentations are used

        erasing_fill_value: fill value used in `RandomErasing`

        same_on_batch: apply the same transformation across the batch

        data_keys: input types to apply augmentations on


        **kwargs: Additional keyword arguments passed to `AugmentationSequential`


    Examples:
        >>> from kornia.augmentation.presets.ada import AdaptiveDiscriminatorAugmentation
        >>> original = torch.randn(2, 3, 16, 16)
        >>> ada = AdaptiveDiscriminatorAugmentation()
        >>> augmented = ada(original)

    This example demonstrates using default augmentations with AdaptiveDiscriminatorAugmentation in a GAN training loop.


        >>> import kornia.augmentation as K
        >>> from kornia.augmentation.presets.ada import AdaptiveDiscriminatorAugmentation
        >>> originals = torch.randn(2, 3, 5, 6)
        >>> aug_list = [
        ...     K.RandomRotation90(times=(0, 3), p=1),
        ...     K.RandomAffine(degrees=10, translate=(.1, .1), scale=(.9, 1.1), p=1),
        ...     K.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.1, p=1),
        ... ]

        >>> ada = AdaptiveDiscriminatorAugmentation(*aug_list)
        >>> augmented = ada(original)

    This example demonstrates using custom augmentations with AdaptiveDiscriminatorAugmentation.
    """

    def __init__(
        self,
        *args: Union[_AugmentationBase, ImageSequential],
        initial_p: float = 1e-5,
        adjustment_speed: float = 1e-2,
        max_p: float = 0.8,
        target_real_acc: float = 0.85,
        ema_lambda: float = 0.99,
        update_every: int = 5,
        erasing_scale: Union[Tensor, Tuple[float, float]] = (0.02, 0.33),
        erasing_ratio: Union[Tensor, Tuple[float, float]] = (0.3, 3.3),
        erasing_fill_value: float = 0.0,
        data_keys: Optional[_data_keys_type] = None,
        same_on_batch: Optional[bool] = False,
        **kwargs: Any,
    ) -> None:
        if not args:
            args = self.default_ada_transfroms(erasing_scale, erasing_ratio, erasing_fill_value)

        super().__init__(
            *args,
            data_keys=data_keys
            if data_keys is not None
            else [
                "input",
            ],
            same_on_batch=same_on_batch,
            **kwargs,
        )

        if adjustment_speed <= 0:
            raise ValueError(f"Invalid `adjustment_speed` ({adjustment_speed}) — must be greater than 0")

        if not 0 <= target_real_acc <= 1:
            raise ValueError(f"Invalid `target_real_acc` ({target_real_acc}) — must be in [0, 1]")

        if not 0 <= ema_lambda <= 1:
            raise ValueError(f"Invalid `ema_lambda` ({ema_lambda}) — must be in [0, 1]")

        if update_every < 1:
            raise ValueError(f"Invalid `update_every` ({update_every}) — must be at least 1")

        if not 0 <= max_p <= 1:
            raise ValueError(f"Invalid `max_p` ({max_p}) — must be in [0, 1]")

        if not 0 <= initial_p <= 1:
            raise ValueError(f"Invalid `initial_p` ({initial_p}) — must be in [0, 1]")

        if initial_p > max_p:
            warnings.warn(
                f"`initial_p` ({initial_p}) is greater than `max_p` ({max_p}), resetting `initial_p` to `max_p`",
                stacklevel=2,
            )
            initial_p = max_p

        self.p = initial_p
        self.adjustment_speed = adjustment_speed
        self.max_p = max_p
        self.target_real_acc = target_real_acc
        self.ema_lambda = ema_lambda
        self.update_every = update_every
        self.real_acc_ema: float = 0.5
        self._num_calls = 0  # -update_every  # to avoid updating in the first `update_every` steps

    def default_ada_transfroms(
        self, scale: Union[Tensor, Tuple[float, float]], ratio: Union[Tensor, Tuple[float, float]], value: float
    ) -> Tuple[Union[_AugmentationBase, ImageSequential], ...]:
        # if changed in the future, please change the expected transforms list in test_presets.py
        return (
            RandomHorizontalFlip(p=1),
            RandomRotation90(times=(0, 3), p=1.0),
            RandomErasing(scale=scale, ratio=ratio, value=value, p=0.9),
            RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), p=1.0),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            RandomGaussianNoise(std=0.1, p=1.0),
        )

    def update(self, real_acc: float) -> None:
        r"""Updates internal params `p` once every `update_every` calls based on discriminator accuracy.

        the update is based on an exponential moving average of `real_acc`
        `p` is updated by adding or subtracting `adjustment_speed` from it and clamp it at [0, `max_p`]

        Args:
            real_acc: the Discriminator's accuracy labeling real samples.
        """
        self._num_calls += 1

        if self._num_calls < self.update_every:
            return
        self._num_calls = 0

        self.real_acc_ema = self.ema_lambda * self.real_acc_ema + (1 - self.ema_lambda) * real_acc

        if self.real_acc_ema < self.target_real_acc:
            self.p = max(0, self.p - self.adjustment_speed)
        else:
            self.p = min(self.p + self.adjustment_speed, self.max_p)

    def _get_inputs_metadata(self, inputs: _inputs_type, data_keys: _data_keys_type) -> Tuple[int, Device]:
        if isinstance(inputs, dict):
            key = data_keys[0]
            batch_size = inputs[key].size(0)
            device = inputs[key].device
        else:
            batch_size = inputs.size(0)
            device = inputs.device

        return batch_size, device

    def _sample_inputs(self, inputs: _inputs_type, data_keys: _data_keys_type, p_tensor: Tensor) -> _inputs_type:
        if isinstance(inputs, dict):
            return {key: inputs[key][p_tensor] for key in data_keys}
        else:
            return inputs[p_tensor]

    def _merge_inputs(
        self,
        original: _inputs_type,
        augmented: _inputs_type,
        p_tensor: Tensor,
    ) -> _inputs_type:
        merged: _inputs_type
        if isinstance(original, dict) and isinstance(augmented, dict):
            merged = {}
            for key in original.keys():
                merged_tensor = original[key].clone()
                merged_tensor[p_tensor] = augmented[key]
                merged[key] = merged_tensor
        elif isinstance(original, Tensor) and isinstance(augmented, Tensor):
            merged = original.clone()
            merged[p_tensor] = augmented
        else:
            raise TypeError(
                f"original inputs and augmented inputs aren't of the same type "
                f"(type({type(original)}), type({type(augmented)}))"
            )
        return merged

    def forward(  # type: ignore[override]
        self,
        inputs: _inputs_type,
        params: Optional[List[ParamItem]] = None,
        data_keys: Optional[_data_keys_type] = None,
        real_acc: Optional[float] = None,
    ) -> _inputs_type:
        r"""Apply augmentations to a subset of input tensors with global probability `p`.

        This method applies the augmentation pipeline to a subset of input samples, randomly selected
        via a Bernoulli distribution with probability `p`

        if `real_acc` is provided, the internal probability `p` is updated via the `update` method.
        Non-augmented samples retain their original values, and the output matches the input structure.

        `real_acc` is the Discriminator's accuracy on real images; for example,
        `(real_logits > 0).float().mean().item()` if using logits andn assuming real labels are positive.
        """
        if real_acc is not None:
            self.update(real_acc)

        if self.p == 0:
            return inputs

        if data_keys is None:
            data_keys = (
                [k.name for k in self.data_keys]
                if self.data_keys is not None
                else [
                    "input",
                ]
            )

        batch_size, device = self._get_inputs_metadata(inputs, data_keys=data_keys)

        p_tensor = torch.bernoulli(torch.full((batch_size,), self.p, dtype=torch.float32, device=device)).bool()

        if not p_tensor.any():
            return inputs

        selected_inputs: _inputs_type = self._sample_inputs(inputs, data_keys=data_keys, p_tensor=p_tensor)
        augmented_inputs = cast(
            _inputs_type,
            super().forward(
                selected_inputs,  # type: ignore[arg-type]
                params=params,
                data_keys=data_keys,
            ),
        )

        return self._merge_inputs(inputs, augmented_inputs, p_tensor)
