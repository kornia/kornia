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

import kornia.augmentation as K
from kornia.augmentation import ImageSequential
from kornia.augmentation.base import _AugmentationBase
from kornia.augmentation.container.params import ParamItem
from kornia.core import Device, Tensor

_data_keys_type = List[str]
_inputs_type = Union[Tensor, Dict[str, Tensor]]


class AdaptiveDiscriminatorAugmentation(K.AugmentationSequential):
    r"""Implementation of Adaptive Discriminator Augmentation (ADA) for GANs training.

    adjust a global probability p over all augmentations list to select a subset of images to augment
    based on an exponential moving average of the Discriminator's accuracy labeling real samples.

    Args:
        *args: a list of kornia augmentation modules, set to a default list if not specified.

        initial_p: initial global probability `p` for applying the augmentations on

        adjustment_speed: step size for updating the global probability `p`

        max_p: the maximum value to clamp `p` at

        target_real_acc: target Discriminator accuracy to prevent overfitting

        ema_lambda: EMA smoothing factor to compute the $
        \mathrm{ema_real_accuracy} = \lambda_\text{EMA} * mathrm{real_accuracy} +
        (1 - \lambda_\text{EMA}) * mathrm{real_accuracy}
        $

        update_every: `p` update frequency

        crop_size: the used in the `RandomCrop` default augmentation

        same_on_batch: apply the same transformation across the batch.
        If None, it will not overwrite the function-wise settings.

        data_keys: the input type sequential for applying augmentations. Accepts "input", "image", "mask",
            "bbox", "bbox_xyxy", "bbox_xywh", "keypoints", "class", "label".

        **kwargs: the rest of the `kwargs` passed to the `AugmentationSequential` attribute containing augmentation

    Examples:
        >>> from kornia.augmentation.presets.ada import AdaptiveDiscriminatorAugmentation
        >>> original = torch.randn(2, 3, 5, 6)
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
        crop_size: Tuple[int, int] = (64, 64),
        data_keys: Optional[_data_keys_type] = None,
        same_on_batch: Optional[bool] = False,
        **kwargs: Any,
    ) -> None:
        if not args:
            args = (
                K.RandomHorizontalFlip(p=1),
                K.RandomRotation90(times=(0, 3), p=1.0),
                K.RandomCrop(size=crop_size, padding=0, p=1.0),
                K.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), p=1.0),
                K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
                K.RandomGaussianNoise(std=0.1, p=1.0),
            )
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
        self.num_calls = -update_every  # to avoid updating in the first `update_every` steps

    def update(self, real_acc: float) -> None:
        r"""Updates internal params `p` once every `update_every` calls.

        `p` is updated based on the `real_acc` arg by adding / subtracting the `adjustment_speed`
        from it and clamp it at [0, `max_p`] increment the internal counter `num_calls` by 1 on each call.

        Args:
            real_acc: the Discriminator's accuracy labeling real samples.
        """
        self.num_calls += 1

        if self.num_calls < self.update_every:
            return
        self.num_calls = 0

        self.real_acc_ema = self.ema_lambda * self.real_acc_ema + (1 - self.ema_lambda) * real_acc

        if self.real_acc_ema < self.target_real_acc:
            self.p = max(0, self.p - self.adjustment_speed)
        else:
            self.p = min(self.p + self.adjustment_speed, self.max_p)

    def _get_inputs_metadata(self, *inputs: _inputs_type, data_keys: _data_keys_type) -> Tuple[int, Device]:
        if isinstance(inputs[0], dict):
            key = data_keys[0]
            batch_size = inputs[0][key].size(0)
            device = inputs[0][key].device
        else:
            batch_size = inputs[0].size(0)
            device = inputs[0].device

        return batch_size, device

    def _sample_inputs(self, *inputs: _inputs_type, data_keys: _data_keys_type, P: Tensor) -> _inputs_type:
        if isinstance(inputs[0], dict):
            return {key: inputs[0][key][P] for key in data_keys}
        else:
            return inputs[P]

    def _merge_inputs(
        self,
        original: _inputs_type,
        augmented: _inputs_type,
        P: Tensor,
    ) -> _inputs_type:
        merged: _inputs_type
        if isinstance(original, dict) and isinstance(augmented, dict):
            merged = {}
            for key in original.keys():
                merged_tensor = original[key].clone()
                merged_tensor[P] = augmented[key]
                merged[key] = merged_tensor
        elif isinstance(original, Tensor) and isinstance(augmented, Tensor):
            merged = original.clone()
            merged[P] = augmented
        else:
            raise Exception(
                "original inputs and augmented inputs aren't of the same type "
                "(type({type(original)}), type({type(augmented)}))"
            )
        return merged

    def forward(  # type: ignore[override]
        self,
        *inputs: _inputs_type,
        params: Optional[List[ParamItem]] = None,
        data_keys: Optional[_data_keys_type] = None,
        real_acc: Optional[float] = None,
    ) -> _inputs_type:
        r"""Apply augmentations to a subset of input tensors with global probability `p`.

        This method applies the augmentation pipeline to a subset of input samples, selected stochastically
        using a Bernoulli distribution with probability $p :math:`P_i \\sim \\text{Bernoulli}(p)`,
        where :math:`P_i` is a boolean mask for the :math:`i`-th
        if `real_acc` is provided, the internal probability `p` is updated via the `update` method.
        Non-augmented samples retain their original values, and the output matches the input structure.
        """
        if real_acc is not None:
            self.update(real_acc)

        if self.p == 0:
            return inputs[0]

        if data_keys is None:
            data_keys = (
                [str(k) for k in self.data_keys]
                if self.data_keys is not None
                else [
                    "input",
                ]
            )

        # assert data_keys is not None, "data_keys is None"

        batch_size, device = self._get_inputs_metadata(*inputs, data_keys=data_keys)

        P = torch.bernoulli(torch.full((batch_size,), self.p, device=device)).bool()

        if not P.any():
            return inputs[0]

        selected_inputs: _inputs_type = self._sample_inputs(*inputs, data_keys=data_keys, P=P)
        augmented_inputs = cast(_inputs_type, super().forward(*selected_inputs, params=params, data_keys=data_keys))

        return self._merge_inputs(inputs[0], augmented_inputs, P)
