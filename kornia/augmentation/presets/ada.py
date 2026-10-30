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

from kornia.constants import DataKey

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
from ..container.ops import DataType  # noqa: TID252
from ..container.params import ParamItem  # noqa: TID252

_data_keys_type = List[str]
_inputs_type = Union[torch.Tensor, Dict[str, torch.Tensor]]


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


    .. note::
        The default augmentation stack contains ``RandomRotation90(times=(0, 3))``, which raises ``ValueError`` on a
        non-square image, so the default needs square images. ADA checks the image against it on every call, before
        it selects samples, so a non-square image raises on the first call. In a custom stack, a
        :class:`~kornia.augmentation.RandomRotation90` that may turn an image by an odd number of quarter turns needs
        the image it receives to be square, after any earlier resize, crop or patch split, and raises when ADA first
        selects a sample.

    Examples:
        >>> from kornia.augmentation.presets.ada import AdaptiveDiscriminatorAugmentation
        >>> original = torch.randn(2, 3, 16, 16)
        >>> ada = AdaptiveDiscriminatorAugmentation()
        >>> augmented = ada(original)

    This example demonstrates using default augmentations with AdaptiveDiscriminatorAugmentation in a GAN training loop.


        >>> import kornia.augmentation as K
        >>> from kornia.augmentation.presets.ada import AdaptiveDiscriminatorAugmentation
        >>> originals = torch.randn(2, 3, 6, 6)
        >>> aug_list = [
        ...     K.RandomRotation90(times=(0, 3), p=1),
        ...     K.RandomAffine(degrees=10, translate=(.1, .1), scale=(.9, 1.1), p=1),
        ...     K.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.1, p=1),
        ... ]

        >>> ada = AdaptiveDiscriminatorAugmentation(*aug_list)
        >>> augmented = ada(originals)

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
        erasing_scale: Union[torch.Tensor, Tuple[float, float]] = (0.02, 0.33),
        erasing_ratio: Union[torch.Tensor, Tuple[float, float]] = (0.3, 3.3),
        erasing_fill_value: float = 0.0,
        data_keys: Optional[_data_keys_type] = None,
        same_on_batch: Optional[bool] = False,
        **kwargs: Any,
    ) -> None:
        default_stack = not args
        if default_stack:
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

        # ADA built the default stack, so it knows the stack's RandomRotation90 receives the input's (H, W).
        self._default_stack = default_stack

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
        self,
        scale: Union[torch.Tensor, Tuple[float, float]],
        ratio: Union[torch.Tensor, Tuple[float, float]],
        value: float,
    ) -> Tuple[Union[_AugmentationBase, ImageSequential], ...]:
        """Return the default augmentation stack used by ADA.

        Args:
            scale: Erase-area range passed to :class:`RandomErasing`.
            ratio: Erase-aspect-ratio range passed to :class:`RandomErasing`.
            value: Fill value used by :class:`RandomErasing`.

        Returns:
            Tuple of augmentation modules applied inside ADA.
        """
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
        r"""Update internal params `p` once every `update_every` calls based on discriminator accuracy.

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

    def _preproc_dict_data(
        self, data: Dict[str, DataType]
    ) -> Tuple[Tuple[str, ...], List[DataKey], Tuple[DataType, ...], Optional[Dict[str, Any]]]:
        # ``data_keys`` describe a tensor input, and ADA always sets them, so the parent's rule that a dictionary needs
        # ``data_keys=None`` would reject every dictionary. A dictionary names its own entries instead.
        keys = tuple(data.keys())
        data_keys, invalid_keys = self._read_datakeys_from_dict(keys)
        invalid_data = {k: data[k] for k in invalid_keys} if invalid_keys else None
        keys = tuple(k for k in keys if k not in invalid_keys) if invalid_keys else keys
        return keys, data_keys, tuple(data[k] for k in keys), invalid_data

    def _dict_keys(self, inputs: Dict[str, torch.Tensor], data_keys: Optional[_data_keys_type]) -> _data_keys_type:
        """Return the augmentable keys of a dictionary input, the image key first."""
        keys, dict_data_keys, _, _ = self._preproc_dict_data(cast(Dict[str, DataType], inputs))
        if not keys:
            raise ValueError(f"No key of the dictionary input names a data key. Got {list(inputs)}.")
        # The dictionary sets the order, so only the data keys themselves have to agree.
        if data_keys is not None and sorted(DataKey.get(k).value for k in data_keys) != sorted(
            k.value for k in dict_data_keys
        ):
            raise ValueError(
                f"A dictionary input names its own data keys, so `data_keys` must match them, in any order, or be "
                f"None. Got {data_keys} for the keys {list(keys)}."
            )
        order = sorted(range(len(keys)), key=lambda i: dict_data_keys[i] != DataKey.INPUT)
        return [keys[i] for i in order]

    def _validate_rotation90_inputs(self, inputs: _inputs_type) -> None:
        # Only the default stack: in a custom one an earlier module can resize the image, or a container can cut it into
        # patches or frames, so the input's (H, W) is not the one its RandomRotation90 receives. There the rotation
        # checks the tensor it receives, when ADA first selects a sample. An ADA pickled before this check has no flag.
        if not getattr(self, "_default_stack", False):
            return
        if isinstance(inputs, dict):
            keys, dict_data_keys, _, _ = self._preproc_dict_data(cast(Dict[str, DataType], inputs))
            images = [inputs[key] for key, data_key in zip(keys, dict_data_keys) if data_key == DataKey.INPUT]
        else:
            images = [inputs]
        rotations = [module for module in self.modules() if isinstance(module, RandomRotation90)]
        for image in images:
            if image.dim() == 4:
                for rotation in rotations:
                    rotation.validate_tensor(image)

    def _get_inputs_metadata(
        self, inputs: _inputs_type, data_keys: _data_keys_type
    ) -> Tuple[int, Optional[Union[str, torch.device]]]:
        if isinstance(inputs, dict):
            key = data_keys[0]
            batch_size = inputs[key].size(0)
            device = inputs[key].device
        else:
            batch_size = inputs.size(0)
            device = inputs.device

        return batch_size, device

    def _sample_inputs(self, inputs: _inputs_type, data_keys: _data_keys_type, p_tensor: torch.Tensor) -> _inputs_type:
        if isinstance(inputs, dict):
            return {key: inputs[key][p_tensor] for key in data_keys}

        return inputs[p_tensor]

    def _merge_inputs(
        self,
        original: _inputs_type,
        augmented: _inputs_type,
        p_tensor: torch.Tensor,
    ) -> _inputs_type:
        merged: _inputs_type
        if isinstance(original, dict) and isinstance(augmented, dict):
            # Entries that name no data key were not augmented and pass through unchanged.
            merged = dict(original)
            for key in augmented:
                merged_tensor = original[key].clone()
                merged_tensor[p_tensor] = augmented[key]
                merged[key] = merged_tensor
        elif isinstance(original, torch.Tensor) and isinstance(augmented, torch.Tensor):
            merged = original.clone()
            merged[p_tensor] = augmented
        else:
            raise TypeError(
                f"original inputs and augmented inputs aren't of the same type "
                f"(type({type(original)}), type({type(augmented)}))"
            )
        return merged

    def forward(
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

        A dictionary input names its own data keys, as with :class:`~kornia.augmentation.AugmentationSequential`,
        whatever ``data_keys`` the preset was built with; ``data_keys`` passed here must match them, in any order, or
        be None. The same samples are augmented in every entry, and an entry whose key names no data key is returned
        unchanged.

        `real_acc` is the Discriminator's accuracy on real images; for example,
        `(real_logits > 0).float().mean().item()` if using logits andn assuming real labels are positive.
        """
        if real_acc is not None:
            self.update(real_acc)

        # ADA only calls its augmentations for the samples its probability selects, which at a small `p` can be many
        # calls in. Check the input here, on every call, so a shape the default `RandomRotation90` rejects fails on the
        # first one.
        self._validate_rotation90_inputs(inputs)

        if self.p == 0:
            return inputs

        if isinstance(inputs, dict):
            # Each entry is row-selected and merged by its own key; the parent reads the data keys from the names.
            data_keys = self._dict_keys(inputs, data_keys)
            forward_data_keys = None
        else:
            if data_keys is None:
                data_keys = [k.name for k in self.data_keys] if self.data_keys is not None else ["input"]
            forward_data_keys = data_keys

        batch_size, device = self._get_inputs_metadata(inputs, data_keys=data_keys)

        sample_size = min(batch_size, 1) if self.same_on_batch else batch_size
        p_tensor = (
            torch.bernoulli(torch.full((sample_size,), self.p, dtype=torch.float32, device=device))
            .bool()
            .expand(batch_size)
        )

        if not p_tensor.any():
            return inputs

        selected_inputs: _inputs_type = self._sample_inputs(inputs, data_keys=data_keys, p_tensor=p_tensor)
        augmented_inputs = cast(
            _inputs_type,
            super().forward(
                selected_inputs,
                params=params,
                data_keys=forward_data_keys,
            ),
        )

        return self._merge_inputs(inputs, augmented_inputs, p_tensor)
